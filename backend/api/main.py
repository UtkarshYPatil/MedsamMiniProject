"""
Nodule measurement API.

    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

Segmentation is never run inside a request. A request enqueues a job and
returns immediately; the client polls. Inference takes about a second on a
GPU and up to a minute on CPU, and an HTTP request that waits for either is a
request that times out behind a proxy.

This container has no torch and no model weights. It only moves bytes and
rows; the worker does the computing.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.job import Job as RQJob
from sqlalchemy.orm import Session

from backend.api.db import Case, Job, SessionLocal, init_db
from backend.common import autoprompt, dicomseries, roi, settings, store

TASK = "backend.worker.tasks.segment_case"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # on_event("startup") is deprecated in current FastAPI; lifespan is the
    # supported hook and also runs on shutdown if that is ever needed.
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Nodule measurement assistant",
    version="0.4.0",
    description=(
        "Measures pulmonary nodules from a prompted 3D segmentation. "
        "It does not diagnose: every number is reported with the mask it came "
        "from and a confidence flag, and interpretation stays with the "
        "radiologist. Research and education use only."
    ),
)
# "*" is fine for a research tool on localhost and wrong for anything exposed.
# CORS_ORIGINS lets a deployment narrow it without a code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_redis = Redis.from_url(settings.REDIS_URL)
queue = Queue(settings.QUEUE_NAME, connection=_redis,
              default_timeout=settings.JOB_TIMEOUT)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------ schemas

class SegmentRequest(BaseModel):
    slice: int = Field(..., ge=0, description="Index of the slice the box is drawn on")
    box: list[float] = Field(..., min_length=4, max_length=4,
                             description="[x0, y0, x1, y1] in pixels on that slice")
    propagate: bool | None = Field(
        None, description="Carry the mask through the volume (default: server setting)")


class PhantomRequest(BaseModel):
    diameter_mm: float = Field(9.0, gt=1.0, le=40.0)
    note: str = Field("", max_length=256)


# ------------------------------------------------------------------- health

@app.get("/healthz")
def healthz():
    """
    What the stack can actually do right now.

    `config` is this container's own settings. `worker` is what the worker
    reported about itself, which is the only trustworthy answer about the
    device: this container has no torch and cannot tell a GPU from a toaster.

    `worker_live` is the signal to trust, because the worker refreshes that
    key on a timer and it expires if nothing refreshes it. `workers_alive`
    comes from RQ's registry, which is written ONCE when a worker starts: a
    Redis restart erases it and a healthy worker then reports as absent for
    as long as it keeps running. It is kept as a diagnostic, not as the
    answer to "can anything drain this queue".
    """
    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    worker = None
    try:
        raw = _redis.get(settings.WORKER_STATUS_KEY)
        if raw:
            worker = json.loads(raw)
    except Exception:
        worker = None

    try:
        from rq import Worker
        alive = Worker.count(connection=_redis, queue=queue)
    except Exception:
        alive = None

    return {
        "ok": True,
        "redis": redis_ok,
        "queued": len(queue),
        "workers_alive": alive,
        # The heartbeat, not the registry: see the docstring.
        "worker_live": worker is not None,
        "worker": worker,
        "config": settings.summary(),
        # Kept so an older UI build does not break on a new API.
        "workers": len(queue),
    }


# -------------------------------------------------------------------- cases

@app.post("/api/cases", status_code=201)
async def create_case(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload one ROI volume as .npz with image_hu (Z, Y, X) and spacing (z, y, x).

    The files scripts/02_prepare_data.py writes are accepted as-is.
    """
    # Read with a cap instead of trusting the client: an unbounded read is
    # how a service gets killed by one request.
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                413, f"upload exceeds {settings.MAX_UPLOAD_MB} MB. A case is one "
                     f"ROI crop (about 1 MB), not a whole series.")
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
        image, spacing = store.read_npz_upload(raw)
    except Exception as e:
        raise HTTPException(400, f"could not read the uploaded npz: {e}")

    case_id = store.save_case(image, spacing, source="upload")
    row = Case(id=case_id, source="upload", note=file.filename or "",
               shape_z=image.shape[0], shape_y=image.shape[1], shape_x=image.shape[2],
               spacing_z=float(spacing[0]), spacing_y=float(spacing[1]),
               spacing_x=float(spacing[2]))
    db.add(row)
    db.commit()
    return row.as_dict()


@app.post("/api/cases/dicom", status_code=201)
async def create_case_dicom(file: UploadFile = File(...),
                            db: Session = Depends(get_db)):
    """
    Upload a CT series as a .zip of DICOM files.

    The series is converted to a Hounsfield-unit volume: slices ordered by
    scanner geometry, RescaleSlope/Intercept applied, spacing taken from the
    gap between slice positions. See backend/common/dicomseries.py for why
    each of those matters.

    Only pixels and geometry are read. No patient identifiers are returned or
    stored. This is research and education software on public de-identified
    data; real patient scans do not belong in it.

    The nodule is NOT located for you. A whole scan has no centre to assume,
    so find it on the viewer and draw the box; the worker then crops the
    validated ROI around wherever you put it.
    """
    limit = settings.MAX_DICOM_MB * 1024 * 1024
    chunks, total = [], 0
    while True:
        chunk = await file.read(4 * 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                413, f"upload exceeds {settings.MAX_DICOM_MB} MB. Zip one "
                     f"series, not a whole study.")
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
        image, spacing, info = dicomseries.read_zip(raw)
    except dicomseries.DicomError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"could not read the DICOM series: {e}")

    case_id = store.save_case(image, spacing, source="dicom")
    row = Case(id=case_id, source="dicom",
               note=dicomseries.describe(info),
               shape_z=image.shape[0], shape_y=image.shape[1], shape_x=image.shape[2],
               spacing_z=float(spacing[0]), spacing_y=float(spacing[1]),
               spacing_x=float(spacing[2]))
    db.add(row)
    db.commit()
    return {**row.as_dict(), "dicom": info}


@app.post("/api/cases/phantom", status_code=201)
def create_phantom(req: PhantomRequest, db: Session = Depends(get_db)):
    """
    Create a synthetic case so the stack can be demonstrated without the
    125 GB dataset. Marked source=phantom; never a patient.
    """
    image, spacing = store.make_phantom(diameter_mm=req.diameter_mm)
    case_id = store.save_case(image, spacing, source="phantom")
    row = Case(id=case_id, source="phantom",
               note=req.note or f"synthetic sphere, {req.diameter_mm:g} mm",
               shape_z=image.shape[0], shape_y=image.shape[1], shape_x=image.shape[2],
               spacing_z=float(spacing[0]), spacing_y=float(spacing[1]),
               spacing_x=float(spacing[2]))
    db.add(row)
    db.commit()
    return row.as_dict()


@app.get("/api/cases")
def list_cases(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    rows = db.query(Case).order_by(Case.created_at.desc()).limit(limit).all()
    return {"cases": [r.as_dict() for r in rows]}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str, db: Session = Depends(get_db)):
    row = db.get(Case, case_id)
    if not row:
        raise HTTPException(404, "no such case")
    return row.as_dict()


@app.get("/api/cases/{case_id}/suggest")
def suggest_prompt(case_id: str, db: Session = Depends(get_db)):
    """
    Propose a prompt box, so the common case is one button press.

    A proposal, not a finding: it assumes the nodule is roughly central, which
    holds for a centroid-cropped ROI and not for an arbitrary scan. The caller
    can always override it, and `info` says how the box was arrived at so the
    proposal can be questioned rather than trusted.
    """
    if not db.get(Case, case_id):
        raise HTTPException(404, "no such case")
    try:
        vol, _ = store.load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "case record exists but its volume is missing")

    # The auto-prompt scores candidates by how central they are, which holds
    # for a crop cut around a nodule centroid and is meaningless on a whole
    # chest CT -- the middle of that is mediastinum. Refusing is the honest
    # answer; a confident box on the heart would be worse than none.
    if roi.needs_crop(vol.shape, settings.ROI_SIZE, settings.ROI_DEPTH):
        return {
            "slice": int(vol.shape[0] // 2),
            "box": None,
            "source": "unavailable",
            "info": {
                "reason": "whole-series upload",
                "detail": ("Automatic prompting assumes the nodule is near the "
                           "centre of a cropped ROI. This volume is a full "
                           "series, so find the nodule on the viewer and draw "
                           "the box yourself. The tool measures what you point "
                           "it at; it does not detect nodules."),
                "shape": [int(v) for v in vol.shape],
            },
        }

    z, box, info = autoprompt.suggest_box(vol)
    if box is None:
        z, box = autoprompt.centred_fallback(vol.shape)
        return {"slice": z, "box": box, "source": "centred_fallback", "info": info}
    return {"slice": z, "box": box, "source": "auto", "info": info}


@app.get("/api/cases/{case_id}/slice/{z}.png")
def case_slice(case_id: str, z: int, db: Session = Depends(get_db)):
    """The windowed slice as a PNG, for the viewer."""
    if not db.get(Case, case_id):
        raise HTTPException(404, "no such case")
    try:
        vol, _ = store.load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(404, "case record exists but its volume is missing")
    if not 0 <= z < vol.shape[0]:
        raise HTTPException(400, f"slice {z} out of range 0..{vol.shape[0] - 1}")
    return Response(store.render_png(vol[z]), media_type="image/png")


# --------------------------------------------------------------------- jobs

@app.post("/api/cases/{case_id}/segment", status_code=202)
def submit_segment(case_id: str, req: SegmentRequest, db: Session = Depends(get_db)):
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "no such case")
    if not 0 <= req.slice < case.shape_z:
        raise HTTPException(400, f"slice {req.slice} out of range 0..{case.shape_z - 1}")

    x0, y0, x1, y1 = req.box
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(400, "box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0")

    propagate = settings.PROPAGATE if req.propagate is None else req.propagate
    rq_job = queue.enqueue(TASK, case_id, int(req.slice), [float(v) for v in req.box],
                           propagate, job_timeout=settings.JOB_TIMEOUT)

    row = Job(id=rq_job.id, case_id=case_id, prompt_z=int(req.slice),
              box=[float(v) for v in req.box], status="queued",
              device=settings.DEVICE, dry_run=settings.DRY_RUN)
    db.add(row)
    db.commit()
    return {**row.as_dict(), "queue_position": len(queue)}


def _sync_job(db: Session, row: Job) -> Job:
    """Reconcile the durable row with RQ's live view of the job."""
    if row.status in ("done", "failed"):
        return row
    try:
        rq_job = RQJob.fetch(row.id, connection=_redis)
    except Exception:
        return row                      # result expired from Redis; row stands

    state = rq_job.get_status(refresh=True)
    if state == "finished":
        res = rq_job.return_value() or {}
        row.status = "done"
        row.result = res
        row.seconds = res.get("seconds")
        row.device = res.get("device", row.device)
        row.dry_run = bool(res.get("dry_run", row.dry_run))
        row.finished_at = datetime.now(timezone.utc)
    elif state == "failed":
        row.status = "failed"
        row.error = (rq_job.exc_info or "worker failed")[-2000:]
        row.finished_at = datetime.now(timezone.utc)
    elif state == "started":
        row.status = "running"
    db.add(row)
    db.commit()
    return row


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    row = db.get(Job, job_id)
    if not row:
        raise HTTPException(404, "no such job")
    return _sync_job(db, row).as_dict()


@app.get("/api/jobs/{job_id}/overlay.png")
def job_overlay(job_id: str, db: Session = Depends(get_db)):
    """The prompted slice with the predicted outline and the prompt box."""
    row = db.get(Job, job_id)
    if not row:
        raise HTTPException(404, "no such job")
    row = _sync_job(db, row)
    if row.status != "done":
        raise HTTPException(409, f"job is {row.status}, not done")
    try:
        mask, prompt_z, box = store.load_mask(job_id)
        vol, _ = store.load_case(row.case_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    z = int(np.clip(prompt_z, 0, vol.shape[0] - 1))
    return Response(store.render_png(vol[z], mask[z], box), media_type="image/png")


@app.get("/api/cases/{case_id}/jobs")
def case_jobs(case_id: str, db: Session = Depends(get_db)):
    rows = db.query(Job).filter(Job.case_id == case_id) \
             .order_by(Job.created_at.desc()).all()
    return {"jobs": [_sync_job(db, r).as_dict() for r in rows]}
