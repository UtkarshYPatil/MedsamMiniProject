"""
Nodule measurement API.

    uvicorn services.api.main:app --host 0.0.0.0 --port 8000

Segmentation is never run inside a request. A request enqueues a job and
returns immediately; the client polls. Inference takes about a second on a
GPU and up to a minute on CPU, and an HTTP request that waits for either is a
request that times out behind a proxy.

This container has no torch and no model weights. It only moves bytes and
rows; the worker does the computing.
"""
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

from services.api.db import Case, Job, SessionLocal, init_db
from services.common import settings, store

TASK = "services.worker.tasks.segment_case"

app = FastAPI(
    title="Nodule measurement assistant",
    version="0.3.0",
    description=(
        "Measures pulmonary nodules from a prompted 3D segmentation. "
        "It does not diagnose: every number is reported with the mask it came "
        "from and a confidence flag, and interpretation stays with the "
        "radiologist. Research and education use only."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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


@app.on_event("startup")
def _startup():
    settings.ensure_dirs()
    init_db()


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
    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"ok": True, "redis": redis_ok, "workers": len(queue), "config": settings.summary()}


# -------------------------------------------------------------------- cases

@app.post("/api/cases", status_code=201)
async def create_case(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload one ROI volume as .npz with image_hu (Z, Y, X) and spacing (z, y, x).

    The files scripts/02_prepare_data.py writes are accepted as-is.
    """
    raw = await file.read()
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
