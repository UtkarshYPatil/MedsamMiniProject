"""
The API over real HTTP, with no Docker, no Postgres and no Redis.

tests/test_backend.py tests the modules. This file tests the ROUTES: status
codes, payload shapes and the request-level guards. Those only exist in the
wiring, so a module test cannot see them -- and the wiring is exactly what
broke in this project before (an endpoint handing the browser a URL only
resolvable inside the Docker network).

Postgres is swapped for SQLite and Redis for fakeredis, both in-process, so
this runs anywhere `pytest` runs.

    python -m pytest tests/test_api_http.py -v
"""
import importlib
import io

import numpy as np
import pytest

_NEED = "pip install -r backend/requirements-dev.txt"
fakeredis = pytest.importorskip("fakeredis", reason=_NEED)
pytest.importorskip("httpx", reason=_NEED)
pytest.importorskip("fastapi", reason=_NEED)
pytest.importorskip("sqlalchemy", reason=_NEED)
pytest.importorskip("rq", reason=_NEED)
pytest.importorskip("multipart", reason=_NEED)   # FastAPI needs it for uploads

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A live app on SQLite + fakeredis, pointed at a scratch data directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{tmp_path/'t.db'}")
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")        # keep the cap test cheap

    from backend.common import settings, store
    importlib.reload(settings)
    importlib.reload(store)

    from backend.api import db
    importlib.reload(db)

    # RQ must not reach a real Redis. Jobs are enqueued but never run; the
    # worker task itself is covered in tests/test_backend.py.
    import redis
    monkeypatch.setattr(redis.Redis, "from_url",
                        classmethod(lambda cls, *a, **k: fakeredis.FakeStrictRedis()))

    from backend.api import main
    importlib.reload(main)

    with TestClient(main.app) as c:
        yield c


def _npz(shape=(16, 48, 48), spacing=(1.0, 0.7, 0.7)):
    """An .npz in the format scripts/02_prepare_data.py writes."""
    vol = np.full(shape, -800, dtype=np.int16)
    z, y, x = (s // 2 for s in shape)
    vol[z - 2:z + 3, y - 4:y + 5, x - 4:x + 5] = 60      # a small central blob
    buf = io.BytesIO()
    np.savez_compressed(buf, image_hu=vol, spacing=np.asarray(spacing, dtype=np.float32))
    return buf.getvalue()


# ------------------------------------------------------------------ health

def test_healthz_reports_configuration(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "device" in body["config"]
    assert body["config"]["dry_run"] is True


# ------------------------------------------------------------------- cases

def test_phantom_case_is_created_and_listed(client):
    r = client.post("/api/cases/phantom", json={"diameter_mm": 9.0})
    assert r.status_code == 201, r.text
    case = r.json()
    assert case["source"] == "phantom"
    assert len(case["shape"]) == 3

    listed = client.get("/api/cases").json()["cases"]
    assert [c["case_id"] for c in listed] == [case["case_id"]]


def test_upload_round_trips_a_prepare_data_npz(client):
    r = client.post("/api/cases",
                    files={"file": ("roi.npz", _npz(), "application/octet-stream")})
    assert r.status_code == 201, r.text
    assert r.json()["shape"] == [16, 48, 48]


def test_upload_over_the_cap_is_refused_with_413(client):
    """
    The cap must be enforced while READING, not after. A service that buffers
    the whole body before checking has already lost.
    """
    big = b"\0" * (2 * 1024 * 1024)             # 2 MB against a 1 MB cap
    r = client.post("/api/cases",
                    files={"file": ("big.npz", big, "application/octet-stream")})
    assert r.status_code == 413, r.status_code
    assert "exceeds" in r.json()["detail"]


def test_upload_of_a_non_npz_is_rejected_cleanly(client):
    r = client.post("/api/cases",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "npz" in r.json()["detail"].lower()


def test_unknown_case_is_404_everywhere(client):
    for path in ("/api/cases/nope", "/api/cases/nope/suggest",
                 "/api/cases/nope/slice/0.png"):
        assert client.get(path).status_code == 404, path


# ---------------------------------------------------------------- suggest

def test_suggest_returns_a_usable_prompt(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    r = client.get(f"/api/cases/{case['case_id']}/suggest")
    assert r.status_code == 200, r.text

    s = r.json()
    assert s["source"] in ("auto", "centred_fallback")
    assert 0 <= s["slice"] < case["shape"][0]
    x0, y0, x1, y1 = s["box"]
    assert x1 > x0 and y1 > y0, "a degenerate box would be rejected by /segment"

    # The suggestion must be directly acceptable by the segment endpoint --
    # the whole point is that the user presses one button.
    sub = client.post(f"/api/cases/{case['case_id']}/segment",
                      json={"slice": s["slice"], "box": s["box"]})
    assert sub.status_code == 202, sub.text


def test_suggest_explains_itself(client):
    """`info` exists so a proposal can be questioned rather than trusted."""
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    info = client.get(f"/api/cases/{case['case_id']}/suggest").json()["info"]
    assert isinstance(info, dict) and info, "a bare proposal with no reasoning"


# ----------------------------------------------------------------- slices

def test_slice_endpoint_returns_a_png(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    r = client.get(f"/api/cases/{case['case_id']}/slice/0.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_slice_out_of_range_is_400_not_500(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    z = case["shape"][0] + 10
    r = client.get(f"/api/cases/{case['case_id']}/slice/{z}.png")
    assert r.status_code == 400
    assert "out of range" in r.json()["detail"]


# ------------------------------------------------------------------- jobs

def test_segment_validates_the_prompt_before_queueing(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    cid = case["case_id"]

    # inverted box
    r = client.post(f"/api/cases/{cid}/segment",
                    json={"slice": 5, "box": [40, 40, 10, 10]})
    assert r.status_code == 400

    # slice past the end of the volume
    r = client.post(f"/api/cases/{cid}/segment",
                    json={"slice": case["shape"][0] + 5, "box": [10, 10, 40, 40]})
    assert r.status_code == 400

    # wrong number of coordinates -- caught by the schema
    r = client.post(f"/api/cases/{cid}/segment",
                    json={"slice": 5, "box": [10, 10, 40]})
    assert r.status_code == 422


def test_submitting_a_job_records_it(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    cid = case["case_id"]
    job = client.post(f"/api/cases/{cid}/segment",
                      json={"slice": 16, "box": [40, 40, 88, 88]}).json()

    assert job["status"] == "queued"
    assert job["case_id"] == cid

    fetched = client.get(f"/api/jobs/{job['job_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == job["job_id"]

    listed = client.get(f"/api/cases/{cid}/jobs").json()["jobs"]
    assert [j["job_id"] for j in listed] == [job["job_id"]]


def test_overlay_is_409_while_the_job_is_unfinished(client):
    """Not 500, and not a blank image pretending to be a result."""
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    job = client.post(f"/api/cases/{case['case_id']}/segment",
                      json={"slice": 16, "box": [40, 40, 88, 88]}).json()

    r = client.get(f"/api/jobs/{job['job_id']}/overlay.png")
    assert r.status_code == 409
    assert "not done" in r.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/overlay.png").status_code == 404


# ----------------------------------------------------- health, in more detail

def test_healthz_reports_no_worker_when_none_is_running(client):
    """
    A queue filling up with nobody draining it must look different from a slow
    job. This is the signal the UI turns into a red banner.
    """
    body = client.get("/healthz").json()
    assert body["workers_alive"] == 0
    assert body["worker"] is None, "no worker has reported, so there is nothing to report"


def test_healthz_passes_through_what_the_worker_published(client):
    """
    The device must come from the worker, not from the API's own environment.
    The API container has no torch; its own DEVICE setting once claimed "cpu"
    while the worker was running on a GPU.
    """
    import json
    from backend.api import main
    from backend.worker import tasks

    main._redis.set(tasks.STATUS_KEY, json.dumps({
        "device": "cuda",
        "device_detail": "cuda: NVIDIA L4",
        "device_requested": "auto",
        "dry_run": False,
        "checkpoint_present": True,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }))

    wk = client.get("/healthz").json()["worker"]
    assert wk["device"] == "cuda"
    assert "L4" in wk["device_detail"]
    # The API's own config still says what IT was configured with, unchanged.
    assert client.get("/healthz").json()["config"]["device"] == "cpu"


def test_healthz_survives_a_corrupt_status_key(client):
    from backend.api import main
    from backend.worker import tasks
    main._redis.set(tasks.STATUS_KEY, b"{not json")

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["worker"] is None


# -------------------------------------------------------------- DICOM upload

def _dicom_zip(n=6, modality="CT"):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_dicom import _slice, _zip
    return _zip([_slice(i, modality=modality) for i in range(n)])


def test_dicom_series_imports_as_a_case(client):
    pytest.importorskip("pydicom")
    r = client.post("/api/cases/dicom",
                    files={"file": ("series.zip", _dicom_zip(), "application/zip")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["source"] == "dicom"
    assert body["shape"][0] == 6
    assert body["dicom"]["slices_used"] == 6
    assert body["dicom"]["spacing_z_source"] == "ImagePositionPatient"


def test_dicom_case_records_geometry_but_no_identifiers(client):
    pytest.importorskip("pydicom")
    body = client.post("/api/cases/dicom",
                       files={"file": ("s.zip", _dicom_zip(), "application/zip")}).json()
    note = body["note"]
    assert "slices" in note and "mm" in note
    assert "1.2.3.4" not in note          # the SeriesInstanceUID must not leak


def test_a_non_ct_series_is_refused_with_a_readable_reason(client):
    pytest.importorskip("pydicom")
    r = client.post("/api/cases/dicom",
                    files={"file": ("mr.zip", _dicom_zip(modality="MR"),
                                    "application/zip")})
    assert r.status_code == 400
    assert "not CT" in r.json()["detail"]


def test_a_non_zip_upload_is_refused(client):
    r = client.post("/api/cases/dicom",
                    files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert r.status_code == 400
    assert "zip" in r.json()["detail"].lower()


def test_suggest_refuses_to_guess_on_a_whole_series(client, monkeypatch):
    """
    The auto-prompt scores candidates by how central they are. On a full chest
    CT the centre is mediastinum, so a confident box there would be worse than
    no box at all.
    """
    import numpy as np
    from backend.common import store

    big = np.full((60, 256, 256), -800, dtype=np.int16)
    case_id = store.save_case(big, (1.0, 0.7, 0.7), source="dicom")
    from backend.api import db as dbmod
    with dbmod.SessionLocal() as sess:
        sess.add(dbmod.Case(id=case_id, source="dicom", note="test",
                            shape_z=60, shape_y=256, shape_x=256,
                            spacing_z=1.0, spacing_y=0.7, spacing_x=0.7))
        sess.commit()

    s = client.get(f"/api/cases/{case_id}/suggest").json()
    assert s["source"] == "unavailable"
    assert s["box"] is None, "a guessed box on a whole scan is worse than none"
    assert "does not detect" in s["info"]["detail"]


def test_suggest_still_works_on_an_roi_sized_case(client):
    case = client.post("/api/cases/phantom", json={"diameter_mm": 9.0}).json()
    s = client.get(f"/api/cases/{case['case_id']}/suggest").json()
    assert s["source"] in ("auto", "centred_fallback")
    assert s["box"] is not None


def test_healthz_trusts_the_heartbeat_not_rqs_registry(client):
    """
    RQ writes its worker registry ONCE, at worker startup. Restart Redis and
    that record is gone while the worker carries on draining the queue, so
    workers_alive reads 0 for a worker that is visibly working -- which is
    exactly what happened in practice.

    worker_live comes from the worker's own expiring heartbeat instead, so it
    recovers on its own once the worker writes again.
    """
    import json
    from backend.api import main
    from backend.common import settings

    main._redis.delete(settings.WORKER_STATUS_KEY)
    body = client.get("/healthz").json()
    assert body["worker_live"] is False, "nothing has published a heartbeat"

    main._redis.set(settings.WORKER_STATUS_KEY,
                   json.dumps({"device": "cuda", "checkpoint_present": True}))

    body = client.get("/healthz").json()
    assert body["worker_live"] is True, \
        "a heartbeat is present, so the API must report a live worker"
    assert body["workers_alive"] == 0, \
        "RQ's registry is still empty -- that is the whole point of the fix"


def test_suggest_never_proposes_a_box_too_small_to_measure(client, monkeypatch):
    """
    Observed on a real case: the auto-prompt won with a component that cleared
    the voxel floor but measured 0.7 x 2.1 mm in plane. Sizing already refuses
    to trust anything under 4mm, so proposing it arms Measure with a prompt
    that cannot produce a usable number. It must fall back instead.
    """
    from backend.api import main
    from backend.common import autoprompt

    case_id = client.post("/api/cases",
                          files={"file": ("roi.npz", _npz(), "application/octet-stream")}
                          ).json()["case_id"]

    # a winning component one pixel wide, which is what the real failure was
    monkeypatch.setattr(autoprompt, "suggest_box",
                        lambda vol, **kw: (3, [60.0, 60.0, 61.0, 62.0], {"reason": "ok"}))

    body = client.get(f"/api/cases/{case_id}/suggest").json()

    assert body["source"] == "centred_fallback", \
        "a sub-4mm proposal must be rejected, not offered"
    assert body["box"] is not None, "rejecting it must still leave a usable starting box"
    x0, y0, x1, y1 = body["box"]
    assert (x1 - x0) >= 8 and (y1 - y0) >= 8, "the fallback must be big enough to prompt with"
    assert "rejected_box_mm" in body["info"], "the rejection has to be inspectable"


def test_suggest_never_proposes_a_box_too_large_to_be_a_nodule(client, monkeypatch):
    """
    Observed on a ground-glass case: the auto-prompt proposed a box spanning
    rows 1..126 of a 128px crop -- one pixel in from each edge, which is
    exactly what survives a border check that only rejects components
    TOUCHING the edge. The model then segmented the lung inside it and
    reported an 80 mm "nodule" at -861 HU, which is air.

    LIDC's own inclusion criterion is 3-30 mm. A proposal wider than that is
    not a nodule, and offering it is worse than offering nothing, because the
    box looks deliberate.
    """
    from backend.common import autoprompt

    # a real-sized ROI: 128px at 0.72mm is the 92mm crop the failure came from
    blob = _npz(shape=(32, 128, 128), spacing=(1.25, 0.72, 0.72))
    case_id = client.post("/api/cases",
                          files={"file": ("roi.npz", blob, "application/octet-stream")}
                          ).json()["case_id"]

    # spans the crop but stops one pixel short of every edge, as the real one did
    monkeypatch.setattr(autoprompt, "suggest_box",
                        lambda vol, **kw: (16, [1.0, 1.0, 126.0, 126.0], {"reason": "ok"}))

    body = client.get(f"/api/cases/{case_id}/suggest").json()

    assert body["source"] == "centred_fallback", \
        "a proposal spanning the crop must be rejected, not offered"
    assert "ceiling" in body["info"]["reason"], "the rejection must say which bound it broke"
    x0, y0, x1, y1 = body["box"]
    assert (x1 - x0) < 125 and (y1 - y0) < 125, "the fallback must be smaller than what it replaced"


def _npz_with_reference():
    """An npz as scripts/02_prepare_data.py writes it: image AND consensus mask."""
    import io as _io
    shape = (32, 128, 128)
    vol = np.full(shape, -800, dtype=np.int16)
    mask = np.zeros(shape, dtype=bool)
    # a small lesion well off centre, so a centred box would NOT find it
    vol[14:19, 30:40, 90:100] = 40
    mask[14:19, 30:40, 90:100] = True
    buf = _io.BytesIO()
    np.savez_compressed(buf, image_hu=vol, mask=mask,
                        spacing=np.asarray([1.25, 0.72, 0.72], dtype=np.float32))
    return buf.getvalue()


def test_reference_box_comes_back_where_the_readers_drew_it(client):
    """
    The consensus mask ships with the annotated ROIs and used to be dropped on
    upload. It is kept now because the tool measures what it is pointed at,
    and a ground-glass lesion is close to invisible -- without this there is
    nothing to aim at. It is also the box the offline baseline used, so the
    UI can be checked against the number claimed for it.
    """
    case_id = client.post("/api/cases",
                          files={"file": ("roi.npz", _npz_with_reference(),
                                          "application/octet-stream")}).json()["case_id"]

    assert client.get(f"/api/cases/{case_id}").json()["has_reference"] is True

    r = client.get(f"/api/cases/{case_id}/reference").json()
    assert r["source"] == "lidc_consensus"
    assert r["slices_annotated"] == 5
    x0, y0, x1, y1 = r["box"]
    # the lesion sits at x 90..99, y 30..39; padding may widen it, never shift it
    assert x0 <= 90 and x1 >= 99 and y0 <= 30 and y1 >= 39
    assert 14 <= r["slice"] <= 18


def test_a_case_without_an_annotation_says_so_instead_of_inventing_one(client):
    """A real scan has no consensus mask. That must be a plain 404."""
    case_id = client.post("/api/cases",
                          files={"file": ("roi.npz", _npz(), "application/octet-stream")}
                          ).json()["case_id"]
    assert client.get(f"/api/cases/{case_id}").json()["has_reference"] is False
    assert client.get(f"/api/cases/{case_id}/reference").status_code == 404


def test_the_reference_mask_never_reaches_the_segmentation_path():
    """
    It is reference data, not an input. If the worker could read it, every
    number this project reports would be worthless.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.glob("backend/worker/*.py")):
        text = path.read_text(encoding="utf-8")
        if "load_reference" in text or "reference_from_npz" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"the worker must not read the consensus mask: {offenders}"
