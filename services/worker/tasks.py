"""
The one job this stack runs: prompt MedSAM2 on a slice, measure the result.

The model is loaded once per worker process and kept. Loading it per job would
dominate the runtime -- the forward pass is about a second, the load is tens of
seconds.

The prompt box arrives from the caller (in the app, from the radiologist's
drag) and is used exactly as given. No padding is added: scripts/06_ablate.py
showed Dice falling 0.804 -> 0.266 as padding grows from 0 to 10 px, because
MedSAM2 fills toward the box it is given.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from services.common import roi, settings, store

_segmenter = None
_device = None

# What the worker actually resolved, published for /healthz to read.
#
# Only the worker can answer "is this running on a GPU?" -- it is the only
# container with torch. The API used to report its own DEVICE setting, which
# said "cpu" while the worker ran on cuda. Reporting a value the reporting
# process cannot verify is how a dashboard ends up lying.
STATUS_KEY = settings.WORKER_STATUS_KEY


def checkpoint_present() -> bool:
    return bool(settings.CHECKPOINT) and os.path.exists(settings.CHECKPOINT)


def worker_status() -> dict:
    return {
        "device": resolve_device(),
        "device_detail": device_detail(),
        "device_requested": settings.DEVICE,
        "dry_run": bool(settings.DRY_RUN),
        "checkpoint": settings.CHECKPOINT,
        "checkpoint_present": checkpoint_present(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def publish_status(conn=None) -> dict:
    """
    Best-effort: write the worker's resolved state where the API can see it.

    Never raises. A status line failing to publish must not take down the
    worker that is doing the actual work.
    """
    st = worker_status()
    try:
        if conn is None:
            from redis import Redis
            conn = Redis.from_url(settings.REDIS_URL)
        conn.set(STATUS_KEY, json.dumps(st))
    except Exception:
        pass
    return st


def resolve_device() -> str:
    """
    Decide once what to run on, and remember it.

    DEVICE=auto asks torch whether a usable GPU is actually visible rather than
    whether CUDA was compiled in -- a CUDA build with no card, or a container
    started without --gpus, both report False and fall back to CPU instead of
    dying at the first forward pass.

    An explicit DEVICE=cuda is honoured as written: if someone asked for the
    GPU and it is not there, that is worth failing loudly over rather than
    silently running 30x slower.
    """
    global _device
    if _device is not None:
        return _device

    want = settings.DEVICE
    if want in ("cpu", "cuda"):
        _device = want
        return _device

    _device = "cpu"
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _device = "cuda"
    except Exception:
        pass            # no torch, no driver, no CUDA build -- all mean cpu
    return _device


def device_detail() -> str:
    """A human-readable note about what resolve_device() settled on."""
    dev = resolve_device()
    if dev != "cuda":
        return f"{dev} (requested {settings.DEVICE})"
    try:
        import torch
        return f"cuda: {torch.cuda.get_device_name(0)}"
    except Exception:
        return "cuda"


def get_segmenter():
    """Lazy, per-process. Warms once per worker, not once per job."""
    global _segmenter
    if _segmenter is None:
        from src.models.medsam2 import build
        _segmenter = build(settings.CHECKPOINT, settings.MODEL_CFG,
                           resolve_device(), dry_run=settings.DRY_RUN).load()
    return _segmenter


def segment_case(case_id: str, prompt_z: int, box, propagate: bool = True) -> dict:
    """
    Segment one nodule and measure it.

    Returns the measurement dict the API stores and the UI renders. Raising is
    fine -- RQ records the traceback and the API surfaces it as a failed job.
    """
    from src.data.preprocess import apply_window
    from src.report.sizing import as_dict, summarize

    vol_hu, spacing = store.load_case(case_id)
    if not 0 <= prompt_z < vol_hu.shape[0]:
        raise ValueError(f"slice {prompt_z} outside 0..{vol_hu.shape[0] - 1}")

    # A whole DICOM series is cropped to the geometry every reported number was
    # measured on (128x128x33), centred on the radiologist's box. Segmenting
    # 512x512x250 instead would change the input distribution the model was
    # evaluated on and give the propagation hundreds of slices to drift
    # through. An uploaded ROI is already that size and passes through
    # untouched.
    cropped = roi.needs_crop(vol_hu.shape, settings.ROI_SIZE, settings.ROI_DEPTH)
    if cropped:
        sub_hu, sub_z, sub_box, origin = roi.crop_around(
            vol_hu, int(prompt_z), box,
            size=settings.ROI_SIZE, depth=settings.ROI_DEPTH)
    else:
        sub_hu, sub_z, sub_box, origin = (vol_hu, int(prompt_z),
                                          [float(v) for v in box], (0, 0, 0))

    vol_u8 = apply_window(sub_hu, settings.WINDOW_LEVEL, settings.WINDOW_WIDTH)

    seg = get_segmenter()
    t0 = time.perf_counter()
    sub_mask = seg.segment_volume(vol_u8, sub_z, sub_box,
                                  propagate=bool(propagate))
    dt = time.perf_counter() - t0

    # Back into full-volume coordinates, so the mask, the overlay and the
    # measurements all speak the same indices the caller prompted in.
    mask = roi.paste(sub_mask, vol_hu.shape, origin) if cropped else sub_mask

    if not mask.any():
        return {
            "empty": True,
            "seconds": round(dt, 3),
            "device": resolve_device(),
            "dry_run": settings.DRY_RUN,
            "message": "the model returned an empty mask for this prompt",
        }

    meas = as_dict(summarize(mask, vol_hu, spacing))
    store.save_mask(_job_id(), mask, prompt_z, box)
    publish_status()        # refresh the timestamp so staleness is visible

    n_slices = int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum())
    return {
        "empty": False,
        "seconds": round(dt, 3),
        "device": resolve_device(),
        "dry_run": settings.DRY_RUN,
        "slices_covered": n_slices,
        "prompt_slice": int(prompt_z),
        "cropped_to_roi": bool(cropped),
        "roi_origin": [int(v) for v in origin],
        "measurement": meas,
        # Repeated at the top level because this is what the UI leads with.
        "diameter_mean_mm": meas.get("diameter_mean_mm"),
        "volume_mm3": meas.get("volume_mm3"),
        "confidence": meas.get("confidence"),
        "flags": meas.get("flags", []),
        "density_class": meas.get("density_class"),
        "density_calibrated": meas.get("density_calibrated", False),
    }


def _job_id() -> str:
    """
    The id of the RQ job currently executing, so the mask is saved under it.

    Falls back to a fresh id when called outside a worker -- which is how the
    tests exercise this function without Redis.
    """
    try:
        from rq import get_current_job
        job = get_current_job()
    except Exception:
        job = None
    return job.id if job else store.new_id()
