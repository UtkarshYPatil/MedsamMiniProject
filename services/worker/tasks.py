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
import time

import numpy as np

from services.common import settings, store

_segmenter = None


def get_segmenter():
    """Lazy, per-process. RQ forks per job, so this warms once per worker."""
    global _segmenter
    if _segmenter is None:
        from src.models.medsam2 import build
        _segmenter = build(settings.CHECKPOINT, settings.MODEL_CFG,
                           settings.DEVICE, dry_run=settings.DRY_RUN).load()
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

    vol_u8 = apply_window(vol_hu, settings.WINDOW_LEVEL, settings.WINDOW_WIDTH)

    seg = get_segmenter()
    t0 = time.perf_counter()
    mask = seg.segment_volume(vol_u8, int(prompt_z), [float(v) for v in box],
                              propagate=bool(propagate))
    dt = time.perf_counter() - t0

    if not mask.any():
        return {
            "empty": True,
            "seconds": round(dt, 3),
            "device": settings.DEVICE,
            "dry_run": settings.DRY_RUN,
            "message": "the model returned an empty mask for this prompt",
        }

    meas = as_dict(summarize(mask, vol_hu, spacing))
    store.save_mask(_job_id(), mask, prompt_z, box)

    n_slices = int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum())
    return {
        "empty": False,
        "seconds": round(dt, 3),
        "device": settings.DEVICE,
        "dry_run": settings.DRY_RUN,
        "slices_covered": n_slices,
        "prompt_slice": int(prompt_z),
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
