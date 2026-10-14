"""
Nodule density characterisation from Hounsfield attenuation.

STATUS: PROVISIONAL -- the thresholds below are NOT validated.

What the data showed (30 nodules, 10 scans, 2026-10-14):
    LIDC "solid" nodules spanned median HU -579 .. +216 and solid-fraction
    0.00 .. 1.00. That range completely contains both the part-solid and the
    ground-glass samples, so no threshold on the whole-mask attenuation can
    separate the classes. Two causes:
      1. Partial volume. The consensus contour is generous and small nodules
         average heavily with surrounding lung (~ -850 HU).
      2. LIDC "texture" is a subjective 1-5 reader rating, not an attenuation
         measurement. It correlates with density but is not defined by it.

Eroding to the nodule core removes most of the partial-volume rim and looks
far more separable (median core HU: solid +15, part-solid -228), but with only
2 part-solid and 1 ground-glass nodule in that sample it cannot be validated.

THEREFORE this module:
  - reports HU statistics always -- those are measurements, and they are what
    a radiologist can actually judge;
  - returns a suggested class marked `provisional`, never as fact;
  - returns `calibrated=False` until scripts/05_calibrate_density.py has been
    run on a sample with enough part-solid and ground-glass nodules.

Run scripts/05_calibrate_density.py over >=150 scans before quoting any
density number in a report.
"""
import numpy as np

# Attenuation cut-off for "solid tissue" (soft tissue sits near 0 HU,
# lung parenchyma near -850, ground glass between).
SOLID_HU = -200.0

# Fraction of CORE voxels above SOLID_HU. Provisional -- see module docstring.
SOLID_FRACTION = 0.85
GGO_FRACTION = 0.25

# Erosion iterations used to reach the nodule core.
CORE_EROSION = 1
MIN_CORE_VOXELS = 10


def core_mask(mask, iterations=CORE_EROSION):
    """
    Strip the partial-volume rim. Falls back to the full mask when the nodule
    is too small to erode -- small nodules are exactly where this is least
    reliable, which is why they are flagged.
    """
    from scipy.ndimage import binary_erosion
    if mask.sum() < MIN_CORE_VOXELS * 2:
        return mask, False
    eroded = binary_erosion(mask, iterations=iterations)
    if eroded.sum() < MIN_CORE_VOXELS:
        return mask, False
    return eroded, True


def hu_profile(vol_hu, mask):
    """Measurements only -- no interpretation. Always trustworthy."""
    if mask.sum() == 0:
        return {}
    core, eroded = core_mask(mask)
    full_hu, core_hu = vol_hu[mask], vol_hu[core]
    return {
        "hu_mean": float(full_hu.mean()),
        "hu_p10": float(np.percentile(full_hu, 10)),
        "hu_p50": float(np.percentile(full_hu, 50)),
        "hu_p90": float(np.percentile(full_hu, 90)),
        "core_hu_p50": float(np.percentile(core_hu, 50)),
        "core_solid_fraction": float((core_hu >= SOLID_HU).mean()),
        "full_solid_fraction": float((full_hu >= SOLID_HU).mean()),
        "core_voxels": int(core.sum()),
        "core_eroded": bool(eroded),
    }


def classify(vol_hu, mask, solid_hu=SOLID_HU):
    """
    Returns (label, evidence).

    label is one of solid / part_solid / ground_glass / unknown and is
    PROVISIONAL. evidence["calibrated"] is False until the thresholds have been
    fitted on a class-balanced sample. Callers must surface that.
    """
    if mask.sum() == 0:
        return "unknown", {"calibrated": False, "reason": "empty mask"}

    ev = hu_profile(vol_hu, mask)
    frac = ev["core_solid_fraction"]

    if frac >= SOLID_FRACTION:
        label = "solid"
    elif frac <= GGO_FRACTION:
        label = "ground_glass"
    else:
        label = "part_solid"

    ev["calibrated"] = False          # flipped to True only by a fitted config
    ev["provisional"] = True
    ev["solid_fraction"] = frac       # kept for backward compatibility
    return label, ev


def solid_component_mask(vol_hu, mask, solid_hu=SOLID_HU):
    """The solid sub-component of a part-solid nodule -- sized separately."""
    return mask & (vol_hu >= solid_hu)
