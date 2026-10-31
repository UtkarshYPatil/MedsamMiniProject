"""
Turn a segmentation mask into the measurement summary a radiologist reads.

SCOPE
-----
Measurements only. No staging, no malignancy score, no follow-up recommendation.
The tool reports what it measured; the radiologist decides what it means.

DENSITY
-------
`density_class` is PROVISIONAL and carries `density_calibrated=False`.
On a 30-nodule sample, HU-based classification agreed with LIDC reader texture
ratings only 60% of the time, and every error was a reader-rated SOLID nodule
whose attenuation looked subsolid. Attenuation and perceptual texture are not
the same measurement. Until scripts/05_calibrate_density.py validates it on a
class-balanced sample, consumers must show the HU statistics, not the label.

SHAPE
-----
A nodule is roughly spherical. A mask that has latched onto the chest wall or
pleura is a slab, and in the Phase 3 tool exactly that happened: a prompt placed
near the edge of the crop produced a 3185 mm3 "nodule" of 17.4 x 30.8 mm with
HIGH confidence and no flags, 43x the reference volume. Every geometric check
passed, because the mask was large, solid and spanned many slices.

On the 137-nodule set, diameter_max / diameter_mean ran 1.02 to 1.55 against
that chest-wall mask's 1.77, so nothing real reached the threshold. At 902
nodules the separation is no longer clean: the median is still 1.16 and p95
1.41, but the maximum is 2.47 and 9 real nodules (1.0%) now cross 1.6.

So this is a heuristic that costs 1% of cases a downgrade to LOW, not a
discriminator. It is kept because that is the right trade -- it lowers
confidence rather than refusing to report, and the failure it catches
produced a 43x volume error at HIGH confidence with no other check firing.
Volume does not catch that case (14 real nodules exceed 3185 mm3) and core
attenuation does not (47 of 137 exceeded 0 HU, one calcified nodule at
+1271).
"""
from dataclasses import dataclass, asdict, field

from src.evaluate.metrics import measure, mean_diameter_mm
from src.report.density import classify, solid_component_mask, hu_profile

# Set above the maximum observed on 137 nodules (1.55) and below the 1.77 of
# the known chest-wall failure. At 902 nodules 9 real nodules (1.0%) exceed
# it, so it is a heuristic with a known false-positive rate, not a clean
# separator -- which is why it lowers confidence rather than refusing to
# report. A spiculated nodule, or one merged with a vessel, is legitimately
# this elongated.
ELONGATION_MAX = 1.6


@dataclass
class NoduleMeasurement:
    # --- measurements: trustworthy -----------------------------------
    diameter_mean_mm: float
    diameter_max_mm: float
    volume_mm3: float
    n_voxels: int
    hu_mean: float
    hu_p50: float
    hu_p90: float
    core_hu_p50: float
    core_solid_fraction: float

    # --- interpretation: provisional ---------------------------------
    density_class: str
    density_calibrated: bool
    solid_component_mm: float

    # --- self-check ---------------------------------------------------
    confidence: str
    flags: list = field(default_factory=list)


def summarize(mask, vol_hu, spacing_zyx):
    m = measure(mask, vol_hu, spacing_zyx)
    label, ev = classify(vol_hu, mask)
    prof = hu_profile(vol_hu, mask) or {}

    solid_mm = 0.0
    if label == "part_solid":
        sc = solid_component_mask(vol_hu, mask)
        if sc.any():
            solid_mm = mean_diameter_mm(sc, spacing_zyx)

    conf, flags = flag_quality(mask, m, label, prof)

    return NoduleMeasurement(
        diameter_mean_mm=round(m["diameter_mean_mm"], 1),
        diameter_max_mm=round(m["diameter_max_mm"], 1),
        volume_mm3=round(m["volume_mm3"], 1),
        n_voxels=m["n_voxels"],
        hu_mean=round(m["hu_mean"], 1),
        hu_p50=round(prof.get("hu_p50", float("nan")), 1),
        hu_p90=round(prof.get("hu_p90", float("nan")), 1),
        core_hu_p50=round(prof.get("core_hu_p50", float("nan")), 1),
        core_solid_fraction=round(prof.get("core_solid_fraction", float("nan")), 3),
        density_class=label,
        density_calibrated=bool(ev.get("calibrated", False)),
        solid_component_mm=round(solid_mm, 1),
        confidence=conf,
        flags=flags,
    )


def flag_quality(mask, m, density_class, prof=None):
    """
    Conservative self-check. Returns (level, reasons).

    The tool should push uncertain cases BACK to the radiologist rather than
    present a number with false confidence.
    """
    prof = prof or {}
    flags = []
    level = "high"

    def drop(to, why):
        nonlocal level
        order = {"high": 0, "medium": 1, "low": 2}
        if order[to] > order[level]:
            level = to
        flags.append(why)

    if m["n_voxels"] < 20:
        drop("low", "mask under 20 voxels")

    n_slices = int((mask.reshape(mask.shape[0], -1).sum(1) > 0).sum())
    if n_slices < 2:
        drop("low", "mask spans a single slice")

    if m["diameter_mean_mm"] < 4.0:
        drop("medium", "under 4 mm - partial volume dominates")

    if density_class == "ground_glass":
        drop("low", "ground-glass: weakest boundary, segmentation least reliable")
    elif density_class == "part_solid":
        drop("medium", "part-solid: mixed attenuation, boundary ambiguous")

    if prof.get("core_eroded") is False and m["n_voxels"] >= 20:
        drop("medium", "too small to erode - HU includes partial-volume rim")

    # --- shape plausibility: is this nodule-shaped at all? ----------------
    # The geometric checks above all passed on a chest-wall mask. These two
    # ask whether the thing measured looks like a nodule rather than a slab.
    mean_d = m.get("diameter_mean_mm") or 0.0
    max_d = m.get("diameter_max_mm") or 0.0
    if mean_d > 0:
        elongation = max_d / mean_d
        if elongation >= ELONGATION_MAX:
            drop("low", f"slab-like shape (max/mean diameter {elongation:.2f}) - "
                        f"check the mask is on a nodule and not the chest wall")

    # A nodule contained in the crop does not run out of its in-plane edges.
    # Z is excluded: a large nodule legitimately spans the full depth.
    try:
        if mask.ndim == 3 and (mask[:, 0, :].any() or mask[:, -1, :].any()
                               or mask[:, :, 0].any() or mask[:, :, -1].any()):
            drop("medium", "mask reaches the edge of the ROI - it may continue "
                           "outside the crop")
    except (AttributeError, IndexError, TypeError):
        pass            # not an array-like mask; the other checks still apply

    return level, flags


def as_dict(nm: NoduleMeasurement) -> dict:
    return asdict(nm)
