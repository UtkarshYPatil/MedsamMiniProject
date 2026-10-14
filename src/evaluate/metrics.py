"""Segmentation metrics and physical measurements."""
import numpy as np


def _axis(region, which="major"):
    """skimage renamed these in 0.26 -- support both."""
    new = f"axis_{which}_length"
    if hasattr(type(region), new) or new in getattr(region, "_extra_properties", {}):
        try:
            return float(getattr(region, new))
        except Exception:
            pass
    try:
        return float(getattr(region, new))
    except AttributeError:
        return float(getattr(region, f"{which}_axis_length"))


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    d = pred.sum() + gt.sum()
    return 1.0 if d == 0 else float(2.0 * np.logical_and(pred, gt).sum() / d)


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return 1.0 if u == 0 else float(np.logical_and(pred, gt).sum() / u)


def hd95(pred, gt, spacing=None):
    """95th-percentile Hausdorff distance in mm. Requires medpy."""
    try:
        from medpy.metric.binary import hd95 as _h
    except ImportError:
        return float("nan")
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    return float(_h(pred.astype(bool), gt.astype(bool), voxelspacing=spacing))


def volume_mm3(mask, spacing_zyx):
    return float(mask.sum() * float(np.prod(spacing_zyx)))


def max_diameter_mm(mask_3d, spacing_zyx):
    """
    Largest in-plane long-axis diameter across slices.
    This is the number nodule-sizing guidelines are keyed on.
    """
    from skimage.measure import label, regionprops
    py, px = float(spacing_zyx[1]), float(spacing_zyx[2])
    scale = (py + px) / 2.0
    best = 0.0
    for z in range(mask_3d.shape[0]):
        sl = mask_3d[z]
        if not sl.any():
            continue
        for r in regionprops(label(sl)):
            best = max(best, _axis(r, "major") * scale)
    return float(best)


def mean_diameter_mm(mask_3d, spacing_zyx):
    """Average of long and short axis -- the convention many protocols use."""
    from skimage.measure import label, regionprops
    py, px = float(spacing_zyx[1]), float(spacing_zyx[2])
    scale = (py + px) / 2.0
    best, best_area = 0.0, 0
    for z in range(mask_3d.shape[0]):
        sl = mask_3d[z]
        if not sl.any():
            continue
        for r in regionprops(label(sl)):
            if r.area > best_area:
                best_area = r.area
                best = (_axis(r, "major") + _axis(r, "minor")) / 2.0 * scale
    return float(best)


def hu_stats(vol_hu, mask):
    if mask.sum() == 0:
        return {"hu_mean": float("nan"), "hu_std": float("nan"), "hu_p90": float("nan")}
    v = vol_hu[mask]
    return {
        "hu_mean": float(v.mean()),
        "hu_std": float(v.std()),
        "hu_p90": float(np.percentile(v, 90)),
    }


def measure(mask, vol_hu, spacing_zyx):
    """All measurements for one nodule mask."""
    out = {
        "volume_mm3": volume_mm3(mask, spacing_zyx),
        "diameter_max_mm": max_diameter_mm(mask, spacing_zyx),
        "diameter_mean_mm": mean_diameter_mm(mask, spacing_zyx),
        "n_voxels": int(mask.sum()),
    }
    out.update(hu_stats(vol_hu, mask))
    return out
