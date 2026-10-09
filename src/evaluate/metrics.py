"""Segmentation metrics. All take boolean arrays of the same shape."""
import numpy as np


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, gt).sum() / denom


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, gt).sum() / union


def hd95(pred, gt, spacing=None):
    """95th-percentile Hausdorff distance in mm (needs medpy)."""
    from medpy.metric.binary import hd95 as _hd95
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    return float(_hd95(pred.astype(bool), gt.astype(bool), voxelspacing=spacing))


def volume_mm3(mask, spacing_zyx):
    """spacing_zyx = (slice_thickness, pixel_y, pixel_x) in mm."""
    return float(mask.sum() * np.prod(spacing_zyx))


def max_diameter_mm(mask_3d, spacing_zyx):
    """Largest in-plane diameter across slices -- what Fleischner-style sizing uses."""
    from skimage.measure import regionprops, label
    best = 0.0
    py, px = spacing_zyx[1], spacing_zyx[2]
    for z in range(mask_3d.shape[0]):
        sl = mask_3d[z]
        if not sl.any():
            continue
        for r in regionprops(label(sl)):
            d = r.major_axis_length * ((py + px) / 2)
            best = max(best, d)
    return float(best)


def mean_hu(volume_hu, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(volume_hu[mask].mean())


def summarize(rows):
    """rows: list of dicts -> per-metric mean/std."""
    import pandas as pd
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    return df, num.agg(["mean", "std", "min", "max"]).T
