"""
Derive a prompt from a nodule annotation.

Phase 2: the box comes from the ground-truth consensus mask. This stands in for
the automatic detector, which is Phase 3 work. State this openly in any report --
these numbers are "segmentation quality given a correct box", not end-to-end.
"""
import numpy as np


def bbox_from_mask(mask_2d, padding=5):
    """Tight box around a boolean 2D mask -> [x0, y0, x1, y1]."""
    ys, xs = np.where(mask_2d)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask_2d.shape
    return [
        max(0, x0 - padding), max(0, y0 - padding),
        min(w - 1, x1 + padding), min(h - 1, y1 + padding),
    ]


def centre_slice_index(mask_3d):
    """Slice with the largest nodule cross-section -- the best slice to prompt."""
    areas = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def point_from_mask(mask_2d):
    """Foreground point prompt at the mask centroid."""
    ys, xs = np.where(mask_2d)
    if ys.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]
