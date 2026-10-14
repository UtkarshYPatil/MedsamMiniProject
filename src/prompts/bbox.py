"""
Prompt generation.

Phase 2/3: the box is derived from the ground-truth consensus mask. That makes
these numbers "segmentation quality GIVEN a correct box", not end-to-end
detection performance. Say so in any report.

In the radiologist tool, the box instead comes from the user's click/drag --
same function signature, different source.
"""
import numpy as np


def bbox_from_mask(mask_2d, padding=5):
    """Tight box around a boolean 2D mask -> [x0, y0, x1, y1] or None."""
    ys, xs = np.where(mask_2d)
    if ys.size == 0:
        return None
    h, w = mask_2d.shape
    return [
        float(max(0, xs.min() - padding)), float(max(0, ys.min() - padding)),
        float(min(w - 1, xs.max() + padding)), float(min(h - 1, ys.max() + padding)),
    ]


def bbox_from_click(cx, cy, half_size, shape):
    """Radiologist clicks a nodule centre -> a square box. Used by the tool."""
    h, w = shape
    return [
        float(max(0, cx - half_size)), float(max(0, cy - half_size)),
        float(min(w - 1, cx + half_size)), float(min(h - 1, cy + half_size)),
    ]


def best_slice(mask_3d):
    """Index of the slice with the largest nodule cross-section."""
    areas = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))
