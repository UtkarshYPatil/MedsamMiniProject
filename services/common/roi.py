"""
Cut the validated ROI out of a full CT series.

WHY
---
Every number this project reports was measured on 128x128x33 crops centred on
a nodule (configs/config.yaml: roi_size 128, roi_depth 16). A whole series is
512x512x~250. Handing the model the whole thing changes the input distribution
it was measured on, costs far more memory and time, and gives the propagation
hundreds of slices to drift through.

So a DICOM upload is cropped to the same geometry the ROI pipeline produces,
around wherever the radiologist put the box. The model then sees what it saw
during evaluation, and the measurements stay comparable to outputs/results.csv.

Spacing is unchanged by cropping, so millimetres computed inside the crop are
the same millimetres as outside it. Only the voxel indices move, which is what
`origin` records.
"""
import numpy as np


def crop_around(vol, prompt_z, box, size=128, depth=16):
    """
    (sub_vol, sub_z, sub_box, origin) with origin = (z0, y0, x0).

    The crop is centred on the prompt box, then shifted to stay inside the
    volume rather than being clipped smaller -- a smaller crop would change
    the geometry the model expects. A volume already smaller than the window
    comes back untouched, which is what an uploaded .npz ROI does.
    """
    D, H, W = vol.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    cz = int(round(prompt_z))
    cy = int(round((y0 + y1) / 2))
    cx = int(round((x0 + x1) / 2))

    def window(centre, span, extent):
        """`span` voxels wide, centred, shifted inside the volume."""
        if extent <= span:
            return 0, extent              # already small enough: take it all
        lo = centre - span // 2
        lo = max(0, min(lo, extent - span))
        return int(lo), int(lo + span)

    # roi_depth is slices EACH SIDE of the centre, so the depth is 2d+1 -- 33
    # for depth 16, which is exactly what scripts/02_prepare_data.py writes.
    # Using 2d here instead clipped a slice off every existing ROI.
    z0, z1 = window(cz, 2 * depth + 1, D)
    yy0, yy1 = window(cy, size, H)
    xx0, xx1 = window(cx, size, W)

    sub = vol[z0:z1, yy0:yy1, xx0:xx1]
    sub_box = [x0 - xx0, y0 - yy0, x1 - xx0, y1 - yy0]
    return sub, cz - z0, sub_box, (z0, yy0, xx0)


def paste(sub_mask, full_shape, origin):
    """Put a crop-sized mask back at its place in the full volume."""
    out = np.zeros(full_shape, dtype=bool)
    z0, y0, x0 = origin
    d, h, w = sub_mask.shape
    out[z0:z0 + d, y0:y0 + h, x0:x0 + w] = sub_mask
    return out


def needs_crop(shape, size=128, depth=16) -> bool:
    """True only for volumes larger than one ROI -- a 33x128x128 is not."""
    D, H, W = shape
    return D > 2 * depth + 1 or H > size or W > size
