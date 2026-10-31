"""
Trim a propagated mask where it has run past the end of the nodule.

THE MEASUREMENT THAT MOTIVATES THIS
-----------------------------------
On 902 nodules (outputs/results.csv):

    mean diameter error   -0.01 mm, 92% within 1 mm   -- unbiased
    volume ratio          1.12 median, 1.90 at p90    -- 12-90% too large
    r(dice, volume ratio) -0.65
    r(dice, diameter ratio) -0.26

Diameter is an in-plane measure; volume is in-plane area times depth. The two
together locate the error: if the diameter is right and the volume is half
again too big, the excess is on slices where the nodule is not. MedSAM2's
memory attention carries the mask forward and keeps carrying it after the
nodule has ended.

WHAT THIS DOES
--------------
Walks outward from the prompt slice in both directions and stops at the first
slice whose mask area has collapsed relative to the prompt slice, discarding
everything beyond. A nodule is convex-ish in Z: area rises to a maximum and
falls away. Propagation drift looks different -- area collapses to a few
voxels and then persists, slice after slice, because the model is tracking
texture rather than the lesion.

WHAT THIS IS NOT
----------------
Not a change to the model. MedSAM2 stays frozen, zero parameters trained;
this is arithmetic on its output, the same category as keep_largest.

It is also not free. Setting the fraction too high truncates genuinely
tapering nodules and costs volume that was correct. The threshold has to be
chosen by measurement, which is what scripts/13_ablate_propagation.py is for.
Default OFF until that has been run.
"""
import numpy as np


def trim_axial_extent(mask, prompt_idx, min_area_frac=0.0, max_tail=2,
                      min_voxels=1):
    """
    Zero slices beyond the point where the mask stops being the nodule.

    mask           : (D, H, W) bool
    prompt_idx     : the slice the box was drawn on
    min_area_frac  : a slice below this fraction of the prompt slice's area
                     counts as "small". 0 disables the trim entirely.
    max_tail       : how many consecutive small slices to allow before
                     deciding it is drift and cutting from the start of the
                     run. A genuine nodule tapers through one or two small
                     slices and then reaches zero; drift stays small for many.
    min_voxels     : a slice below this always counts as small

    Returns a new mask; the input is not modified.

    The rule is not "stop when the area gets small". A real nodule's last
    slices ARE small -- cutting those throws away correct volume, which is
    what the first version of this function did: on a mask with areas
    5/40/120/200/120/40/5 it removed both 5-voxel ends, which were the
    nodule. What separates drift from a taper is persistence: a taper
    collapses to zero within a slice or two, drift holds at a few voxels for
    as long as you let it propagate.
    """
    if min_area_frac <= 0 or mask.ndim != 3 or not mask.any():
        return mask

    D = mask.shape[0]
    z = int(np.clip(prompt_idx, 0, D - 1))
    areas = mask.reshape(D, -1).sum(axis=1)

    anchor = float(areas[z])
    if anchor <= 0:
        # The prompt slice is empty, so there is no scale to judge the rest
        # against. Trimming on a guess would be worse than not trimming.
        return mask

    floor = max(float(min_voxels), min_area_frac * anchor)

    def scan(indices):
        """Keep until a run of more than max_tail small slices, or empty."""
        keep, run, pending = [], 0, []
        for d in indices:
            a = areas[d]
            if a <= 0:
                # Reached zero: the nodule genuinely ended here, so the small
                # slices leading up to it were its taper. Keep them. (Dropping
                # them was the second bug in this function -- the `for/else`
                # only kept `pending` when the loop ran off the end, and a
                # real nodule always ends by breaking here instead.)
                keep.extend(pending)
                break
            if a < floor:
                run += 1
                pending.append(d)
                if run > max_tail:
                    break                   # persistent and small: drift
            else:
                keep.extend(pending)        # the taper resumed; it was real
                pending, run = [], 0
                keep.append(d)
        else:
            keep.extend(pending)            # ran to the edge without a break
        return keep

    keep = np.zeros(D, dtype=bool)
    keep[z] = True
    for d in scan(range(z + 1, D)):
        keep[d] = True
    for d in scan(range(z - 1, -1, -1)):
        keep[d] = True

    if keep.all():
        return mask
    out = mask.copy()
    out[~keep] = False
    return out


def describe_trim(mask_before, mask_after):
    """What the trim removed, for an ablation table or a log line."""
    b = int(mask_before.sum())
    a = int(mask_after.sum())
    sb = int((mask_before.reshape(mask_before.shape[0], -1).sum(1) > 0).sum())
    sa = int((mask_after.reshape(mask_after.shape[0], -1).sum(1) > 0).sum())
    return {
        "voxels_before": b,
        "voxels_after": a,
        "voxels_removed": b - a,
        "fraction_removed": round((b - a) / b, 4) if b else 0.0,
        "slices_before": sb,
        "slices_after": sa,
    }
