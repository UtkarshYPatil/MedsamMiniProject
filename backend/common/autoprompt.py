"""
Propose a prompt box automatically, so the usual case is one button press.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This proposes a box inside an ROI that was already cropped around a nodule.
It is NOT nodule detection on a whole scan -- it assumes the nodule is roughly
central, which is true by construction for anything scripts/02_prepare_data.py
produced, and false for an arbitrary chest CT. Detection remains open work.

It is a convenience and a safety net, not a diagnosis: the radiologist can
always override the box, and the measurement that follows is still prompted
segmentation, not automatic finding.

HOW IT WORKS
------------
1. Keep voxels denser than lung parenchyma but not bone-bright. A nodule sits
   roughly between -600 and +300 HU; air is far below, rib and contrast far
   above.
2. Discard every component touching the volume border. This is the important
   step: the chest wall and pleura enter the crop at its edges, and a mask on
   the chest wall was exactly the failure that produced a 43x-too-large
   measurement with no warning. A nodule inside the lung does not touch the
   border of a centroid-centred crop.
3. Score what is left by size and by closeness to the centre, and take the
   best. The centre prior is justified by (1); without a centroid-centred
   crop it would not be.
4. Return the tight box on that component's largest-area slice, with no
   padding -- scripts/06_ablate.py showed Dice falling 0.804 -> 0.266 as
   padding grows from 0 to 10 px.
"""
import numpy as np

# A nodule's attenuation window. Wide enough for ground-glass at the bottom
# and a calcified nodule at the top, tight enough to exclude air and rib.
HU_LOW = -600.0
HU_HIGH = 300.0

MIN_VOXELS = 8          # smaller than this is noise, not a 3 mm nodule
CENTRE_SIGMA = 0.35     # centre prior width, as a fraction of the ROI radius


def _label(binary):
    """Connected components, preferring scipy and falling back to numpy."""
    try:
        from scipy.ndimage import label
        lab, n = label(binary)
        return lab, n
    except Exception:
        return _label_numpy(binary)


def _label_numpy(binary):
    """
    Union-find labelling, so this module works without scipy.

    Only used as a fallback; scipy's is faster and is what runs in practice.
    """
    lab = np.zeros(binary.shape, dtype=np.int32)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    idx = np.argwhere(binary)
    for z, y, x in idx:
        neigh = []
        for dz, dy, dx in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
            pz, py, px = z + dz, y + dy, x + dx
            if pz >= 0 and py >= 0 and px >= 0 and lab[pz, py, px]:
                neigh.append(int(lab[pz, py, px]))
        if not neigh:
            lab[z, y, x] = nxt
            parent[nxt] = nxt
            nxt += 1
        else:
            m = min(neigh)
            lab[z, y, x] = m
            for o in neigh:
                union(m, o)

    if nxt == 1:
        return lab, 0
    remap, out = {}, 0
    for old in range(1, nxt):
        root = find(old)
        if root not in remap:
            out += 1
            remap[root] = out
        remap[old] = remap[root]
    flat = lab.ravel()
    nz = flat > 0
    flat[nz] = [remap[v] for v in flat[nz]]
    return flat.reshape(lab.shape), out


def _touches_border(mask):
    return bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any()
                or mask[:, :, 0].any() or mask[:, :, -1].any())


def suggest_box(vol_hu, hu_low=HU_LOW, hu_high=HU_HIGH, min_voxels=MIN_VOXELS):
    """
    Propose (slice_index, [x0, y0, x1, y1], info).

    Returns (None, None, info) when nothing plausible is found, so the caller
    can fall back to a centred default rather than guess. `info` always
    explains what happened, because a proposal the user cannot interrogate is
    worse than no proposal.
    """
    vol = np.asarray(vol_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume (Z, Y, X), got {vol.shape}")

    Z, Y, X = vol.shape
    candidate = (vol >= hu_low) & (vol <= hu_high)
    info = {"hu_window": [hu_low, hu_high],
            "candidate_voxels": int(candidate.sum())}

    if not candidate.any():
        info["reason"] = "nothing in the nodule attenuation window"
        return None, None, info

    lab, n = _label(candidate)
    info["components"] = int(n)
    if n == 0:
        info["reason"] = "no connected components"
        return None, None, info

    cz, cy, cx = (Z - 1) / 2, (Y - 1) / 2, (X - 1) / 2
    radius = max(1.0, min(Y, X) / 2.0)

    best, best_score, dropped_border = None, -1.0, 0
    for i in range(1, n + 1):
        comp = lab == i
        size = int(comp.sum())
        if size < min_voxels:
            continue
        if _touches_border(comp):
            dropped_border += 1          # chest wall / pleura, not a nodule
            continue

        zz, yy, xx = np.where(comp)
        dist = np.sqrt(((zz.mean() - cz) / max(1.0, Z / 2)) ** 2
                       + ((yy.mean() - cy) / radius) ** 2
                       + ((xx.mean() - cx) / radius) ** 2)
        # size favours a real nodule over speckle; the gaussian centre prior
        # favours the thing the crop was built around.
        score = np.sqrt(size) * float(np.exp(-(dist ** 2) / (2 * CENTRE_SIGMA ** 2)))
        if score > best_score:
            best_score, best = score, (comp, size, dist)

    info["dropped_touching_border"] = dropped_border

    if best is None:
        info["reason"] = ("every component either touches the border (chest "
                          "wall) or is below the size floor")
        return None, None, info

    comp, size, dist = best
    areas = comp.reshape(Z, -1).sum(axis=1)
    z = int(np.argmax(areas))

    ys, xs = np.where(comp[z])
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    info.update({
        "voxels": size,
        "slices_spanned": int((areas > 0).sum()),
        "normalised_distance_from_centre": round(float(dist), 3),
        "reason": "ok",
    })
    return z, box, info


def centred_fallback(shape, half=16):
    """A centred box, for when suggest_box finds nothing worth proposing."""
    Z, Y, X = shape
    z = Z // 2
    cy, cx = Y // 2, X // 2
    return z, [float(max(0, cx - half)), float(max(0, cy - half)),
               float(min(X - 1, cx + half)), float(min(Y - 1, cy + half))]
