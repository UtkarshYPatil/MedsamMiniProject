"""
LIDC-IDRI annotation parsing.

Each scan was read by four radiologists independently. pylidc clusters their
contours per nodule; `consensus()` merges them at a chosen agreement level.
The merged mask is the reference we score against -- it is a construct, not
absolute truth, so n_annotations is carried through for analysis.

AXIS ORDER -- the thing that silently corrupts everything if you get it wrong:
    scan.to_volume()  -> (Y, X, Z)
    consensus() mask  -> (Y, X, Z), and its bbox is 3 slices in the same order
    this project uses -> (Z, Y, X) everywhere downstream
Conversion happens here, once, and nowhere else.
"""
from dataclasses import dataclass, asdict

from src import compat  # noqa: F401  -- MUST come before pylidc
import numpy as np

# LIDC texture rating: 1 = non-solid (ground-glass) ... 5 = solid
TEXTURE_BINS = [(1, 2.5, "ground_glass"), (2.5, 3.5, "part_solid"), (3.5, 5.01, "solid")]


def texture_label(mean_texture: float) -> str:
    for lo, hi, name in TEXTURE_BINS:
        if lo <= mean_texture < hi:
            return name
    return "unknown"


@dataclass
class NoduleRecord:
    key: str
    patient_id: str
    nodule_idx: int
    centroid_zyx: tuple
    diameter_mm: float
    n_annotations: int          # 1-4 radiologists marked this nodule
    texture_mean: float
    texture_class: str          # solid / part_solid / ground_glass
    subtlety_mean: float
    malignancy_mean: float      # METADATA ONLY -- never a prediction target
    spacing_zyx: tuple


def iter_scans(limit=None):
    import pylidc as pl
    q = pl.query(pl.Scan)
    for i, scan in enumerate(q):
        if limit is not None and i >= limit:
            break
        yield scan


def nodules_in_scan(scan, clevel=0.5, pad=2, min_diameter_mm=3.0):
    """
    Yield (NoduleRecord, mask_zyx, bbox_zyx) per nodule.

    mask_zyx  : cropped consensus mask, already transposed to (Z, Y, X)
    bbox_zyx  : (slice_z, slice_y, slice_x) into the full (Z, Y, X) volume
    """
    from pylidc.utils import consensus

    spacing = (float(scan.slice_thickness),
               float(scan.pixel_spacing), float(scan.pixel_spacing))

    for idx, anns in enumerate(scan.cluster_annotations()):
        diam = float(np.mean([a.diameter for a in anns]))
        if diam < min_diameter_mm:
            continue

        mask_yxz, bbox_yxz, _ = consensus(anns, clevel=clevel, pad=pad)
        if not mask_yxz.any():
            continue

        # --- axis conversion, done once -------------------------------
        mask_zyx = np.transpose(mask_yxz, (2, 0, 1))          # (Y,X,Z) -> (Z,Y,X)
        sy, sx, sz = bbox_yxz[0], bbox_yxz[1], bbox_yxz[2]
        bbox_zyx = (sz, sy, sx)

        # centroid in FULL-volume (Z, Y, X) coordinates
        a0, a1, a2 = np.where(mask_yxz)       # a0=y, a1=x, a2=z within the crop
        cz = sz.start + float(a2.mean())
        cy = sy.start + float(a0.mean())
        cx = sx.start + float(a1.mean())

        tex = float(np.mean([a.texture for a in anns]))

        rec = NoduleRecord(
            key=f"{scan.patient_id}_n{idx}",
            patient_id=scan.patient_id,
            nodule_idx=idx,
            centroid_zyx=(cz, cy, cx),
            diameter_mm=diam,
            n_annotations=len(anns),
            texture_mean=tex,
            texture_class=texture_label(tex),
            subtlety_mean=float(np.mean([a.subtlety for a in anns])),
            malignancy_mean=float(np.mean([a.malignancy for a in anns])),
            spacing_zyx=spacing,
        )
        yield rec, mask_zyx, bbox_zyx


def to_dict(rec: NoduleRecord) -> dict:
    return asdict(rec)
