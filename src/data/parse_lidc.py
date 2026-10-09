"""
Parse LIDC-IDRI radiologist annotations into consensus nodule masks.

LIDC-IDRI ships four independent radiologist contours per nodule. pylidc
clusters them and `consensus()` merges them at a chosen agreement level.

Prerequisite: configure pylidc once with the DICOM path (~/.pylidcrc).
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class NoduleRecord:
    patient_id: str
    nodule_idx: int
    centroid_zyx: tuple
    bbox_zyx: tuple           # ((z0,z1),(y0,y1),(x0,x1))
    diameter_mm: float
    n_annotations: int        # how many of the 4 radiologists marked it
    malignancy_mean: float    # LIDC attribute, metadata only -- NOT a label we predict
    subtlety_mean: float
    texture_mean: float       # proxy for solid / part-solid / ground-glass


def iter_scans(limit=None):
    """Yield pylidc Scan objects."""
    import pylidc as pl
    q = pl.query(pl.Scan)
    for i, scan in enumerate(q):
        if limit and i >= limit:
            break
        yield scan


def consensus_masks(scan, clevel=0.5, pad=2):
    """
    Return [(NoduleRecord, mask, bbox)] for every nodule in one scan.
    mask is a boolean volume cropped to bbox.
    """
    from pylidc.utils import consensus

    out = []
    for idx, anns in enumerate(scan.cluster_annotations()):
        mask, bbox, _ = consensus(anns, clevel=clevel, pad=pad)
        zz, yy, xx = np.where(mask)
        if zz.size == 0:
            continue
        rec = NoduleRecord(
            patient_id=scan.patient_id,
            nodule_idx=idx,
            centroid_zyx=(float(zz.mean()), float(yy.mean()), float(xx.mean())),
            bbox_zyx=tuple((s.start, s.stop) for s in bbox),
            diameter_mm=float(np.mean([a.diameter for a in anns])),
            n_annotations=len(anns),
            malignancy_mean=float(np.mean([a.malignancy for a in anns])),
            subtlety_mean=float(np.mean([a.subtlety for a in anns])),
            texture_mean=float(np.mean([a.texture for a in anns])),
        )
        out.append((rec, mask, bbox))
    return out


def record_to_dict(rec: NoduleRecord) -> dict:
    return asdict(rec)
