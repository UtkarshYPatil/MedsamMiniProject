"""
Build the processed dataset: consensus masks + ROI crops + a manifest.

Run after both LIDC-IDRI downloads are complete (DICOM images AND the
radiologist XML annotations) and pylidc is configured.
"""
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.config import CFG
from src.data.parse_lidc import iter_scans, consensus_masks, record_to_dict
from src.data.preprocess import apply_window, crop_roi


def main():
    out = Path(CFG.paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for scan in tqdm(list(iter_scans(limit=CFG.eval.n_scans)), desc="scans"):
        vol_hu = scan.to_volume().transpose(2, 0, 1)   # -> (Z, Y, X)
        spacing = (scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing)

        for rec, mask, bbox in consensus_masks(
            scan, clevel=CFG.annotation.consensus_level
        ):
            if rec.diameter_mm < CFG.annotation.min_nodule_diameter_mm:
                continue

            full = np.zeros(vol_hu.shape, dtype=bool)
            full[bbox] = mask

            img_roi, origin = crop_roi(
                vol_hu, rec.centroid_zyx,
                size=CFG.preprocess.roi_size, depth=CFG.preprocess.roi_depth,
            )
            msk_roi, _ = crop_roi(
                full, rec.centroid_zyx,
                size=CFG.preprocess.roi_size, depth=CFG.preprocess.roi_depth,
            )

            key = f"{rec.patient_id}_n{rec.nodule_idx}"
            np.savez_compressed(
                out / f"{key}.npz",
                image_hu=img_roi.astype(np.int16),
                mask=msk_roi,
                spacing=np.array(spacing),
                origin=np.array(origin),
            )
            d = record_to_dict(rec)
            d["key"] = key
            d["spacing_zyx"] = spacing
            manifest.append(d)

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(manifest)} nodules -> {out}")


if __name__ == "__main__":
    main()
