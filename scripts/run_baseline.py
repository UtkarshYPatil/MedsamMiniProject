"""
Phase 2 baseline: frozen MedSAM2 on LIDC-IDRI nodule ROIs.

Prompt = bounding box derived from the consensus mask on the largest slice.
This is NOT end-to-end -- the automatic detector is Phase 3. Report it as
"segmentation quality given a correct box".
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG
from src.data.preprocess import apply_window, to_rgb
from src.prompts.bbox import bbox_from_mask, centre_slice_index
from src.models.medsam2 import MedSAM2Segmenter
from src.evaluate import metrics as M


def main():
    proc = Path(CFG.paths.processed)
    manifest = json.load(open(proc / "manifest.json"))

    seg = MedSAM2Segmenter(
        CFG.model.checkpoint, CFG.model.config, CFG.model.device
    ).load()

    rows = []
    for item in tqdm(manifest, desc="nodules"):
        d = np.load(proc / f"{item['key']}.npz")
        vol_hu, gt, spacing = d["image_hu"], d["mask"], d["spacing"]

        if not gt.any():
            continue

        vol_u8 = apply_window(
            vol_hu, CFG.preprocess.window_level, CFG.preprocess.window_width
        )
        vol_rgb = np.stack([to_rgb(s) for s in vol_u8])

        z = centre_slice_index(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            continue

        pred = seg.segment_volume(vol_rgb, z, box)

        rows.append({
            "key": item["key"],
            "diameter_mm_gt": item["diameter_mm"],
            "n_annotations": item["n_annotations"],
            "texture_mean": item["texture_mean"],
            "dice": M.dice(pred, gt),
            "iou": M.iou(pred, gt),
            "hd95": M.hd95(pred, gt, spacing=tuple(spacing)),
            "volume_mm3_pred": M.volume_mm3(pred, spacing),
            "volume_mm3_gt": M.volume_mm3(gt, spacing),
            "diameter_mm_pred": M.max_diameter_mm(pred, spacing),
            "mean_hu_pred": M.mean_hu(vol_hu, pred),
        })

    out = Path(CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    df, summary = M.summarize(rows)
    df.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
