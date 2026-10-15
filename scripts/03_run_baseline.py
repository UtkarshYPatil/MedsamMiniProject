"""
Baseline: frozen MedSAM2 on LIDC-IDRI nodule ROIs.

    python scripts/03_run_baseline.py                 # real model
    python scripts/03_run_baseline.py --dry-run       # no GPU, pipeline test only
    python scripts/03_run_baseline.py --limit 20

Prompt = bounding box from the consensus mask on the largest-area slice.
That means these numbers are "segmentation quality GIVEN a correct box",
NOT end-to-end detection performance. Report them that way.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# sam2 is loud: a missing optional _C extension and flash-attention fallbacks.
# Neither affects results; the spam buries the numbers.
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import bbox_from_mask, best_slice
from src.models.medsam2 import build
from src.evaluate import metrics as M
from src.report.sizing import summarize, as_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="use the dummy segmenter -- for pipeline testing only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proc = Path(CFG.paths.processed)
    man_path = proc / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest at {man_path} -- run scripts/02_prepare_data.py first")

    manifest = json.loads(man_path.read_text())
    if a.limit:
        manifest = manifest[: a.limit]
    print(f"{len(manifest)} nodules")

    seg = build(CFG.model.checkpoint, CFG.model.cfg_name,
                CFG.model.device, dry_run=a.dry_run).load()
    if a.dry_run:
        print("!! DRY RUN -- dummy segmenter. Results are meaningless.")

    rows = []
    for item in tqdm(manifest, desc="nodules"):
        d = np.load(proc / item["file"])
        vol_hu, gt, spacing = d["image_hu"].astype(np.float32), d["mask"], d["spacing"]
        if not gt.any():
            continue

        vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level,
                              CFG.preprocess.window_width)

        z = best_slice(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            continue

        t0 = time.perf_counter()
        pred = seg.segment_volume(vol_u8, z, box, propagate=CFG.prompt.propagate)
        dt = time.perf_counter() - t0

        meas_pred = summarize(pred, vol_hu, spacing)
        meas_gt = summarize(gt, vol_hu, spacing)

        row = {
            "key": item["key"],
            "patient_id": item["patient_id"],
            "n_annotations": item["n_annotations"],
            "texture_class_lidc": item["texture_class"],
            "diameter_mm_lidc": item["diameter_mm"],
            "dice": M.dice(pred, gt),
            "iou": M.iou(pred, gt),
            "hd95_mm": M.hd95(pred, gt, spacing=tuple(float(s) for s in spacing)),
            "seconds": round(dt, 3),
        }
        row.update({f"pred_{k}": v for k, v in as_dict(meas_pred).items()})
        row.update({f"gt_{k}": v for k, v in as_dict(meas_gt).items()})
        rows.append(row)

    if not rows:
        raise SystemExit("no nodules evaluated")

    outdir = Path(a.out or CFG.paths.outputs)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tag = "_dryrun" if a.dry_run else ""
    df.to_csv(outdir / f"results{tag}.csv", index=False)

    # headline numbers
    print("\n" + "=" * 58)
    print(f"n = {len(df)} nodules")
    for m in ("dice", "iou", "hd95_mm"):
        v = df[m].dropna()
        if len(v):
            print(f"  {m:<9} mean {v.mean():.3f}   std {v.std():.3f}   median {v.median():.3f}")
    print(f"  {'seconds':<9} mean {df['seconds'].mean():.2f}")

    # the split that actually matters -- ground-glass is the hard case
    print("\nDice by LIDC texture class:")
    g = df.groupby("texture_class_lidc")["dice"].agg(["count", "mean", "std"])
    print(g.to_string())
    g.to_csv(outdir / f"dice_by_texture{tag}.csv")

    # measurement agreement
    if "pred_diameter_mean_mm" in df and "gt_diameter_mean_mm" in df:
        err = df["pred_diameter_mean_mm"] - df["gt_diameter_mean_mm"]
        print(f"\nDiameter error  mean {err.mean():+.2f} mm   "
              f"abs-mean {err.abs().mean():.2f} mm   "
              f"within 1mm: {(err.abs() <= 1).mean()*100:.0f}%")

    # density classifier vs LIDC texture rating
    if "pred_density_class" in df:
        agree = (df["pred_density_class"] == df["texture_class_lidc"]).mean()
        print(f"\nDensity-class agreement with LIDC texture: {agree*100:.0f}%")

    # --- everything docs/model_card.md needs, in one block ---------------
    print("\n" + "=" * 58)
    print("MODEL CARD VALUES  (copy into docs/model_card.md)")
    print("=" * 58)
    print(f"  Nodules evaluated : {len(df)}")
    print(f"  Scans             : {df['patient_id'].nunique()}")
    for m, label in (("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)")):
        v = df[m].dropna()
        if len(v):
            print(f"  {label:<18}: mean {v.mean():.3f} | std {v.std():.3f} | median {v.median():.3f}")
    if "pred_diameter_mean_mm" in df:
        err = df["pred_diameter_mean_mm"] - df["gt_diameter_mean_mm"]
        print(f"  Diameter err mean : {err.mean():+.2f} mm")
        print(f"  Diameter err |mean|: {err.abs().mean():.2f} mm")
        print(f"  Within 1 mm       : {(err.abs() <= 1).mean()*100:.0f} %")
    print(f"  Seconds/nodule    : {df['seconds'].mean():.2f}")
    print("\n  By texture class:")
    for cls, row in df.groupby("texture_class_lidc")["dice"].agg(["count","mean","std"]).iterrows():
        print(f"    {cls:<14} n={int(row['count']):<4} dice {row['mean']:.3f} +/- "
              f"{0.0 if pd.isna(row['std']) else row['std']:.3f}")

    print(f"\nwrote {outdir/f'results{tag}.csv'}")


if __name__ == "__main__":
    main()
