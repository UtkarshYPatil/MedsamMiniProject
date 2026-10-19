"""
Ablation: what is costing us Dice?

    python scripts/06_ablate.py                      sweep box padding
    python scripts/06_ablate.py --limit 30

The first baseline gave Dice 0.534 with a systematic +3.27 mm diameter bias --
predictions are consistently LARGER than the reference. Two candidate causes:

  1. BOX PADDING. We pad the prompt box by 5 px. At ~0.7 mm/px that is +7 mm on
     the diameter, and the observed bias is 47% of it. The official reference
     (medsam2_infer_3D_CT.py) has its box expansion commented out -- it prompts
     with the tight box.

  2. UPSCALING. Our ROI is 128 px and MedSAM2's input is 512 px, so every crop
     is upsampled 4x before the model sees it. The reference feeds native
     512 px slices with no scaling, so the lesion occupies a small part of the
     frame rather than filling it.

This script settles (1) directly on the existing processed data. For (2),
re-run 02_prepare_data.py with a larger roi_size and compare -- see the note
printed at the end.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# sam2 is loud: missing _C extension, flash-attention fallbacks. None of it
# affects results and it drowns the numbers we care about.
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import bbox_from_mask, best_slice
from src.models.medsam2 import build
from src.evaluate import metrics as M
from src.evaluate.metrics import mean_diameter_mm

PADDINGS = [0, 2, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paddings", type=int, nargs="+", default=PADDINGS)
    ap.add_argument("--no-largest-cc", action="store_true",
                    help="also test without largest-connected-component filtering")
    a = ap.parse_args()

    proc = Path(CFG.paths.processed)
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    if a.limit:
        manifest = manifest[: a.limit]

    cases = []
    for item in manifest:
        d = np.load(proc / item["file"])
        gt = d["mask"]
        if not gt.any():
            continue
        cases.append((item, d["image_hu"].astype(np.float32), gt, d["spacing"]))
    print(f"{len(cases)} nodules\n")

    seg = build(CFG.model.checkpoint, CFG.model.cfg_name, CFG.model.device).load()

    settings = [(p, True) for p in a.paddings]
    if a.no_largest_cc:
        settings += [(p, False) for p in a.paddings]

    rows = []
    for pad, largest in settings:
        label = f"pad={pad}" + ("" if largest else " noCC")
        per = []
        for item, vol_hu, gt, spacing in tqdm(cases, desc=f"{label:<14}", leave=False):
            vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level,
                                  CFG.preprocess.window_width)
            z = best_slice(gt)
            box = bbox_from_mask(gt[z], padding=pad)
            if box is None:
                continue
            pred = seg.segment_volume(vol_u8, z, box,
                                      propagate=CFG.prompt.propagate,
                                      keep_largest=largest)
            per.append({
                "key": item["key"],
                "texture": item["texture_class"],
                "dice": M.dice(pred, gt),
                "iou": M.iou(pred, gt),
                "hd95": M.hd95(pred, gt, spacing=tuple(float(s) for s in spacing)),
                "d_pred": mean_diameter_mm(pred, spacing),
                "d_gt": mean_diameter_mm(gt, spacing),
            })

        df = pd.DataFrame(per)
        err = df["d_pred"] - df["d_gt"]
        rows.append({
            "setting": label,
            "padding": pad,
            "largest_cc": largest,
            "n": len(df),
            "dice_mean": df["dice"].mean(),
            "dice_std": df["dice"].std(),
            "dice_median": df["dice"].median(),
            "iou_mean": df["iou"].mean(),
            "hd95_mean": df["hd95"].mean(),
            "diam_bias_mm": err.mean(),
            "diam_abs_mm": err.abs().mean(),
            "within_1mm_pct": (err.abs() <= 1).mean() * 100,
        })
        r = rows[-1]
        print(f"  {label:<14} dice {r['dice_mean']:.3f} +/- {r['dice_std']:.3f}   "
              f"bias {r['diam_bias_mm']:+.2f} mm   within1mm {r['within_1mm_pct']:.0f}%")

    out = Path(CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "ablation_padding.csv", index=False)

    print("\n" + "=" * 70)
    print(res[["setting", "n", "dice_mean", "dice_median",
               "diam_bias_mm", "within_1mm_pct"]].to_string(index=False))

    best = res.loc[res["dice_mean"].idxmax()]
    print("\n" + "=" * 70)
    print(f"BEST: {best['setting']}  ->  dice {best['dice_mean']:.3f} "
          f"(baseline pad=5 was {res[res['padding']==5]['dice_mean'].iloc[0]:.3f})"
          if (res["padding"] == 5).any() else
          f"BEST: {best['setting']}  ->  dice {best['dice_mean']:.3f}")
    print(f"\nSet  prompt.bbox_padding_px: {int(best['padding'])}  in configs/config.yaml")

    print("\n--- hypothesis 2: upscaling ---")
    print(f"ROI is {CFG.preprocess.roi_size} px; MedSAM2 input is 512 px "
          f"({512/CFG.preprocess.roi_size:.0f}x upsample).")
    print("The reference implementation feeds native 512 px slices. To test:")
    print("    1. set preprocess.roi_size: 256   in configs/config.yaml")
    print("    2. python scripts/02_prepare_data.py --limit 10 --fresh")
    print("    3. python scripts/03_run_baseline.py")
    print("  then compare Dice against this run. Keep whichever wins and say so")
    print("  in the report -- that comparison IS a result.")
    print(f"\nwrote {out/'ablation_padding.csv'}")


if __name__ == "__main__":
    main()
