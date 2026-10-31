"""
Sweep the mask binarisation threshold. The cheapest shot at Dice we have.

    python scripts/14_ablate_threshold.py --limit 200
    python scripts/14_ablate_threshold.py --thresholds -1,0,0.5,1,1.5,2,3

WHY THIS IS THE FIRST THING TO TRY
----------------------------------
src/models/medsam2.py binarised the model's output at `logits > 0.0` --
sigmoid 0.5 -- and that number was never chosen, it was inherited. Meanwhile
across 902 nodules:

    diameter error   -0.01 mm, 92% within 1 mm      unbiased
    volume ratio     1.08 median                    8% too large

A systematically over-sized mask and an unexamined threshold in front of it.
Raising the threshold shrinks the mask everywhere -- in-plane and axially --
which matters because the axial trim (scripts/13) recovered only about 3 of
the 17 points of excess, so most of it is NOT a trimmable tail.

One inference per nodule; every threshold is applied to the same logits, so
the comparison carries no run-to-run variation and the sweep costs one
baseline.

Nothing is trained. This is a comparison operator.

READING IT
----------
Watch dice and volume_ratio together, and harmed alongside the mean. A
threshold that lifts the mean while making a tenth of nodules worse is a bad
trade for a tool that reports one measurement per nodule to a radiologist.
"""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import bbox_from_mask, best_slice
from src.models.medsam2 import build
from src.models.medsam2 import largest_connected_component as lcc
from src.evaluate import metrics as M
from src.report.sizing import summarize, as_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--thresholds", default="-1,-0.5,0,0.5,1,1.5,2,3")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ths = sorted({float(x) for x in a.thresholds.split(",") if x.strip()} | {0.0})

    proc = Path(CFG.paths.processed)
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))[: a.limit]
    seg = build(CFG.model.checkpoint, CFG.model.cfg_name,
                CFG.model.device, dry_run=a.dry_run).load()
    if a.dry_run:
        print("!! DRY RUN -- numbers are meaningless\n")
    print(f"{len(manifest)} nodules x {len(ths)} thresholds "
          f"(one inference each)\n")

    rows = []
    for item in tqdm(manifest, desc="nodules"):
        d = np.load(proc / item["file"])
        vol_hu, gt, spacing = d["image_hu"].astype(np.float32), d["mask"], d["spacing"]
        if not gt.any() or item.get("n_annotations", 0) > 4:
            continue
        vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level,
                              CFG.preprocess.window_width)
        z = best_slice(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            continue

        _, logits = seg.segment_volume(vol_u8, z, box,
                                       propagate=CFG.prompt.propagate,
                                       return_logits=True)
        gt_vol = as_dict(summarize(gt, vol_hu, spacing))["volume_mm3"]

        for t in ths:
            m = logits > t
            if m.any():
                m = lcc(m)                      # same post-step as the baseline
            meas = as_dict(summarize(m, vol_hu, spacing)) if m.any() else {}
            rows.append({
                "key": item["key"], "threshold": t,
                "dice": M.dice(m, gt),
                "volume_ratio": (meas.get("volume_mm3", 0.0) / gt_vol) if gt_vol else np.nan,
                "diameter_mean_mm": meas.get("diameter_mean_mm", 0.0),
                "gt_diameter_mean_mm": as_dict(summarize(gt, vol_hu, spacing))["diameter_mean_mm"],
                "is_empty": not m.any(),
            })

    if not rows:
        raise SystemExit("nothing evaluated")
    df = pd.DataFrame(rows)
    out = Path(CFG.paths.outputs); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_threshold.csv", index=False)

    base = df[df.threshold == 0.0].set_index("key")
    n = len(base)
    print("\n" + "=" * 76)
    print(f"n = {n} nodules     (threshold 0.0 is the shipped default)")
    print("=" * 76)
    print(f"  {'thresh':>7} {'dice':>7} {'d dice':>8} {'vol ratio':>10} "
          f"{'diam err':>9} {'helped':>7} {'harmed':>7} {'empty':>6}")
    # NB: the column is is_empty, not empty -- DataFrame.empty is a property
    # that returns a bool, so g.empty silently gives the frame's emptiness
    # instead of the column and fails with 'bool has no attribute sum'.
    print("  " + "-" * 68)
    best = None
    for t in ths:
        g = df[df.threshold == t].set_index("key")
        dice = g.dice.mean()
        diff = (g.dice - base.dice).dropna()
        derr = (g.diameter_mean_mm - g.gt_diameter_mean_mm).mean()
        print(f"  {t:>7.2f} {dice:>7.3f} {dice - base.dice.mean():>+8.3f} "
              f"{g.volume_ratio.replace([np.inf,-np.inf],np.nan).median():>10.2f} "
              f"{derr:>+9.2f} {int((diff > 1e-9).sum()):>7} "
              f"{int((diff < -1e-9).sum()):>7} {int(g["is_empty"].sum()):>6}")
        if best is None or dice > best[1]:
            best = (t, dice)
    print("=" * 76)

    t, dice = best
    g = df[df.threshold == t].set_index("key")
    diff = (g.dice - base.dice).dropna()
    harmed = int((diff < -1e-9).sum())
    delta = dice - base.dice.mean()
    print(f"\nBest: threshold {t:+.2f} -> Dice {dice:.3f} ({delta:+.3f})")
    print(f"  helped {int((diff > 1e-9).sum())}, harmed {harmed} ({harmed/n:.1%})")
    print(f"  volume ratio {base.volume_ratio.median():.2f} -> "
          f"{g.volume_ratio.median():.2f}")
    if t == 0.0:
        print("\n  The shipped default is already best. Good to know, and worth")
        print("  saying in the report -- it was never checked before.")
    elif delta < 0.005:
        print("\n  Within noise. Leave it at 0.")
    elif harmed > n * 0.10:
        print(f"\n  Mean improves but {harmed/n:.0%} get worse. Prefer a gentler value.")
    else:
        print(f"\n  Worth taking. Thread mask_threshold={t} through "
              "segment_volume's callers,\n  re-run 03_run_baseline.py, update the model card.")
    print(f"\nwrote {out / 'ablation_threshold.csv'}")


if __name__ == "__main__":
    main()
