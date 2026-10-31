"""
Does trimming the propagation tail recover accuracy, and at what cost?

    python scripts/13_ablate_propagation.py
    python scripts/13_ablate_propagation.py --limit 200
    python scripts/13_ablate_propagation.py --fracs 0,0.05,0.10,0.15,0.25

WHAT THIS TESTS
---------------
On 902 nodules the error has a shape:

    diameter error   -0.01 mm, 92% within 1 mm      unbiased
    volume ratio     1.12 median, 1.90 at p90       12-90% too large
    r(dice, volume ratio)   -0.65
    r(dice, diameter ratio) -0.26

Diameter is in-plane; volume is area x depth. Right diameter and inflated
volume puts the excess on slices where the nodule is not -- MedSAM2's memory
attention carries the mask past the end of the lesion.

src/models/postprocess.py cuts that tail: walking out from the prompt slice,
a run of more than `max_tail` slices below a fraction of the prompt slice's
area is drift and is discarded. A genuine taper collapses to zero within a
slice or two and is kept. This sweeps the fraction.

NO INFERENCE IS RE-RUN. The model is called once per nodule and every
threshold is applied to that same mask, so the comparison is exact rather
than subject to run-to-run variation, and the sweep costs one baseline.

HOW TO READ IT
--------------
Trimming can only remove voxels, so it helps where the mask over-extends and
hurts where it does not. Watch three things together:

  dice         should rise then fall as the threshold gets aggressive
  volume_ratio should fall toward 1.00
  harmed       nodules whose Dice got WORSE -- the cost, and the honest
               reason not to simply take the best mean

A threshold that lifts the mean while harming a fifth of cases is not an
improvement for a tool that reports per-nodule measurements to a radiologist.
"""
import argparse
import json
import sys
import warnings
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
from src.models.postprocess import trim_axial_extent
from src.evaluate import metrics as M
from src.report.sizing import summarize, as_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fracs", default="0,0.05,0.10,0.15,0.20,0.25,0.35",
                    help="comma-separated area fractions to sweep")
    ap.add_argument("--max-tail", type=int, default=2,
                    help="consecutive small slices allowed before it is drift")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fracs = [float(x) for x in a.fracs.split(",") if x.strip()]
    if 0.0 not in fracs:
        fracs.insert(0, 0.0)            # the untrimmed baseline to compare to

    proc = Path(CFG.paths.processed)
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    if a.limit:
        manifest = manifest[: a.limit]
    print(f"{len(manifest)} nodules x {len(fracs)} thresholds "
          f"(one inference each, thresholds applied to the same mask)\n")

    seg = build(CFG.model.checkpoint, CFG.model.cfg_name,
                CFG.model.device, dry_run=a.dry_run).load()
    if a.dry_run:
        print("!! DRY RUN -- dummy segmenter. Numbers are meaningless.\n")

    rows = []
    for item in tqdm(manifest, desc="nodules"):
        d = np.load(proc / item["file"])
        vol_hu, gt, spacing = d["image_hu"].astype(np.float32), d["mask"], d["spacing"]
        if not gt.any():
            continue
        if item.get("n_annotations", 0) > 4:
            continue                    # merged cluster: two nodules in one

        vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level,
                              CFG.preprocess.window_width)
        z = best_slice(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            continue

        raw = seg.segment_volume(vol_u8, z, box, propagate=CFG.prompt.propagate)
        if not raw.any():
            continue
        gt_vol = as_dict(summarize(gt, vol_hu, spacing))["volume_mm3"]

        for f in fracs:
            m = raw if f == 0 else trim_axial_extent(raw, z, f, max_tail=a.max_tail)
            meas = as_dict(summarize(m, vol_hu, spacing)) if m.any() else {}
            rows.append({
                "key": item["key"],
                "frac": f,
                "dice": M.dice(m, gt),
                "iou": M.iou(m, gt),
                "volume_mm3": meas.get("volume_mm3", 0.0),
                "volume_ratio": (meas.get("volume_mm3", 0.0) / gt_vol) if gt_vol else np.nan,
                "diameter_mean_mm": meas.get("diameter_mean_mm", 0.0),
                "n_voxels": meas.get("n_voxels", 0),
                "empty": not m.any(),
            })

    if not rows:
        raise SystemExit("nothing evaluated")

    df = pd.DataFrame(rows)
    out = Path(a.out or CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_dryrun" if a.dry_run else ""
    df.to_csv(out / f"ablation_propagation{tag}.csv", index=False)

    # ------------------------------------------------------------- report
    base = df[df.frac == 0].set_index("key")
    n = len(base)
    print("\n" + "=" * 74)
    print(f"n = {n} nodules   max_tail = {a.max_tail}")
    print("=" * 74)
    print(f"  {'frac':>6} {'dice':>7} {'d dice':>8} {'vol ratio':>10} "
          f"{'helped':>7} {'harmed':>7} {'emptied':>8}")
    print("  " + "-" * 64)

    best = None
    for f in sorted(fracs):
        g = df[df.frac == f].set_index("key")
        dice = g["dice"].mean()
        delta = dice - base["dice"].mean()
        vr = g["volume_ratio"].replace([np.inf, -np.inf], np.nan).median()
        diff = (g["dice"] - base["dice"]).dropna()
        helped = int((diff > 1e-9).sum())
        harmed = int((diff < -1e-9).sum())
        emptied = int(g["empty"].sum())
        flag = ""
        if f > 0 and (best is None or dice > best[1]):
            best, flag = (f, dice), ""
        print(f"  {f:>6.2f} {dice:>7.3f} {delta:>+8.3f} {vr:>10.2f} "
              f"{helped:>7} {harmed:>7} {emptied:>8}{flag}")

    print("=" * 74)
    if best is None:
        print("only the baseline was evaluated")
        return

    f, dice = best
    g = df[df.frac == f].set_index("key")
    diff = (g["dice"] - base["dice"]).dropna()
    helped, harmed = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
    delta = dice - base["dice"].mean()

    print(f"\nBest mean Dice at frac={f:.2f}: {dice:.3f} ({delta:+.3f})")
    print(f"  helped {helped} nodules, harmed {harmed} "
          f"({harmed / n:.1%} of the set)")
    print(f"  worst single regression: {diff.min():+.3f}")

    if delta <= 0.002:
        print("\n  Not worth taking. The gain is within noise; leave")
        print("  trim_area_frac at 0 and report the mechanism as a known")
        print("  limitation instead of a fix.")
    elif harmed > n * 0.10:
        print(f"\n  Mean improves but {harmed / n:.0%} of nodules get worse.")
        print("  For a tool that reports a measurement per nodule to a")
        print("  radiologist, that trade is worse than it looks in the mean.")
        print("  Consider a gentler threshold even at a smaller gain.")
    else:
        print(f"\n  Worth taking. Set prompt.trim_area_frac: {f:.2f} in")
        print("  configs/config.yaml, re-run scripts/03_run_baseline.py, and")
        print("  update docs/model_card.md from the new numbers.")

    print(f"\nwrote {out / f'ablation_propagation{tag}.csv'}")


if __name__ == "__main__":
    main()
