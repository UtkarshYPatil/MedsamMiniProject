"""
Does the one-button auto-prompt cost accuracy? Measure it.

    python scripts/11_autoprompt_vs_consensus.py
    python scripts/11_autoprompt_vs_consensus.py --limit 30

WHY THIS EXISTS
---------------
Every number in outputs/results.csv was produced with a prompt box derived from
the CONSENSUS MASK -- that is, from the answer. The UI cannot do that. It calls
/suggest, which finds the nodule from the image alone, and that box is not the
same box.

On LIDC-IDRI-0136_n7 the difference showed: the UI produced 211 voxels and
442 mm3 where the baseline produced 155 and 325, against a reference of 117 and
245. Mean diameter barely moved (8.1 vs 7.6 vs 7.9 mm) because volume goes as
the cube of the boundary, so a mask one voxel generous all round hardly shifts
the diameter and nearly doubles the volume.

That was one nodule. This runs both prompts over the whole manifest so the cost
is a measured number rather than an anecdote.

WHAT TO LOOK AT
---------------
  dice_consensus - dice_auto    how much accuracy the convenience costs
  box_area_ratio                is the auto box systematically larger?
  found_rate                    how often it locates the nodule at all
  volume_ratio_*                over- or under-estimation against the reference

The padding ablation already showed Dice falling 0.804 -> 0.266 as box padding
grows 0 to 10 px, because MedSAM2 fills toward the box it is given. If
box_area_ratio is above 1 and dice_auto is below dice_consensus, that is the
same mechanism and the fix is to tighten suggest_box, not to retrain anything.

Run it on the machine with the dataset. It needs a GPU to be quick: two
inferences per nodule, so about twice a baseline run.
"""
import argparse
import json
import sys
import time
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
from src.evaluate import metrics as M
from src.report.sizing import summarize, as_dict
from backend.common import autoprompt


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="dummy segmenter -- checks the plumbing, numbers are meaningless")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    proc = Path(CFG.paths.processed)
    man_path = proc / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest at {man_path} -- run scripts/02_prepare_data.py first")

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if a.limit:
        manifest = manifest[: a.limit]
    print(f"{len(manifest)} nodules, two prompts each\n")

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

        # --- prompt A: from the consensus mask (what the baseline used) ------
        z_c = best_slice(gt)
        box_c = bbox_from_mask(gt[z_c], padding=CFG.prompt.bbox_padding_px)
        if box_c is None:
            continue

        # --- prompt B: from the image alone (what the UI uses) ---------------
        z_a, box_a, info = autoprompt.suggest_box(vol_hu)
        found = box_a is not None
        if not found:
            z_a, box_a = autoprompt.centred_fallback(vol_hu.shape)

        t0 = time.perf_counter()
        pred_c = seg.segment_volume(vol_u8, z_c, box_c, propagate=CFG.prompt.propagate)
        t_c = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_a = seg.segment_volume(vol_u8, z_a, box_a, propagate=CFG.prompt.propagate)
        t_a = time.perf_counter() - t0

        m_c = as_dict(summarize(pred_c, vol_hu, spacing)) if pred_c.any() else {}
        m_a = as_dict(summarize(pred_a, vol_hu, spacing)) if pred_a.any() else {}
        m_gt = as_dict(summarize(gt, vol_hu, spacing))

        gt_vol = m_gt.get("volume_mm3") or float("nan")

        rows.append({
            "key": item["key"],
            "texture_class_lidc": item["texture_class"],
            "n_annotations": item["n_annotations"],

            "dice_consensus": M.dice(pred_c, gt),
            "dice_auto": M.dice(pred_a, gt),
            "iou_consensus": M.iou(pred_c, gt),
            "iou_auto": M.iou(pred_a, gt),

            # Did the auto-prompt even look at the right thing?
            "found": found,
            "same_slice": int(z_a) == int(z_c),
            "slice_offset": int(z_a) - int(z_c),
            "box_iou": box_iou(box_a, box_c),
            "box_area_ratio": (box_area(box_a) / box_area(box_c)
                               if box_area(box_c) > 0 else float("nan")),

            # The measurements a radiologist would actually read.
            "diam_gt_mm": m_gt.get("diameter_mean_mm"),
            "diam_consensus_mm": m_c.get("diameter_mean_mm"),
            "diam_auto_mm": m_a.get("diameter_mean_mm"),
            "vol_gt_mm3": gt_vol,
            "vol_consensus_mm3": m_c.get("volume_mm3"),
            "vol_auto_mm3": m_a.get("volume_mm3"),
            "volume_ratio_consensus": (m_c.get("volume_mm3") or np.nan) / gt_vol,
            "volume_ratio_auto": (m_a.get("volume_mm3") or np.nan) / gt_vol,

            "conf_auto": m_a.get("confidence"),
            "flags_auto": m_a.get("flags"),
            "seconds_consensus": round(t_c, 3),
            "seconds_auto": round(t_a, 3),
        })

    if not rows:
        raise SystemExit("no nodules evaluated")

    outdir = Path(a.out or CFG.paths.outputs)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tag = "_dryrun" if a.dry_run else ""
    df.to_csv(outdir / f"autoprompt_vs_consensus{tag}.csv", index=False)

    # ----------------------------------------------------------------- report
    n = len(df)
    dc, da = df["dice_consensus"].dropna(), df["dice_auto"].dropna()
    delta = (df["dice_consensus"] - df["dice_auto"]).dropna()

    print("\n" + "=" * 64)
    print(f"n = {n} nodules\n")
    print(f"  Dice, consensus box : {dc.mean():.3f}  (median {dc.median():.3f}, std {dc.std():.3f})")
    print(f"  Dice, auto box      : {da.mean():.3f}  (median {da.median():.3f}, std {da.std():.3f})")
    print(f"  cost of one-button  : {delta.mean():+.3f} mean, {delta.median():+.3f} median")
    print()
    print(f"  found a candidate   : {100 * df['found'].mean():.1f}%  "
          f"({int((~df['found']).sum())} fell back to a centred box)")
    print(f"  same slice as consensus: {100 * df['same_slice'].mean():.1f}%")
    print(f"  box IoU vs consensus: {df['box_iou'].mean():.3f} mean, "
          f"{df['box_iou'].median():.3f} median")
    print(f"  box area ratio      : {df['box_area_ratio'].median():.2f} median "
          f"({df['box_area_ratio'].quantile(0.9):.2f} at p90)")
    print()
    vrc = df["volume_ratio_consensus"].replace([np.inf, -np.inf], np.nan).dropna()
    vra = df["volume_ratio_auto"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  volume / reference, consensus box : {vrc.median():.2f} median")
    print(f"  volume / reference, auto box      : {vra.median():.2f} median")
    print("=" * 64)

    # Where it goes wrong matters more than the average.
    bad = df[df["dice_auto"] < 0.5].sort_values("dice_auto")
    print(f"\n{len(bad)} nodules scored under 0.5 with the auto box "
          f"({100 * len(bad) / n:.1f}%)")
    if len(bad):
        print("  worst ten -- check whether the box landed on the wrong object:")
        for _, r in bad.head(10).iterrows():
            print(f"    {r['key']:<22} dice {r['dice_auto']:.3f} "
                  f"(consensus {r['dice_consensus']:.3f})  "
                  f"box_iou {r['box_iou']:.2f}  slice_off {r['slice_offset']:+d}  "
                  f"{'FOUND' if r['found'] else 'fallback'}")

    # Did the confidence flags catch the failures? That is what they are for.
    if "conf_auto" in df and df["conf_auto"].notna().any():
        print("\nDid the confidence flag catch the bad ones?")
        tab = df.assign(poor=df["dice_auto"] < 0.5) \
                .groupby("conf_auto")["poor"].agg(["count", "sum"])
        tab["share_poor"] = (tab["sum"] / tab["count"]).round(3)
        print(tab.to_string())
        high_bad = df[(df["conf_auto"] == "high") & (df["dice_auto"] < 0.5)]
        if len(high_bad):
            print(f"\n  {len(high_bad)} nodules scored HIGH confidence and under 0.5 Dice.")
            print("  Those are the dangerous ones: wrong, and not flagged as wrong.")
            for k in high_bad["key"].head(10):
                print(f"    {k}")
        else:
            print("\n  No nodule scored HIGH confidence while under 0.5 Dice.")

    print(f"\nwrote {outdir / f'autoprompt_vs_consensus{tag}.csv'}")


if __name__ == "__main__":
    main()
