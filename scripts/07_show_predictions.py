"""
Render what the model actually outputs.

    python scripts/07_show_predictions.py                 best / median / worst
    python scripts/07_show_predictions.py --keys LIDC-IDRI-0136_n0 LIDC-IDRI-0069_n1
    python scripts/07_show_predictions.py --n 6

Produces two figures in outputs/figures/:

  model_output.png   per nodule: the CT slice, the prompt box we give the model,
                     the radiologist consensus, MedSAM2's prediction, and the two
                     overlaid with the Dice score.

  propagation.png    one nodule across the volume. The model is prompted on a
                     SINGLE slice (marked); every other slice is produced by
                     memory propagation with no further input. This is the
                     figure that shows the output is a 3D mask, not a stack of
                     independent 2D guesses.

Predicted masks are cached to outputs/masks/<key>.npz so re-running the figures
does not need the GPU again.

Note on sanity_check.png (the earlier figure): that one shows the radiologist
consensus contours ONLY -- it was the data-verification step. This script is the
first one that draws model predictions.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# sam2 is loud: missing optional _C extension, flash-attention fallbacks.
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import bbox_from_mask, best_slice
from src.models.medsam2 import build
from src.evaluate import metrics as M
from src.report.sizing import summarize, as_dict

GT_COLOR = "#00E5A0"      # radiologist consensus
PRED_COLOR = "#FF4D4D"    # MedSAM2
BOX_COLOR = "#FFD400"     # the prompt


def outline(ax, mask2d, color, lw=1.6):
    """Draw a mask boundary. Silent no-op on an empty mask."""
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#444")


def pick_keys(manifest, n, explicit, outdir):
    """Best / worst / evenly-spaced-between, by Dice, from the baseline run."""
    if explicit:
        return explicit

    res = outdir / "results.csv"
    keys = [m["key"] for m in manifest]
    if not res.exists():
        print("  (no results.csv -- taking the first few from the manifest)")
        return keys[:n]

    df = pd.read_csv(res).dropna(subset=["dice"]).sort_values("dice")
    df = df[df["key"].isin(keys)]
    if df.empty:
        return keys[:n]

    idx = np.unique(np.linspace(0, len(df) - 1, min(n, len(df))).round().astype(int))
    chosen = df.iloc[idx]["key"].tolist()
    # worst first reads as an apology; lead with a good case, keep the worst in.
    return chosen[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=None,
                    help="specific nodule keys; default picks best/median/worst")
    ap.add_argument("--n", type=int, default=4, help="how many nodules to show")
    ap.add_argument("--prop-key", default=None,
                    help="nodule for the propagation figure (default: the best one shown)")
    ap.add_argument("--dry-run", action="store_true",
                    help="dummy segmenter -- layout check only, output is meaningless")
    a = ap.parse_args()

    proc = Path(CFG.paths.processed)
    outdir = Path(CFG.paths.outputs)
    figdir = outdir / "figures"
    maskdir = outdir / "masks"
    figdir.mkdir(parents=True, exist_ok=True)
    maskdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    by_key = {m["key"]: m for m in manifest}

    keys = pick_keys(manifest, a.n, a.keys, outdir)
    keys = [k for k in keys if k in by_key]
    if not keys:
        raise SystemExit("none of those keys are in the manifest")
    print(f"showing {len(keys)} nodules: {', '.join(keys)}\n")

    seg = build(CFG.model.checkpoint, CFG.model.cfg_name,
                CFG.model.device, dry_run=a.dry_run).load()
    if a.dry_run:
        print("!! DRY RUN -- dummy segmenter, the masks below mean nothing.\n")

    # ---------------------------------------------------------------- run
    cases = []
    for key in keys:
        item = by_key[key]
        d = np.load(proc / item["file"])
        vol_hu = d["image_hu"].astype(np.float32)
        gt = d["mask"]
        spacing = d["spacing"]

        vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level,
                              CFG.preprocess.window_width)
        z = best_slice(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            print(f"  skip {key}: empty mask")
            continue

        pred = seg.segment_volume(vol_u8, z, box, propagate=CFG.prompt.propagate)
        np.savez_compressed(maskdir / f"{key}.npz", pred=pred, gt=gt, prompt_z=z,
                            box=np.asarray(box, dtype=np.float32))

        mp = as_dict(summarize(pred, vol_hu, spacing))
        mg = as_dict(summarize(gt, vol_hu, spacing))
        dice = M.dice(pred, gt)

        cases.append(dict(key=key, vol=vol_u8, gt=gt, pred=pred, z=z, box=box,
                          dice=dice, mp=mp, mg=mg,
                          texture=item.get("texture_class", "?")))
        print(f"  {key:<22} dice {dice:.3f}   "
              f"pred {mp['diameter_mean_mm']:.1f} mm vs gt {mg['diameter_mean_mm']:.1f} mm   "
              f"[{mp['confidence']}]")

    if not cases:
        raise SystemExit("nothing to draw")

    # -------------------------------------------------- figure A: output
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.5 * n), facecolor="white")
    axes = np.atleast_2d(axes)

    for r, c in enumerate(cases):
        img = c["vol"][c["z"]]
        g, p = c["gt"][c["z"]], c["pred"][c["z"]]
        x0, y0, x1, y1 = c["box"]

        for col in range(4):
            axes[r, col].imshow(img, cmap="gray", interpolation="nearest")
            bare(axes[r, col])

        # 1. input + prompt
        axes[r, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, edgecolor=BOX_COLOR, lw=1.8))
        # 2. reference
        outline(axes[r, 1], g, GT_COLOR)
        # 3. prediction
        outline(axes[r, 2], p, PRED_COLOR)
        # 4. both
        outline(axes[r, 3], g, GT_COLOR, lw=1.4)
        outline(axes[r, 3], p, PRED_COLOR, lw=1.4)

        axes[r, 0].set_ylabel(f"{c['key'].replace('LIDC-IDRI-', '')}\n{c['texture']}",
                              fontsize=9, color="#222")
        axes[r, 3].text(0.97, 0.04, f"Dice {c['dice']:.3f}", transform=axes[r, 3].transAxes,
                        ha="right", va="bottom", fontsize=11, color="white", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#111", ec="none", alpha=0.75))
        axes[r, 2].text(0.03, 0.04,
                        f"{c['mp']['diameter_mean_mm']:.1f} mm\n{c['mp']['volume_mm3']:.0f} mm³",
                        transform=axes[r, 2].transAxes, ha="left", va="bottom",
                        fontsize=8.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#111", ec="none", alpha=0.7))
        axes[r, 1].text(0.03, 0.04, f"{c['mg']['diameter_mean_mm']:.1f} mm",
                        transform=axes[r, 1].transAxes, ha="left", va="bottom",
                        fontsize=8.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#111", ec="none", alpha=0.7))

    titles = ["CT slice + prompt box\n(the only input)",
              "Radiologist consensus\n(reference, 4 readers)",
              "MedSAM2 output\n(frozen, no training)",
              "Overlaid"]
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=10.5, pad=10, color="#111")

    fig.suptitle("MedSAM2 model output on LIDC-IDRI pulmonary nodules",
                 fontsize=13.5, weight="bold", y=0.997)
    fig.text(0.5, 0.004,
             f"Frozen MedSAM2 ({Path(CFG.model.checkpoint).name}) · prompt = box on one slice, "
             f"padding {CFG.prompt.bbox_padding_px} px · lung window "
             f"{CFG.preprocess.window_level}/{CFG.preprocess.window_width} HU",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.012, 1, 0.985])
    out_a = figdir / "model_output.png"
    fig.savefig(out_a, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out_a}")

    # --------------------------------------------- figure B: propagation
    pick = a.prop_key or max(cases, key=lambda c: c["dice"])["key"]
    c = next((x for x in cases if x["key"] == pick), cases[0])

    zs = [z for z in range(c["pred"].shape[0])
          if c["pred"][z].any() or c["gt"][z].any()]
    if len(zs) > 7:
        zs = list(np.unique(np.linspace(zs[0], zs[-1], 7).round().astype(int)))
    if c["z"] not in zs:
        zs = sorted(set(zs + [c["z"]]))

    fig, axes = plt.subplots(1, len(zs), figsize=(2.05 * len(zs), 2.9), facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, z in zip(axes, zs):
        ax.imshow(c["vol"][z], cmap="gray", interpolation="nearest")
        outline(ax, c["gt"][z], GT_COLOR, lw=1.3)
        outline(ax, c["pred"][z], PRED_COLOR, lw=1.3)
        bare(ax)
        if z == c["z"]:
            x0, y0, x1, y1 = c["box"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=BOX_COLOR, lw=1.6))
            for s in ax.spines.values():
                s.set_edgecolor(BOX_COLOR)
                s.set_linewidth(2.5)
            ax.set_title(f"slice {z}\nPROMPTED", fontsize=9, color="#9A7B00", weight="bold")
        else:
            ax.set_title(f"slice {z}\npropagated", fontsize=9, color="#555")

    fig.suptitle(f"One prompt, whole volume — {c['key']}", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.015,
             "The box is given on one slice only. Every other mask comes from MedSAM2's memory "
             "attention carrying it through the volume — which is why volume is measured, not extrapolated.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.065, 1, 0.9])
    out_b = figdir / "propagation.png"
    fig.savefig(out_b, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_b}")

    # ------------------------------------------------------------ recap
    print("\n" + "=" * 62)
    print("WHAT THESE FIGURES SHOW")
    print("=" * 62)
    print("  model_output.png  green = radiologists, red = MedSAM2, yellow = the")
    print("                    prompt box. Column 3 is the model's own output.")
    print("  propagation.png   a single prompted slice; the rest is propagation.")
    print(f"\n  cached masks -> {maskdir}  (re-run the figures without the GPU)")


if __name__ == "__main__":
    main()
