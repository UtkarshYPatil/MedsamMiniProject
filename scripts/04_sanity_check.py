"""
Visual sanity check -- run this right after 02_prepare_data.py.

Renders a few processed nodules with the consensus outline drawn on top.
If the outline does not sit on an actual nodule, the axis order or the
centroid is wrong and nothing downstream is trustworthy.

    python scripts/04_sanity_check.py
    python scripts/04_sanity_check.py --n 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import best_slice
from src.report.sizing import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    proc = Path(CFG.paths.processed)
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("manifest is empty -- run scripts/02_prepare_data.py")

    # spread the sample across density classes so the hard cases show up
    by_class = {}
    for m in manifest:
        by_class.setdefault(m["texture_class"], []).append(m)
    picks, i = [], 0
    while len(picks) < min(a.n, len(manifest)):
        added = False
        for cls in sorted(by_class):
            if i < len(by_class[cls]):
                picks.append(by_class[cls][i])
                added = True
            if len(picks) >= a.n:
                break
        if not added:
            break
        i += 1

    rows = (len(picks) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    print(f"{'key':<28} {'LIDC':<13} {'HU-class':<13} {'diam mm':>8} {'conf':>7}")
    print("-" * 74)

    for ax, item in zip(axes, picks):
        d = np.load(proc / item["file"])
        vol_hu, mask, spacing = d["image_hu"].astype(np.float32), d["mask"], d["spacing"]

        z = best_slice(mask)
        img = apply_window(vol_hu[z], CFG.preprocess.window_level,
                           CFG.preprocess.window_width)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        if mask[z].any():
            ax.contour(mask[z], levels=[0.5], colors="#ff5a3c", linewidths=1.4)

        m = summarize(mask, vol_hu, spacing)
        ax.set_title(
            f"{item['key']}\nLIDC: {item['texture_class']} | HU: {m.density_class}\n"
            f"{m.diameter_mean_mm} mm | {item['n_annotations']} readers",
            fontsize=8)
        ax.axis("off")

        print(f"{item['key']:<28} {item['texture_class']:<13} "
              f"{m.density_class:<13} {m.diameter_mean_mm:>8.1f} {m.confidence:>7}")

    for ax in axes[len(picks):]:
        ax.axis("off")

    out = Path(CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sanity_check.png"
    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\nwrote {p}")
    print("Open it. Every red outline must sit on a visible nodule.")
    print("If any outline floats in empty lung or on the chest wall, stop and")
    print("report it -- the crop or the annotation mapping is wrong.")

if __name__ == "__main__":
    main()
