"""
Calibrate (or refute) the density classifier against LIDC reader texture ratings.

    python scripts/05_calibrate_density.py

Reads every processed nodule, computes HU features, and reports:
  - class balance (the usual blocker -- LIDC is dominated by solid nodules)
  - whether any single threshold separates the classes at all
  - the best threshold found, and its agreement
  - a scatter plot: outputs/density_calibration.png

It deliberately REFUSES to suggest thresholds when the sample is too small or
too imbalanced. A classifier fitted on 2 part-solid nodules is not a finding.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CFG
from src.report.density import hu_profile

MIN_PER_CLASS = 15          # below this, any fitted threshold is noise
CLASSES = ["solid", "part_solid", "ground_glass"]


def load_features():
    proc = Path(CFG.paths.processed)
    man = json.loads((proc / "manifest.json").read_text())
    rows = []
    for m in man:
        d = np.load(proc / m["file"])
        v, msk = d["image_hu"].astype(np.float32), d["mask"]
        if not msk.any():
            continue
        f = hu_profile(v, msk)
        f.update(key=m["key"], lidc=m["texture_class"],
                 texture_mean=m["texture_mean"], diameter_mm=m["diameter_mm"],
                 readers=m["n_annotations"])
        rows.append(f)
    return rows


def best_threshold(rows, feature, positive, negative):
    """Best single cut on `feature` separating two class groups."""
    pos = [r[feature] for r in rows if r["lidc"] in positive]
    neg = [r[feature] for r in rows if r["lidc"] in negative]
    if not pos or not neg:
        return None
    cands = sorted(set(pos + neg))
    best = (0.0, None)
    for t in cands:
        acc = (sum(p >= t for p in pos) + sum(n < t for n in neg)) / (len(pos) + len(neg))
        if acc > best[0]:
            best = (acc, t)
    return {"accuracy": best[0], "threshold": best[1],
            "n_pos": len(pos), "n_neg": len(neg),
            "pos_range": (min(pos), max(pos)), "neg_range": (min(neg), max(neg))}


def main():
    rows = load_features()
    if not rows:
        raise SystemExit("no processed nodules -- run scripts/02_prepare_data.py")

    counts = Counter(r["lidc"] for r in rows)
    print(f"n = {len(rows)} nodules")
    print("class balance:", dict(counts))

    thin = [c for c in CLASSES if counts.get(c, 0) < MIN_PER_CLASS]
    print()

    # -- separability: does solid's range swallow the others? --------------
    print(f"{'feature':<22} {'solid median':>13} {'solid range':>20} "
          f"{'subsolid median':>16}  verdict")
    print("-" * 86)
    features = ["full_solid_fraction", "core_solid_fraction", "hu_p50",
                "core_hu_p50", "hu_p90"]
    results = {}
    for f in features:
        s = [r[f] for r in rows if r["lidc"] == "solid"]
        sub = [r[f] for r in rows if r["lidc"] in ("part_solid", "ground_glass")]
        if not s or not sub:
            continue
        overlap = all(min(s) <= x <= max(s) for x in sub)
        bt = best_threshold(rows, f, {"solid"}, {"part_solid", "ground_glass"})
        results[f] = bt
        rng = f"[{min(s):.2f}, {max(s):.2f}]"
        if overlap:
            verdict = "total overlap"
        else:
            verdict = f"cut {bt['threshold']:.2f} -> {bt['accuracy']*100:.0f}%"
        print(f"{f:<22} {np.median(s):>13.2f} {rng:>20} {np.median(sub):>16.2f}  {verdict}")

    # -- verdict -----------------------------------------------------------
    print("\n" + "=" * 64)
    if thin:
        print("NOT CALIBRATED. Too few nodules in:", ", ".join(thin))
        print(f"Need >= {MIN_PER_CLASS} per class. LIDC is dominated by solid")
        print("nodules, so process more scans:")
        print("    python scripts/02_prepare_data.py --limit 300")
        print("\nUntil then, density stays PROVISIONAL and the tool must show")
        print("HU statistics rather than a class label.")
    else:
        best_f = max(results, key=lambda k: results[k]["accuracy"])
        b = results[best_f]
        print(f"best feature : {best_f}")
        print(f"threshold    : {b['threshold']:.3f}")
        print(f"agreement    : {b['accuracy']*100:.0f}%  "
              f"(n={b['n_pos']} solid vs {b['n_neg']} subsolid)")
        if b["accuracy"] < 0.80:
            print("\nBelow 80% -- a threshold classifier is not good enough here.")
            print("Report HU statistics instead, or train a small classifier on")
            print("[core_hu_p50, core_solid_fraction, diameter_mm].")

    # -- plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colours = {"solid": "#2a78d6", "part_solid": "#eb6834", "ground_glass": "#1baf7a"}
    for ax, (fx, fy) in zip(axes, [("core_hu_p50", "core_solid_fraction"),
                                   ("diameter_mm", "core_hu_p50")]):
        for c in CLASSES:
            g = [r for r in rows if r["lidc"] == c]
            if not g:
                continue
            ax.scatter([r[fx] for r in g], [r[fy] for r in g],
                       label=f"{c} (n={len(g)})", c=colours[c], s=38,
                       alpha=0.8, edgecolors="white", linewidths=0.8)
        ax.set_xlabel(fx); ax.set_ylabel(fy)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Density features vs LIDC reader texture rating", fontsize=11)

    out = Path(CFG.paths.outputs); out.mkdir(parents=True, exist_ok=True)
    p = out / "density_calibration.png"
    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\nwrote {p}")

    (out / "density_features.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"wrote {out/'density_features.json'}")


if __name__ == "__main__":
    main()
