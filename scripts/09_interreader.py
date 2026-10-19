"""
How much do the four radiologists disagree with each other?

    python3.12 scripts/09_interreader.py
    python3.12 scripts/09_interreader.py --limit 50

WHY THIS EXISTS
---------------
We report Dice 0.804 against a 50% consensus of up to four readers. That number
has a ceiling below 1.0, because the readers do not agree with each other, and
until the ceiling is measured there is no way to know whether 0.804 is close to
it or far from it. Optimising without knowing is chasing an unknown target.

Two numbers come out, on exactly the nodules in the manifest so the comparison
is like for like:

  reader vs reader      how much two humans outlining the same nodule differ
  reader vs consensus   the ceiling for anything scored the way we score

The second is the one to compare 0.804 against. A single reader measured the
way we measure the model cannot beat it, so it is the practical upper bound.

AXIS ORDER
----------
pylidc works in (Y, X, Z). Nothing here converts to (Z, Y, X) because every
mask is compared against another mask in the same frame -- Dice is
order-independent. Do not copy these indices into pipeline code without
converting; getting that wrong cost us a silent centroid bug once already.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import compat  # noqa: F401  -- MUST precede pylidc

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    tot = a.sum() + b.sum()
    return float(2 * inter / tot) if tot else float("nan")


def aligned_masks(anns):
    """
    Put every reader's mask for one nodule into a shared frame.

    Each annotation has its own bounding box, so they are placed into the union
    box by offset rather than assumed to be the same shape.
    """
    masks = [a.boolean_mask() for a in anns]
    boxes = [a.bbox() for a in anns]          # (Y, X, Z) slices, pylidc order

    lo = [min(b[d].start for b in boxes) for d in range(3)]
    hi = [max(b[d].stop for b in boxes) for d in range(3)]
    shape = tuple(hi[d] - lo[d] for d in range(3))

    out = []
    for m, b in zip(masks, boxes):
        buf = np.zeros(shape, dtype=bool)
        sl = tuple(slice(b[d].start - lo[d], b[d].start - lo[d] + m.shape[d])
                   for d in range(3))
        buf[sl] = m
        out.append(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of scans (default: every scan in the manifest)")
    a = ap.parse_args()

    import pylidc as pl
    from pylidc.utils import consensus

    man_path = Path(CFG.paths.processed) / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest at {man_path} -- run scripts/02_prepare_data.py first")
    manifest = json.loads(man_path.read_text())

    wanted = {}
    for m in manifest:
        wanted.setdefault(m["patient_id"], set()).add(m["key"])
    patients = sorted(wanted)
    if a.limit:
        patients = patients[: a.limit]
    print(f"{len(patients)} scans, {sum(len(v) for v in wanted.values())} nodules in the manifest\n")

    rows = []
    for pid in tqdm(patients, desc="scans"):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            continue
        try:
            clusters = scan.cluster_annotations()
        except Exception:
            continue

        for i, anns in enumerate(clusters):
            key = f"{pid}_n{i}"
            if key not in wanted[pid]:
                continue                      # not one we evaluated
            if len(anns) < 2:
                rows.append({"key": key, "n_readers": len(anns),
                             "reader_vs_reader": float("nan"),
                             "reader_vs_consensus": float("nan")})
                continue

            try:
                masks = aligned_masks(anns)
                cons, cbbox, _ = consensus(anns, clevel=CFG.annotation.consensus_level,
                                           pad=None)
            except Exception:
                continue

            pairwise = [dice(x, y) for x, y in combinations(masks, 2)]

            # Each reader against the consensus, in the consensus frame.
            vs_cons = []
            for ann in anns:
                m = ann.boolean_mask()
                b = ann.bbox()
                buf = np.zeros(cons.shape, dtype=bool)
                try:
                    sl = tuple(slice(b[d].start - cbbox[d].start,
                                     b[d].start - cbbox[d].start + m.shape[d])
                               for d in range(3))
                    buf[sl] = m
                except (ValueError, IndexError):
                    continue                  # readers outside the consensus box
                vs_cons.append(dice(buf, cons))

            rows.append({
                "key": key,
                "n_readers": len(anns),
                "reader_vs_reader": float(np.mean(pairwise)) if pairwise else float("nan"),
                "reader_vs_consensus": float(np.mean(vs_cons)) if vs_cons else float("nan"),
            })

    if not rows:
        raise SystemExit("no nodules measured -- is the manifest from this dataset?")

    df = pd.DataFrame(rows)
    out = Path(CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "interreader.csv", index=False)

    rr = df["reader_vs_reader"].dropna()
    rc = df["reader_vs_consensus"].dropna()

    print("\n" + "=" * 62)
    print(f"nodules measured        : {len(df)}  "
          f"({int((df['n_readers'] < 2).sum())} had a single reader and are excluded)")
    print(f"readers per nodule      : {dict(df['n_readers'].value_counts().sort_index())}")
    print()
    print(f"reader vs reader   Dice : mean {rr.mean():.3f}  median {rr.median():.3f}  "
          f"std {rr.std():.3f}   n={len(rr)}")
    print(f"reader vs consensus Dice: mean {rc.mean():.3f}  median {rc.median():.3f}  "
          f"std {rc.std():.3f}   n={len(rc)}")
    print("=" * 62)

    # The comparison this script exists to make.
    res = out / "results.csv"
    if res.exists():
        model = pd.read_csv(res)["dice"].dropna()
        ceiling = rc.mean()
        print(f"\n  our model vs consensus  : {model.mean():.3f}  (n={len(model)})")
        print(f"  a human  vs consensus   : {ceiling:.3f}")
        gap = ceiling - model.mean()
        print(f"  gap to the human ceiling: {gap:+.3f}")
        if gap <= 0:
            print("\n  The model scores at or above a single reader against the same\n"
                  "  consensus. Further Dice optimisation is chasing reader noise;\n"
                  "  spend the effort on detection or on the hard subtypes instead.")
        elif gap < 0.05:
            print("\n  Within 0.05 of what a human reader achieves against the same\n"
                  "  consensus. There is little headroom left in this metric.")
        else:
            print(f"\n  {gap:.3f} of headroom before reaching single-reader agreement.\n"
                  "  Dice improvements are still meaningful.")
    print(f"\nwrote {out / 'interreader.csv'}")


if __name__ == "__main__":
    main()
