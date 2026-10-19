"""
Build the processed dataset: consensus masks + ROI crops + manifest.

    python scripts/02_prepare_data.py --limit 10          smoke test
    python scripts/02_prepare_data.py --limit 300         full run
    python scripts/02_prepare_data.py --start 100 --limit 100   resume a chunk

Appends to an existing manifest, so chunked runs accumulate. Pass --fresh to
start over.

Do not paste shell comments after the command on Windows CMD -- argparse will
reject them.
"""
import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import compat  # noqa: F401  -- MUST precede pylidc

import numpy as np
from tqdm import tqdm

from src.config import CFG
from src.data.parse_lidc import nodules_in_scan, to_dict
from src.data.preprocess import crop_roi


def qualifying_annotations(scan, min_diameter_mm):
    """
    Cheap pre-check: does this scan have any nodule big enough to keep?

    Annotation clustering reads the bundled database; loading the volume reads
    ~200 DICOM files off disk. Checking first skips that cost entirely for
    scans with nothing to process -- roughly a third of LIDC.
    """
    try:
        clusters = scan.cluster_annotations()
    except Exception:
        return None                      # let the caller surface the error
    return [c for c in clusters
            if float(np.mean([a.diameter for a in c])) >= min_diameter_mm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=CFG.eval.n_scans)
    ap.add_argument("--start", type=int, default=0, help="skip this many scans first")
    ap.add_argument("--fresh", action="store_true", help="discard the existing manifest")
    ap.add_argument("--verbose-errors", action="store_true")
    a = ap.parse_args()

    import pylidc as pl

    out = Path(CFG.paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"

    manifest = []
    if man_path.exists() and not a.fresh:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    seen = {m["key"] for m in manifest}

    scans = pl.query(pl.Scan).all()[a.start:a.start + a.limit]
    print(f"{len(scans)} scans (offset {a.start}); manifest already holds {len(manifest)}")
    if not scans:
        raise SystemExit("no scans -- check ~/.pylidcrc or pylidc.conf")

    skipped_scans = skipped_nodules = no_nodules = 0
    first_error = None
    t0 = time.time()

    for scan in tqdm(scans, desc="scans"):
        qualifying = qualifying_annotations(scan, CFG.annotation.min_diameter_mm)
        if qualifying is None:
            skipped_scans += 1
            continue
        if not qualifying:
            no_nodules += 1
            continue                      # volume never loaded -- the whole point

        try:
            vol_hu = scan.to_volume(verbose=False)         # (Y, X, Z)
            vol_hu = np.transpose(vol_hu, (2, 0, 1))       # -> (Z, Y, X)
        except Exception as e:
            skipped_scans += 1
            if first_error is None:
                first_error = traceback.format_exc()
            tqdm.write(f"skip {scan.patient_id}: {type(e).__name__}: {e}")
            if a.verbose_errors:
                tqdm.write(traceback.format_exc())
            continue

        for rec, mask_zyx, bbox_zyx in nodules_in_scan(
                scan,
                clevel=CFG.annotation.consensus_level,
                min_diameter_mm=CFG.annotation.min_diameter_mm):

            if rec.key in seen:
                continue

            full = np.zeros(vol_hu.shape, dtype=bool)
            sz, sy, sx = bbox_zyx
            try:
                full[sz, sy, sx] = mask_zyx
            except ValueError as e:
                tqdm.write(f"  bbox mismatch {rec.key}: {e}")
                skipped_nodules += 1
                continue

            img_roi, origin = crop_roi(
                vol_hu, rec.centroid_zyx,
                size=CFG.preprocess.roi_size, depth=CFG.preprocess.roi_depth)
            msk_roi, _ = crop_roi(
                full, rec.centroid_zyx,
                size=CFG.preprocess.roi_size, depth=CFG.preprocess.roi_depth)

            if not msk_roi.any():
                tqdm.write(f"  empty crop {rec.key} -- skipping")
                skipped_nodules += 1
                continue

            np.savez_compressed(
                out / f"{rec.key}.npz",
                image_hu=img_roi.astype(np.int16),
                mask=msk_roi,
                spacing=np.asarray(rec.spacing_zyx, dtype=np.float32),
                origin=np.asarray(origin, dtype=np.int32))

            d = to_dict(rec)
            d["file"] = f"{rec.key}.npz"
            manifest.append(d)
            seen.add(rec.key)

    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"\n{len(manifest)} nodules in manifest  ({dt/max(1,len(scans)):.1f}s/scan)")
    print(f"   scans with no qualifying nodule: {no_nodules}")
    print(f"   skipped scans: {skipped_scans}   skipped nodules: {skipped_nodules}")

    if manifest:
        counts = Counter(m["texture_class"] for m in manifest)
        print("   density (LIDC texture):", dict(counts))
        print("   readers per nodule:", dict(Counter(m["n_annotations"] for m in manifest)))
        sizes = sorted(m["diameter_mm"] for m in manifest)
        print(f"   diameter mm: min {sizes[0]:.1f}  median {sizes[len(sizes)//2]:.1f}  "
              f"max {sizes[-1]:.1f}")
        thin = [c for c in ("part_solid", "ground_glass") if counts.get(c, 0) < 15]
        if thin:
            need = ", ".join(thin)
            print(f"\n   NOTE: still under 15 nodules for: {need}")
            print(f"   Density calibration needs more -- rerun with "
                  f"--start {a.start + a.limit} --limit {a.limit}")
    else:
        print("\nNo nodules written. First error:\n")
        print(first_error or "(no exception -- all scans had 0 qualifying nodules)")


if __name__ == "__main__":
    main()
