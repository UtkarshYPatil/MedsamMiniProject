"""
Keep only CT series from a LIDC-IDRI download (drops DX/CR X-rays).

Each series folder is classified by the Modality tag of its first DICOM file.
Non-CT series are MOVED (not deleted) to --quarantine so nothing is lost.
Dry-run by default; pass --apply to actually move.

Usage:
    python scripts/filter_ct.py --root D:/datasets/LIDC-IDRI
    python scripts/filter_ct.py --root D:/datasets/LIDC-IDRI --apply
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import pydicom


def series_modality(folder: Path):
    """Modality of the first readable .dcm in folder, or None if no DICOM."""
    for f in sorted(folder.glob("*.dcm")):
        try:
            return pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["Modality"]).Modality
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LIDC-IDRI DICOM root")
    ap.add_argument("--quarantine", default=None, help="where non-CT series go (default: <root>_non_ct)")
    ap.add_argument("--apply", action="store_true", help="move non-CT series (default: dry run)")
    args = ap.parse_args()

    root = Path(args.root)
    quarantine = Path(args.quarantine) if args.quarantine else root.with_name(root.name + "_non_ct")

    # series folders = directories that directly contain .dcm files
    series_dirs = sorted({p.parent for p in root.rglob("*.dcm")})
    counts, to_move = Counter(), []
    for d in series_dirs:
        mod = series_modality(d)
        counts[mod] += 1
        if mod != "CT":
            to_move.append((d, mod))

    print(f"Series found: {len(series_dirs)}  by modality: {dict(counts)}")
    for d, mod in to_move:
        print(f"  non-CT [{mod}] {d}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to move non-CT series to", quarantine)
        return

    for d, _ in to_move:
        dest = quarantine / d.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
    print(f"\nMoved {len(to_move)} series to {quarantine}")


if __name__ == "__main__":
    main()
