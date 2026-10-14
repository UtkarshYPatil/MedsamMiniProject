"""
Download only the CT series of LIDC-IDRI from the IDC (via idc-index / s5cmd).

Resumable: re-running skips files already present (s5cmd sync), so a crash
or Ctrl+C costs nothing. Dry-run by default; pass --apply to download.

Usage:
    python scripts/download_ct.py --out E:/MiniProject/DATASET_ct
    python scripts/download_ct.py --out E:/MiniProject/DATASET_ct --apply
"""
import argparse
from pathlib import Path

from idc_index import IDCClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="download directory")
    ap.add_argument("--apply", action="store_true", help="actually download (default: dry run)")
    args = ap.parse_args()

    client = IDCClient()
    df = client.index
    ct = df[(df["collection_id"] == "lidc_idri") & (df["Modality"] == "CT")]

    print(f"LIDC-IDRI CT series: {len(ct)}  patients: {ct['PatientID'].nunique()}  "
          f"instances: {int(ct['instanceCount'].sum())}  size: {ct['series_size_MB'].sum() / 1024:.1f} GB")

    if not args.apply:
        print("\nDry run. Re-run with --apply to download to", args.out)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client.download_from_selection(
        downloadDir=str(out),
        seriesInstanceUID=ct["SeriesInstanceUID"].tolist(),
        use_s5cmd_sync=True,
        quiet=False,
    )
    print("Done ->", out)


if __name__ == "__main__":
    main()
