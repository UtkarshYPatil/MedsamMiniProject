"""
STEP 0 -- run this before anything else.

Verifies that both halves of LIDC-IDRI are present and usable:
  1. DICOM images   (the .tcia manifest download)
  2. XML annotations (a SEPARATE download from TCIA -- "Radiologist
     Annotations/Segmentations", ~60 MB)

Without (2) there is no ground truth and nothing downstream can run.

Usage:
    python scripts/00_verify_data.py --dicom "D:/LIDC-IDRI" --xml "D:/LIDC-XML-only"
"""
import argparse
import sys
from collections import Counter
from pathlib import Path


def scan_dicom(root: Path):
    print(f"\n[1] DICOM images  : {root}")
    if not root.exists():
        print("    MISSING -- path does not exist")
        return None

    patients = sorted(p for p in root.glob("LIDC-IDRI-*") if p.is_dir())
    if not patients:
        patients = sorted(p for p in root.iterdir() if p.is_dir())

    n_dcm = 0
    series_counts = Counter()
    for p in patients:
        files = list(p.rglob("*.dcm"))
        n_dcm += len(files)
        series_counts[p.name] = len(files)

    print(f"    patient folders : {len(patients)}")
    print(f"    .dcm files      : {n_dcm:,}")
    if patients:
        small = [k for k, v in series_counts.items() if v < 50]
        print(f"    first patient   : {patients[0].name} ({series_counts[patients[0].name]} slices)")
        if small:
            print(f"    WARNING: {len(small)} patients have <50 slices (partial download?)")
    return {"patients": len(patients), "dcm": n_dcm}


def scan_xml(root: Path):
    print(f"\n[2] XML annotations: {root}")
    if not root.exists():
        print("    MISSING -- path does not exist")
        print("    Download 'Radiologist Annotations/Segmentations (XML)' from the")
        print("    LIDC-IDRI page on TCIA. It is a separate ~60 MB zip, NOT in the .tcia manifest.")
        return None

    xmls = list(root.rglob("*.xml"))
    print(f"    .xml files      : {len(xmls):,}")
    if xmls:
        print(f"    example         : {xmls[0].relative_to(root)}")
    return {"xml": len(xmls)}


def check_pylidc():
    print("\n[3] pylidc config")
    cfg = Path.home() / ".pylidcrc"
    if not cfg.exists():
        print(f"    MISSING -- create {cfg} with:")
        print("        [dicom]")
        print("        path = /full/path/to/LIDC-IDRI")
        print("        warn = True")
        return False
    print(f"    found {cfg}")
    try:
        import pylidc as pl
        n = pl.query(pl.Scan).count()
        print(f"    scans visible to pylidc: {n}")
        if n == 0:
            print("    WARNING: 0 scans -- the path in .pylidcrc is probably wrong")
            return False
        scan = pl.query(pl.Scan).first()
        nods = scan.cluster_annotations()
        print(f"    sample scan {scan.patient_id}: {len(nods)} nodule cluster(s)")
        return True
    except ImportError:
        print("    pylidc not installed -- pip install pylidc")
        return False
    except Exception as e:
        print(f"    pylidc error: {e}")
        return False


def check_torch():
    print("\n[4] Torch / GPU")
    try:
        import torch
        print(f"    torch   : {torch.__version__}")
        print(f"    cuda    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    device  : {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"    vram    : {free/1e9:.1f} GB free / {total/1e9:.1f} GB")
        else:
            print("    WARNING: no CUDA. MedSAM2 will run on CPU -- very slow.")
    except ImportError:
        print("    torch not installed")


def check_checkpoint():
    print("\n[5] MedSAM2 checkpoint")
    for name in ("MedSAM2_CTLesion.pt", "MedSAM2_latest.pt"):
        p = Path("checkpoints") / name
        if p.exists():
            print(f"    {name}: {p.stat().st_size/1e6:.0f} MB")
        else:
            print(f"    {name}: missing -- run scripts/download_checkpoint.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", required=True, help="folder containing LIDC-IDRI-XXXX patient dirs")
    ap.add_argument("--xml", required=True, help="folder containing the radiologist XML annotations")
    a = ap.parse_args()

    print("=" * 62)
    print("LIDC-IDRI dataset verification")
    print("=" * 62)

    d = scan_dicom(Path(a.dicom))
    x = scan_xml(Path(a.xml))
    ok = check_pylidc()
    check_torch()
    check_checkpoint()

    print("\n" + "=" * 62)
    blockers = []
    if not d or d["dcm"] == 0:
        blockers.append("no DICOM images found")
    if not x or x["xml"] == 0:
        blockers.append("no XML annotations found")
    if not ok:
        blockers.append("pylidc cannot read the dataset")

    if blockers:
        print("BLOCKED:")
        for b in blockers:
            print(f"  - {b}")
        sys.exit(1)
    print("READY -- run scripts/02_prepare_data.py next")


if __name__ == "__main__":
    main()
