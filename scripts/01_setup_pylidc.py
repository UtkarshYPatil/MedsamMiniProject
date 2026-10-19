"""
Point pylidc at your DICOM folder. Run once.

    python scripts/01_setup_pylidc.py --find              search drives for the dataset
    python scripts/01_setup_pylidc.py --dicom "E:/LIDC-IDRI"

WHY THIS SCRIPT EXISTS
----------------------
pylidc's config file has a different name on every platform:
    Linux / macOS : ~/.pylidcrc
    Windows       : %USERPROFILE%\\pylidc.conf
Writing the wrong one fails silently -- pylidc still lists scans (its annotation
database ships with the package) but cannot load a single DICOM image.
This script asks pylidc itself where the file belongs.
"""
import argparse
import os
import platform
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import compat  # noqa: F401  -- MUST precede any pylidc import


def config_path() -> Path:
    """Ask pylidc where its config lives; fall back to its own platform rule."""
    try:
        from pylidc._utils import _get_config_file  # pylidc >= 0.2.2
        return Path(_get_config_file())
    except Exception:
        pass
    try:
        from pylidc.Scan import _get_config_file
        return Path(_get_config_file())
    except Exception:
        pass
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    return home / ("pylidc.conf" if platform.system() == "Windows" else ".pylidcrc")


def looks_like_lidc(folder: Path) -> int:
    """Number of LIDC-IDRI-XXXX subfolders directly inside `folder`."""
    try:
        return sum(1 for p in folder.iterdir()
                   if p.is_dir() and p.name.upper().startswith("LIDC-IDRI-"))
    except (PermissionError, OSError):
        return 0


def find_dataset(max_depth=4):
    """Walk every fixed drive looking for a folder full of LIDC-IDRI-XXXX dirs."""
    roots = []
    if platform.system() == "Windows":
        for letter in string.ascii_uppercase:
            d = Path(f"{letter}:/")
            if d.exists():
                roots.append(d)
    else:
        roots = [Path.home(), Path("/mnt"), Path("/media"), Path("/data")]

    skip = {"windows", "program files", "program files (x86)", "programdata",
            "$recycle.bin", "system volume information", "appdata", "node_modules",
            "$winreagent", "recovery", "perflogs", ".git", "steamlibrary",
            "xboxgames", "riot games"}

    hits = []
    print("searching drives (this takes a minute)...")
    for root in roots:
        print(f"  {root}")
        stack = [(root, 0)]
        while stack:
            cur, depth = stack.pop()
            if depth > max_depth:
                continue
            n = looks_like_lidc(cur)
            if n:
                hits.append((cur, n))
                print(f"    FOUND  {cur}  ({n} patient folders)")
                continue                      # no need to descend further
            if depth == max_depth:
                continue
            try:
                for child in cur.iterdir():
                    if child.is_dir() and child.name.lower() not in skip:
                        stack.append((child, depth + 1))
            except (PermissionError, OSError):
                continue
    return hits


def find_xml(near: Path):
    """Look for the separate radiologist-annotation download near the images."""
    cands = []
    for base in (near.parent, near.parent.parent, near):
        if not base.exists():
            continue
        try:
            for p in base.iterdir():
                if p.is_dir() and ("xml" in p.name.lower() or "annot" in p.name.lower()):
                    n = sum(1 for _ in p.rglob("*.xml"))
                    if n:
                        cands.append((p, n))
        except (PermissionError, OSError):
            continue
    return cands


def write_config(dicom_root: Path):
    cfg = config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"[dicom]\npath = {dicom_root.as_posix()}\nwarn = True\n",
                   encoding="utf-8")
    print(f"\nwrote {cfg}")
    print("-" * 56)
    print(cfg.read_text(encoding="utf-8").rstrip())
    print("-" * 56)
    return cfg


def verify():
    try:
        import pylidc as pl
    except ImportError:
        print("pylidc not installed")
        return False

    n = pl.query(pl.Scan).count()
    print(f"\npylidc annotation DB: {n} scans")
    if n == 0:
        print("  0 scans -- pylidc installation problem, not a path problem")
        return False

    scan = pl.query(pl.Scan).first()
    try:
        path = scan.get_path_to_dicom_files()
        print(f"  DICOM path resolves: {path}")
    except RuntimeError as e:
        print(f"  CANNOT resolve DICOM path: {e}")
        print("  -> pylidc read the config but found no matching patient folder.")
        print("     The path must be the folder CONTAINING LIDC-IDRI-0001 etc.")
        return False
    except Exception as e:
        print(f"  CANNOT resolve DICOM path: {type(e).__name__}: {e}")
        print("  -> this is an environment problem, not a path problem.")
        return False

    try:
        vol = scan.to_volume(verbose=False)
        print(f"  loaded {scan.patient_id}: volume {vol.shape}, "
              f"HU range {vol.min():.0f}..{vol.max():.0f}")
        print(f"  nodule clusters in this scan: {len(scan.cluster_annotations())}")
        print("\nREADY -- run: python scripts/02_prepare_data.py --limit 10")
        return True
    except Exception as e:
        print(f"  volume load FAILED: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", help="folder containing the LIDC-IDRI-XXXX directories")
    ap.add_argument("--find", action="store_true", help="search all drives for it")
    a = ap.parse_args()

    print(f"platform    : {platform.system()}")
    print(f"config file : {config_path()}")

    root = None

    if a.find:
        hits = find_dataset()
        if not hits:
            print("\nNo LIDC-IDRI-XXXX folders found on any drive.")
            print("The image download has not completed. The .tcia manifest is")
            print("only a shopping list -- the TCIA Data Retriever must finish")
            print("downloading, and you choose its destination folder separately.")
            sys.exit(1)
        hits.sort(key=lambda h: -h[1])
        print(f"\n{len(hits)} candidate(s):")
        for p, n in hits:
            print(f"   {n:>5} patients   {p}")
        root = hits[0][0]
        print(f"\nusing: {root}")
    elif a.dicom:
        root = Path(a.dicom)
    else:
        ap.error("pass --dicom PATH or --find")

    root = root.resolve()
    if not root.exists():
        sys.exit(f"path does not exist: {root}")

    n = looks_like_lidc(root)
    print(f"\nLIDC-IDRI-XXXX folders directly inside: {n}")
    if n == 0:
        print("WARNING: none found. pylidc needs the folder that CONTAINS")
        print("         LIDC-IDRI-0001, LIDC-IDRI-0002, ... not one level above or below.")

    write_config(root)

    xmls = find_xml(root)
    if xmls:
        print("\npossible XML annotation folders:")
        for p, n in xmls:
            print(f"   {n:>6} xml files   {p}")
    else:
        print("\nNo XML annotation folder found near the images.")
        print("pylidc ships its own annotation DB, so this is NOT required --")
        print("but note it in your report: contours come from the packaged DB.")

    ok = verify()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
