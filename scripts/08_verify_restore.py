"""
Did the restore reproduce the Phase 2 dataset exactly?

    python3.12 scripts/08_verify_restore.py

Run this after 02_prepare_data.py and before trusting 03_run_baseline.py.

WHY
---
outputs/results.csv is committed and lists the exact 30 nodules the Phase 2
numbers were computed on. processed/ is not committed -- it is rebuilt from the
DICOM files. If the rebuild yields a different set of nodules (a scan that
failed to download, a different --limit, a partially downloaded series), then
03_run_baseline.py will report a Dice that is not comparable to the 0.789 in
the report, and nothing will say so.

This script compares the two key sets and tells you which case you are in.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG


def main():
    proc = Path(CFG.paths.processed)
    man_path = proc / "manifest.json"
    res_path = Path(CFG.paths.outputs) / "results.csv"

    if not man_path.exists():
        raise SystemExit(f"no manifest at {man_path}\n"
                         f"run:  python3.12 scripts/02_prepare_data.py --limit 50")
    if not res_path.exists():
        raise SystemExit(f"no {res_path} - nothing to compare against")

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    new = {m["key"] for m in manifest}

    rows = list(csv.DictReader(res_path.open(encoding="utf-8", newline="")))
    old = {r["key"] for r in rows}
    old_dice = {r["key"]: float(r["dice"]) for r in rows if r.get("dice")}

    print(f"rebuilt manifest : {len(new)} nodules, "
          f"{len({m['patient_id'] for m in manifest})} patients")
    print(f"committed results: {len(old)} nodules, "
          f"{len({r['patient_id'] for r in rows})} patients")

    missing = sorted(old - new)     # in the report but not rebuilt
    extra   = sorted(new - old)     # rebuilt but not in the report
    shared  = new & old

    print(f"\nmatching nodules : {len(shared)} / {len(old)}")

    if not missing and not extra:
        print("\nEXACT MATCH.")
        print("03_run_baseline.py should reproduce Dice 0.789 on these 30 nodules.")
        print("If it does not, the difference is in the model or config, not the data.")
        ok = True
    else:
        ok = False
        if missing:
            print(f"\nMISSING ({len(missing)}) - in the report, not rebuilt:")
            for k in missing:
                print(f"    {k:<24} was dice {old_dice.get(k, float('nan')):.3f}")
            pats = sorted({k.rsplit('_n', 1)[0] for k in missing})
            print(f"  affected patients: {', '.join(pats)}")
            print("  likely cause: those scans are absent or incompletely downloaded.")
            print("  try a larger --limit, or re-run the dataset download to repair them.")
        if extra:
            print(f"\nEXTRA ({len(extra)}) - rebuilt but not in the report:")
            for k in extra:
                print(f"    {k}")
            print("  a larger --limit than the original run would do this.")

    # texture mix matters for the density discussion in the report
    if manifest:
        mix = Counter(m["texture_class"] for m in manifest)
        print(f"\ntexture mix: {dict(mix)}")
        old_mix = Counter(r["texture_class_lidc"] for r in rows)
        print(f"   report had: {dict(old_mix)}")

    print("\n" + "=" * 60)
    if ok:
        print("VERDICT: dataset reproduced. Safe to compare new results to the report.")
    else:
        print("VERDICT: dataset differs. Any Dice you get is NOT comparable to 0.789")
        print("         unless you say which nodules it was measured on.")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
