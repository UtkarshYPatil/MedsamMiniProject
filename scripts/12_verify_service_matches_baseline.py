"""
Does uploading an .npz through the running stack reproduce the baseline?

    python scripts/12_verify_service_matches_baseline.py
    python scripts/12_verify_service_matches_baseline.py --limit 40
    python scripts/12_verify_service_matches_baseline.py --api http://localhost:8000

WHAT THIS ANSWERS
-----------------
outputs/results.csv was produced by scripts/03_run_baseline.py, calling the
model in-process. The UI calls it through FastAPI -> Redis -> a worker in a
different container, with its own torch build and its own device.

Those are two code paths to the same model, and nothing has ever checked that
they agree. If they do not, every accuracy figure in the report describes the
research script rather than the thing a radiologist actually uses.

So: for each nodule, this replays THE SAME PROMPT the baseline used -- the box
derived from the consensus mask, recomputed identically -- through the live
API, and compares what comes back against the stored pred_* row.

The prompt is recomputed rather than read from the CSV because the CSV never
stored it; best_slice() and bbox_from_mask() are deterministic, so recomputing
reproduces it exactly.

READING THE RESULT
------------------
n_voxels is the sharp one. It is an integer count, so an exact match means the
two paths produced the identical mask, and any difference at all means they
did not. The millimetre figures are derived from it, so they follow.

Differences that are REAL and worth knowing about:

  - device. results.csv was measured on CPU. On CUDA, MedSAM2 runs
    fill_holes_in_mask_scores; on CPU it is skipped. Different masks, both
    legitimate.
  - the Python 3.11/3.12 mismatch in the CUDA worker image, which disables
    that same post-processing with a warning in the worker log. If the GPU
    worker matches the CPU baseline exactly, that is evidence the
    post-processing is being skipped in both.

So a mismatch is not automatically a bug. It is a question with an answer, and
this script exists so the answer is measured instead of assumed.

Needs the stack running and the dataset present.
"""
import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG
from src.prompts.bbox import bbox_from_mask, best_slice

# Compared field by field. Tolerance is in the field's own unit; the two
# millimetre figures are rounded to 1 dp by the reporter, so anything under
# 0.05 is representational rather than real.
FIELDS = [
    ("n_voxels", 0),                 # integer: exact or not
    ("diameter_mean_mm", 0.05),
    ("diameter_max_mm", 0.05),
    ("volume_mm3", 0.05),
    ("hu_mean", 0.05),
    ("core_hu_p50", 0.05),
]


def npz_bytes(image_hu, spacing):
    """The upload the UI would send: image and spacing, no mask."""
    buf = io.BytesIO()
    np.savez_compressed(buf, image_hu=image_hu.astype(np.int16),
                        spacing=np.asarray(spacing, dtype=np.float32))
    return buf.getvalue()


def run_one(api, image_hu, spacing, z, box, timeout=300):
    """Upload, segment with the given prompt, wait. Returns the result dict."""
    r = requests.post(f"{api}/api/cases",
                      files={"file": ("roi.npz", npz_bytes(image_hu, spacing),
                                      "application/octet-stream")},
                      timeout=120)
    r.raise_for_status()
    case_id = r.json()["case_id"]

    r = requests.post(f"{api}/api/cases/{case_id}/segment",
                      json={"slice": int(z), "box": [float(v) for v in box]},
                      timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        j = requests.get(f"{api}/api/jobs/{job_id}", timeout=30).json()
        if j["status"] == "done":
            return j.get("result") or {}
        if j["status"] == "failed":
            raise RuntimeError(f"worker failed: {(j.get('error') or '')[-400:]}")
        time.sleep(1.0)
    raise TimeoutError(f"job {job_id} still {j['status']} after {timeout}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=20,
                    help="nodules to check (default 20; each is one inference)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    api = a.api.rstrip("/")
    try:
        health = requests.get(f"{api}/healthz", timeout=15).json()
    except Exception as e:
        raise SystemExit(f"the API is not reachable at {api} -- is the stack up?\n{e}")

    worker = health.get("worker") or {}
    print(f"API   : {api}")
    print(f"worker: {worker.get('device_detail', 'not reported')}")
    if worker.get("dry_run") or health.get("config", {}).get("dry_run"):
        raise SystemExit("DRY_RUN is on -- the dummy segmenter would be compared "
                         "against real baseline numbers. Set DRY_RUN=0.")
    if health.get("workers_alive") == 0:
        raise SystemExit("no worker is connected; jobs would queue forever")

    proc = Path(CFG.paths.processed)
    manifest = json.loads((proc / "manifest.json").read_text(encoding="utf-8"))
    res_path = Path(CFG.paths.outputs) / "results.csv"
    if not res_path.exists():
        raise SystemExit(f"no baseline at {res_path} -- run scripts/03_run_baseline.py")
    base = pd.read_csv(res_path).set_index("key")
    print(f"baseline: {len(base)} nodules in results.csv\n")

    items = [m for m in manifest if m["key"] in base.index][: a.limit]
    if not items:
        raise SystemExit("no manifest entries match results.csv")

    rows, failures = [], []
    for item in tqdm(items, desc="nodules"):
        key = item["key"]
        d = np.load(proc / item["file"])
        image_hu, gt, spacing = d["image_hu"], d["mask"], d["spacing"]
        if not gt.any():
            continue

        # Exactly what 03_run_baseline.py prompts with.
        z = best_slice(gt)
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            continue

        try:
            res = run_one(api, image_hu, spacing, z, box)
        except Exception as e:
            failures.append((key, str(e)[:200]))
            continue

        if res.get("empty"):
            failures.append((key, "service returned an empty mask"))
            continue

        meas = res.get("measurement", {})
        row = {"key": key, "device": res.get("device")}
        for field, tol in FIELDS:
            want = base.at[key, f"pred_{field}"]
            got = meas.get(field)
            row[f"{field}_baseline"] = want
            row[f"{field}_service"] = got
            try:
                row[f"{field}_delta"] = float(got) - float(want)
            except (TypeError, ValueError):
                row[f"{field}_delta"] = np.nan
        row["conf_baseline"] = base.at[key, "pred_confidence"]
        row["conf_service"] = meas.get("confidence")
        rows.append(row)

    if not rows:
        raise SystemExit("nothing was compared" +
                         (f" -- {len(failures)} failures" if failures else ""))

    df = pd.DataFrame(rows)
    out = Path(a.out or CFG.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "service_vs_baseline.csv", index=False)

    # ------------------------------------------------------------- report
    n = len(df)
    print("\n" + "=" * 68)
    print(f"compared {n} nodules   service device: {df['device'].iloc[0]}")
    print(f"baseline device: CPU (as recorded in outputs/results.csv)\n")
    print(f"  {'field':<20} {'exact':>7} {'within tol':>11} {'max |delta|':>12}")
    print("  " + "-" * 52)

    all_exact = True
    for field, tol in FIELDS:
        delta = df[f"{field}_delta"].dropna()
        if not len(delta):
            continue
        exact = int((delta == 0).sum())
        within = int((delta.abs() <= tol).sum())
        if exact != n:
            all_exact = False
        print(f"  {field:<20} {exact:>4}/{n:<3} {within:>8}/{n:<3} "
              f"{delta.abs().max():>12.4f}")

    agree_conf = int((df["conf_baseline"] == df["conf_service"]).sum())
    print(f"\n  confidence flag agrees on {agree_conf}/{n}")

    print("\n" + "=" * 68)
    if all_exact:
        print("IDENTICAL. The service reproduces the baseline exactly, so every")
        print("figure in the report describes the deployed tool as well as the")
        print("research script.")
    else:
        nv = df["n_voxels_delta"].dropna()
        worse = int((nv > 0).sum())
        fewer = int((nv < 0).sum())
        print("THE TWO PATHS DIFFER.")
        print(f"  masks larger in the service: {worse}   smaller: {fewer}")
        print(f"  median voxel difference: {nv.median():+.0f}")
        print()
        print("  Most likely cause, in order:")
        print("   1. device. results.csv was measured on CPU, where MedSAM2")
        print("      skips fill_holes_in_mask_scores. On CUDA it runs, and the")
        print("      masks legitimately differ.")
        print("   2. the Python 3.11/3.12 mismatch in the CUDA worker image,")
        print("      which disables that same step with a warning in the worker")
        print("      log. If the GPU service matches the CPU baseline EXACTLY,")
        print("      that is evidence hole-filling is skipped in both and the")
        print("      GPU image is not doing what it appears to.")
        print()
        print("  Check the worker log for 'Skipping the post-processing step'.")

    if failures:
        print(f"\n{len(failures)} nodule(s) failed outright:")
        for k, e in failures[:10]:
            print(f"  {k}: {e}")

    print(f"\nwrote {out / 'service_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
