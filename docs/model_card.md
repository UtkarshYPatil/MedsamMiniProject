# Model Card — MedSAM2 for pulmonary nodule segmentation

Fill the marked fields from `outputs/results.csv` after running
`scripts/03_run_baseline.py`. Everything else is already final.

## Model

| Field | Value |
|---|---|
| Name | MedSAM2 |
| Checkpoint | `MedSAM2_CTLesion.pt` |
| Source | wanglab/MedSAM2 on Hugging Face |
| Base architecture | SAM 2.1 (Hiera encoder, prompt encoder, mask decoder, memory attention) |
| Paper | Ma et al., *MedSAM2: Segment Anything in 3D Medical Images and Videos*, arXiv:2504.03600 |
| Licence | CC-BY-SA-4.0 — research and education only |
| Modified by us? | **No. Used frozen, inference only. Zero parameters trained.** |
| Checkpoint SHA256 | `FILL` |
| Comparison run | `MedSAM2_latest.pt` — `FILL` |

## Intended use

Measuring pulmonary nodules in chest CT to assist a radiologist during lung
cancer screening.

**Out of scope:** diagnosis, malignancy prediction, triage, or any clinical
decision. The output is a measurement, not a finding.

## Input / output

| | |
|---|---|
| Input | CT volume ROI, Hounsfield Units, lung window L −600 / W 1500, 3-channel uint8 |
| ROI | 128 × 128 px in-plane, ±16 slices around the nodule centroid |
| Prompt | bounding box on the largest-area nodule slice (+5 px padding) |
| Propagation | bidirectional through the volume via SAM 2.1 memory attention |
| Output | boolean 3D mask |
| Derived | volume (mm³), mean/max in-plane diameter (mm), HU p10/p50/p90, core HU median |

## Evaluation

| | |
|---|---|
| Dataset | LIDC-IDRI |
| Reference | 50% consensus of up to 4 radiologist contours (pylidc) |
| Scans processed | `FILL` |
| Nodules evaluated | `FILL` |
| Split | by patient |
| Hardware | `FILL` |
| Seed | 42 |
| Commit | `FILL` |

### Segmentation quality

| Metric | Mean | Std | Median |
|---|---|---|---|
| Dice | `FILL` | `FILL` | `FILL` |
| IoU | `FILL` | `FILL` | `FILL` |
| HD95 (mm) | `FILL` | `FILL` | `FILL` |

### By reader texture class

Ground-glass and part-solid are expected to score lowest — faint boundaries.

| Texture | n | Dice mean | Dice std |
|---|---|---|---|
| Solid | `FILL` | `FILL` | `FILL` |
| Part-solid | `FILL` | `FILL` | `FILL` |
| Ground-glass | `FILL` | `FILL` | `FILL` |

### Measurement agreement

| | |
|---|---|
| Diameter error, mean | `FILL` mm |
| Diameter error, abs mean | `FILL` mm |
| Within 1 mm of reference | `FILL` % |

### Latency

| | |
|---|---|
| Seconds per nodule (mean) | `FILL` |

## Limitations

1. **Prompts are not automatic.** The bounding box is derived from the
   reference mask, so these figures measure *segmentation quality given a
   correct box*, not end-to-end detection. Automatic detection is Phase 3.
2. **Density classification is unvalidated** — 60% agreement with reader
   texture ratings on 30 nodules, with every error on a reader-rated solid
   nodule that attenuates like ground glass. Reported as provisional; the tool
   shows HU statistics instead. See `docs/hld.md` §7.
3. **Expected to perform worst on ground-glass and part-solid nodules.**
   Both are confidence-flagged by default.
4. **The reference is a construct** — up to 4 readers disagree; the 50%
   consensus is one choice among several. `n_annotations` is carried on every
   row.
5. **Single dataset, single reader population.** Generalisation untested.
6. **Weights are licensed for research and education only.**

## Reproducibility

```
python scripts/01_setup_pylidc.py --dicom "E:/MiniProject/DATASET_ct/lidc_idri"
python scripts/02_prepare_data.py --limit 300
python scripts/04_sanity_check.py
python scripts/03_run_baseline.py
python scripts/05_calibrate_density.py
```

Environment shims for pylidc on Python 3.13 / NumPy 2.x live in
`src/compat.py` and are documented there.
