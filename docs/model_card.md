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
| Scans processed | 296 |
| Nodules evaluated | 902 |
| Split | by patient |
| Hardware | NVIDIA RTX 4060 Laptop GPU (CUDA 12.8) |
| Seed | 42 |
| Commit | `28ee7d7` |

### Segmentation quality

| Metric | Mean | Std | Median |
|---|---|---|---|
| Dice | 0.766 | 0.142 | 0.800 |
| IoU | 0.640 | 0.165 | 0.667 |
| HD95 (mm) | 2.760 | 4.099 | 1.250 |

### By reader texture class

Ground-glass and part-solid are expected to score lowest — faint boundaries.

| Texture | n | Dice mean | Dice std |
|---|---|---|---|
| Solid | 767 | 0.776 | 0.139 |
| Part-solid | 43 | 0.779 | 0.093 |
| Ground-glass | 92 | 0.679 | 0.159 |

### Measurement agreement

| | |
|---|---|
| Diameter error, mean | −0.01 mm |
| Diameter error, abs mean | 0.55 mm |
| Within 1 mm of reference | 92 % |

### Accuracy tracks reader agreement

The reference is a 50% consensus, so a nodule marked by one radiologist has a
weaker reference than one all four agreed on. Dice follows that exactly:

| Readers who marked it | n | Dice mean |
|---|---|---|
| 1 | 262 | 0.692 |
| 2 | 175 | 0.742 |
| 3 | 163 | 0.790 |
| 4 | 299 | 0.834 |

The gradient held from n=137 to n=902 unchanged in shape. Quote 0.766 as the
headline, but 0.834 is the figure on nodules where the reference is strongest,
and the spread is a property of the reference as much as of the model.

### The dominant failure mode: over-segmentation along Z

Mean diameter is essentially unbiased (−0.01 mm, 92% within 1 mm) while volume
runs 12% high at the median and 90% high at p90. Those two facts together
locate the error: diameter is in-plane, volume is area × depth, so the excess
is on slices where the nodule has ended. Volume ratio correlates −0.65 with
Dice; diameter ratio only −0.26. MedSAM2's propagation continues past the end
of the nodule.

**Report diameter with confidence. Qualify volume.**

### Latency

| | |
|---|---|
| Seconds per nodule (mean) | 1.67 (GPU, including I/O) |

## Limitations

1. **Prompts are not automatic.** The bounding box is derived from the
   reference mask, so these figures measure *segmentation quality given a
   correct box*, not end-to-end detection. Automatic detection is Phase 3.
2. **Density classification does not work and is not reported.** On 902
   nodules it agrees with reader texture 51% of the time. Answering "solid"
   every time would agree 85% of the time, so it is 34 points worse than a
   constant. The earlier "60% on 30 nodules" was a small sample flattering a
   classifier that has no signal. The tool now shows the HU statistics it was
   derived from — those are real measurements — and withholds the label.
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
