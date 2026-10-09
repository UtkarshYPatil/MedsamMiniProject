# Model Card

Fill the metric fields after `scripts/run_baseline.py` completes.

## Model

| Field | Value |
|---|---|
| Name | MedSAM2 |
| Checkpoint | `MedSAM2_CTLesion.pt` |
| Version / source | wanglab/MedSAM2, Hugging Face |
| Base architecture | SAM 2.1 |
| Paper | Ma et al., *MedSAM2: Segment Anything in 3D Medical Images and Videos*, arXiv:2504.03600 |
| Licence | CC-BY-SA-4.0 (research and education only) |
| Modified by us? | **No -- used frozen, inference only** |

## Intended use

Segmenting pulmonary nodules in chest CT to produce size and density
measurements that assist a radiologist during screening.

**Out of scope:** diagnosis, malignancy prediction, triage, or any clinical
decision. The output is a measurement, not a finding.

## Input / output

| | |
|---|---|
| Input | CT volume ROI, HU, lung window L -600 / W 1500, 3-channel uint8 |
| Prompt | bounding box on the largest-area nodule slice |
| Output | boolean 3D mask |
| Derived | volume (mm3), max in-plane diameter (mm), mean HU |

## Evaluation

| | |
|---|---|
| Dataset | LIDC-IDRI |
| Reference | 50% consensus of 4 radiologist contours |
| Scans evaluated | _TBD_ |
| Nodules evaluated | _TBD_ |
| Split | by patient |

### Results

| Metric | Mean | Std |
|---|---|---|
| Dice | _TBD_ | _TBD_ |
| IoU | _TBD_ | _TBD_ |
| HD95 (mm) | _TBD_ | _TBD_ |

### By nodule texture

| Texture | n | Dice |
|---|---|---|
| Solid | _TBD_ | _TBD_ |
| Part-solid | _TBD_ | _TBD_ |
| Ground-glass | _TBD_ | _TBD_ |

## Limitations

- Prompts are derived from ground-truth masks; detection is not yet automatic.
- Expected to perform worst on ground-glass and part-solid nodules.
- Single dataset, single scanner population -- generalisation untested.
- Weights are licensed for research and education only.

## Reproducibility

| | |
|---|---|
| Seed | 42 |
| Config | `configs/config.yaml` |
| Commit | _TBD_ |
| Hardware | _TBD_ |
