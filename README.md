# Lung Nodule Segmentation in CT Scans using MedSAM2

Minor project — B.Tech CSE (AI&ML), MIT Vishwaprayag University.

Segmenting pulmonary nodules in chest CT to reduce the manual measurement
effort radiologists spend during lung cancer screening.

## Scope

- **In scope:** pulmonary nodule segmentation and measurement (diameter, area,
  volume, density) from CT, using a pretrained MedSAM2 checkpoint.
- **Out of scope:** diagnosis, malignancy prediction, or any clinical verdict.
  The system measures; the radiologist decides.
- **Not a medical device.** Research prototype on public, de-identified data.

## Approach

MedSAM2 (`MedSAM2_CTLesion.pt`) is used **frozen** — no retraining. Nodule
annotations from LIDC-IDRI supply a bounding-box prompt; MedSAM2 produces the
mask and propagates it across adjacent slices. Measurements are computed from
the resulting mask and scored against the radiologist consensus contour.

```
CT volume -> nodule ROI -> box prompt -> MedSAM2 -> mask -> measurement -> Dice / IoU / HD95
```

## Data

| Source | Contents |
|---|---|
| LIDC-IDRI (TCIA) | 1,018 chest CT scans, DICOM |
| LIDC-IDRI XML annotations | Contours from 4 independent radiologists |

Both are public and de-identified. **The images and the XML annotations are two
separate downloads on TCIA** — the `.tcia` manifest pulls images only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# MedSAM2 code (provides the sam2 package)
git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
pip install -e external/MedSAM2

# checkpoint
python scripts/download_checkpoint.py
```

Set the dataset paths in `configs/config.yaml` before running anything.

## Run

```bash
python scripts/prepare_data.py      # parse XML, build consensus masks, crop ROIs
python scripts/run_baseline.py      # MedSAM2 inference + metrics
python scripts/make_report.py       # figures and results table
```

Results land in `outputs/results.csv` and `outputs/figures/`.

## Status

Phase 2 — model development and integration.

- [x] Repository and folder structure
- [x] High-level design (`docs/hld.md`)
- [ ] LIDC-IDRI download complete
- [ ] XML annotations parsed, consensus masks built
- [ ] MedSAM2 baseline inference running
- [ ] Metrics on a held-out subset
- [ ] Model card populated (`docs/model_card.md`)

## Repository layout

See `docs/hld.md` for the full design. Top level:

```
configs/     run configuration (paths, model, thresholds)
data/        raw / interim / processed  (gitignored)
docs/        design, model card, figures
src/         library code
scripts/     entry points
outputs/     masks, figures, results  (gitignored except results.csv)
tests/       unit tests
```

## License and attribution

MedSAM2 weights are released for research and education only
(CC-BY-SA-4.0, WangLab UofT). LIDC-IDRI is distributed by TCIA under its own
terms. This repository's own code is for academic coursework.
