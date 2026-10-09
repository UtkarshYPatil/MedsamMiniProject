# High-Level Design

## 1. Objective

Segment pulmonary nodules in chest CT and compute their size and density, so
that a radiologist spends less time on manual measurement during lung cancer
screening. The system does not diagnose.

## 2. Design principle

**Transfer learning, not training.** MedSAM2 is used frozen. A fully trained
model such as nnU-Net would need large annotated data covering every nodule
subtype and scan quality, and fails silently outside the distribution it saw.
A prompted foundation model generalises from one prompt per target, which fits
the data and compute available for this project.

## 3. Pipeline

```
LIDC-IDRI DICOM  +  radiologist XML
            |
            v
[A] Annotation parsing      pylidc -> cluster 4 radiologists -> consensus mask
            |
            v
[B] Preprocessing           HU conversion -> lung window (L -600 / W 1500)
                            -> ROI crop around nodule centroid
            |
            v
[C] Prompt generation       bounding box on the largest-area slice
                            (Phase 2: from ground truth. Phase 3: from a detector)
            |
            v
[D] Segmentation            MedSAM2 (frozen), prompt one slice,
                            propagate the mask across adjacent slices
            |
            v
[E] Measurement             volume (mm3), max in-plane diameter (mm), mean HU
            |
            v
[F] Evaluation              Dice, IoU, HD95 vs consensus mask
```

## 4. Components

| ID | Module | Responsibility | Trained? |
|---|---|---|---|
| A | `src/data/parse_lidc.py` | XML -> consensus nodule masks | No |
| B | `src/data/preprocess.py` | windowing, ROI cropping | No |
| C | `src/prompts/bbox.py` | box / point prompt from mask | No |
| D | `src/models/medsam2.py` | MedSAM2 inference wrapper | **Frozen** |
| E | `src/evaluate/metrics.py` | measurements | No |
| F | `src/evaluate/metrics.py` | Dice, IoU, HD95 | No |

Nothing in Phase 2 is trained. The only learned component is the pretrained
MedSAM2 checkpoint.

## 5. Model

| | |
|---|---|
| Model | MedSAM2 (`MedSAM2_CTLesion.pt`) |
| Base architecture | SAM 2.1 (Hiera image encoder, prompt encoder, mask decoder, memory attention) |
| Training data (upstream) | 455,000+ 3D image-mask pairs, 76,000+ annotated video frames |
| Why this checkpoint | fine-tuned for CT lesion segmentation, the closest match to lung nodules |
| Baseline comparison | `MedSAM2_latest.pt` |
| Source | https://huggingface.co/wanglab/MedSAM2 |
| Licence | CC-BY-SA-4.0, research and education only |

MedSAM2 is volumetric. Prompting one slice and propagating gives a 3D mask, so
volume is available directly rather than estimated from a single cross-section.

## 6. Data

| | |
|---|---|
| Dataset | LIDC-IDRI (TCIA) |
| Scans | 1,018 chest CT |
| Annotations | 4 independent radiologists per scan, XML (separate download) |
| Consensus | 50% agreement level via `pylidc.utils.consensus` |
| Inclusion | nodules >= 3 mm diameter |
| Licence | public, de-identified |

LIDC also carries per-nodule attribute ratings (malignancy, subtlety, texture).
These are **metadata only** -- recorded for analysis, never predicted.

## 7. Phase boundaries

| Phase | Deliverable |
|---|---|
| 2 (current) | Frozen MedSAM2 integrated; box prompt from ground truth; Dice/IoU/HD95 on a subset |
| 3 | Automatic nodule detector replacing the ground-truth prompt; full-stack integration |
| 4 | Deployment and reproducibility |
| 5 | Documentation |

## 8. Known limitations

- Phase 2 numbers assume a correct bounding box. They measure segmentation
  quality, not end-to-end detection performance.
- Ground-glass and part-solid nodules have the weakest boundaries and are
  expected to score lowest. Results are reported split by nodule texture.
- Radiologists disagree with each other; the consensus mask is a construct, not
  absolute truth. Nodules marked by fewer than 4 readers are flagged.
