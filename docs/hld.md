# High-Level Design

Lung nodule measurement assistant — MedSAM2, LIDC-IDRI.
Last updated 2026-10-14.

## 1. Objective

Segment pulmonary nodules in chest CT and report their size and attenuation, so
a radiologist spends less time on manual measurement during lung cancer
screening.

**Scope boundary.** The system measures. It does not diagnose, score
malignancy, or recommend follow-up. All interpretation stays with the
radiologist. Research prototype on public de-identified data — not a medical
device.

## 2. Design principle: transfer learning, not training

MedSAM2 is used **frozen**. Nothing in this project trains a segmentation model.

The alternative — a fully trained model such as nnU-Net — needs large annotated
data covering every nodule subtype and scanner setting, and fails silently
outside the distribution it saw. A prompted foundation model generalises from
one prompt per target, which fits the data and compute available here.

| Component | Trained? |
|---|---|
| MedSAM2 segmenter | **No — frozen checkpoint** |
| Density characterisation | No — HU thresholds (provisional, see §7) |
| Measurement, metrics | No — deterministic |

## 3. Pipeline

```
LIDC-IDRI DICOM (1010 patients)  +  pylidc bundled annotation DB
            |
            v
[A] Annotation parsing      cluster 4 readers -> 50% consensus mask
            |
            v
[B] Preprocessing           HU -> lung window (L -600 / W 1500)
                            -> ROI crop around nodule centroid
            |
            v
[C] Prompt generation       bounding box on the largest-area slice
                            dev: from consensus mask | tool: from user click
            |
            v
[D] Segmentation            MedSAM2 (frozen), prompt one slice,
                            propagate the mask across adjacent slices
            |
            v
[E] Measurement             volume (mm3), mean/max in-plane diameter (mm),
                            HU profile (p10/p50/p90, core median)
            |
            v
[F] Self-check              confidence level + itemised quality flags
            |
            v
[G] Evaluation              Dice, IoU, HD95 vs consensus mask,
                            split by reader texture class
```

## 4. Components

| ID | Module | Responsibility |
|---|---|---|
| A | `src/data/parse_lidc.py` | reader consensus; (Y,X,Z)→(Z,Y,X) axis conversion, done once |
| B | `src/data/preprocess.py` | windowing, ROI cropping |
| C | `src/prompts/bbox.py` | box from mask (dev) or from click (tool) |
| D | `src/models/medsam2.py` | frozen MedSAM2 wrapper + offline dummy segmenter |
| E | `src/evaluate/metrics.py` | Dice, IoU, HD95, volume, diameter, HU stats |
| E | `src/report/density.py` | HU profile + provisional density class |
| F | `src/report/sizing.py` | measurement summary, confidence, flags |
| — | `app/app.py` | Streamlit radiologist-facing tool |

Entry points run in order: `00_verify_data` → `01_setup_pylidc` →
`02_prepare_data` → `04_sanity_check` → `03_run_baseline` →
`05_calibrate_density`.

## 5. Model

| | |
|---|---|
| Model | MedSAM2 (`MedSAM2_CTLesion.pt`) |
| Base architecture | SAM 2.1 — Hiera encoder, prompt encoder, mask decoder, memory attention |
| Upstream training | 455,000+ 3D image-mask pairs, 76,000+ annotated video frames |
| Why this checkpoint | fine-tuned for CT lesion segmentation, closest match to lung nodules |
| Comparison | `MedSAM2_latest.pt` |
| Source | https://huggingface.co/wanglab/MedSAM2 |
| Licence | CC-BY-SA-4.0, research and education only |

MedSAM2 is volumetric. Prompting one slice and propagating yields a 3D mask, so
volume is measured directly rather than extrapolated from one cross-section.

## 6. Data

| | |
|---|---|
| Dataset | LIDC-IDRI, 1010 patient folders at `E:/MiniProject/DATASET_ct/lidc_idri` |
| Annotations | pylidc's bundled database (derived from the LIDC XML); raw XML also held at `E:/MiniProject/LIDC-XML-only` |
| Consensus | 50% agreement across the 4 readers (`pylidc.utils.consensus`) |
| Inclusion | nodules ≥ 3 mm |
| Processed | ROI crops to `E:/MiniProject/processed`, outside the repository |

LIDC also records per-nodule malignancy ratings. These are carried as
**metadata only** and are never a prediction target.

## 7. Known limitations

**Prompts are not automatic.** In development the box comes from the consensus
mask, so metrics read as *segmentation quality given a correct box*, not
end-to-end detection. Automatic detection is Phase 3.

**Density classification is unvalidated.** On 30 nodules, HU-based
classification agreed with reader texture ratings 60% of the time. Every error
was a reader-rated *solid* nodule that attenuates like ground glass — e.g.
LIDC-IDRI-0136 nodules rated 4.3–5.0 by three readers with core median HU of
−382 to −398. Reader "texture" is a perceptual 1–5 judgement; HU is a physical
measurement. They correlate but are not the same quantity, and the solid class
spans the entire attenuation range (−579 to +216 HU), swallowing both other
classes. The tool therefore reports HU statistics and marks any density label
provisional. See `scripts/05_calibrate_density.py`.

**Ground-glass and part-solid nodules are the hardest case** regardless of
architecture — their boundaries are physically faint. Both are confidence-
flagged by default, as are masks under 20 voxels or spanning a single slice.

**The reference is a construct.** Readers disagree; `n_annotations` (1–4) is
carried on every result row so disagreement stays visible.

**Single dataset, single reader population.** Generalisation untested.

## 8. Environment

pylidc predates Python 3.12 and NumPy 2.x. `src/compat.py` restores
`configparser.SafeConfigParser` (removed in 3.12) and the `np.int` family
(removed in NumPy 1.24), and silences the `pkg_resources` warning. Both were
pure renames/aliases, so the patches change no behaviour. Without them pylidc
lists scans happily but cannot load an image or find a nodule.

## 9. Phase map

| Phase | Deliverable | State |
|---|---|---|
| 1 | Problem statement, synopsis | done |
| 2 | Model development & integration; metrics | in progress |
| 3 | Automatic nodule detection; full-stack integration | not started |
| 4 | Deployment & reproducibility | not started |
| 5 | Documentation | not started |
