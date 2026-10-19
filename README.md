# Nodule measurement assistant

Minor project — B.Tech CSE (AI&ML), MIT Vishwaprayag University.

Segments pulmonary nodules in chest CT and reports their size and attenuation,
so a radiologist spends less time on manual measurement during lung cancer
screening.

**Scope.** The tool measures. It does not diagnose, score malignancy, or
recommend a clinical action. All interpretation stays with the radiologist.
Research prototype on public de-identified data — not a medical device.

## How it works

MedSAM2 (`MedSAM2_CTLesion.pt`) is used **frozen** — nothing here trains a
segmentation model. A bounding-box prompt on one slice is propagated through
the volume by MedSAM2's memory attention, giving a 3D mask. Measurements come
from that mask.

```
CT volume → box prompt on one slice → MedSAM2 (frozen) → 3D mask
          → diameter / volume / HU profile → confidence flags → radiologist
```

In development the box comes from the LIDC consensus outline, so metrics read
as *segmentation quality given a correct box* — not end-to-end detection.
In the tool the box comes from the user's click.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Installs torch (CUDA 12.1), requirements, the vendored MedSAM2 package, and the
checkpoints — then verifies all four. Safe to re-run.

Manual equivalent:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e external\MedSAM2
python scripts\download_checkpoint.py
```

`external\MedSAM2` is already vendored (bowang-lab/MedSAM2 @ 332f30d).

## Run, in order

```powershell
python scripts\00_verify_data.py --dicom "E:/MiniProject/DATASET_ct/lidc_idri" --xml "E:/MiniProject/LIDC-XML-only"
python scripts\01_setup_pylidc.py --dicom "E:/MiniProject/DATASET_ct/lidc_idri"
python scripts\02_prepare_data.py --limit 300
python scripts\04_sanity_check.py          # LOOK AT THE PNG
python scripts\03_run_baseline.py --limit 50
python scripts\05_calibrate_density.py
python scripts\09_interreader.py          # what Dice can a human even reach?
streamlit run app\app.py
```

`02_prepare_data.py` appends to the manifest, so `--start 300 --limit 300`
resumes without redoing work. `03_run_baseline.py --dry-run` swaps in a dummy
segmenter for pipeline testing without a GPU; its numbers are meaningless.

`09_interreader.py` measures how much the four radiologists disagree with each
other on the same nodules, which is the ceiling our Dice is competing against.
Run it before trying to improve accuracy: if a single reader scores 0.85
against the same consensus the model is scored against, then 0.804 has about
0.05 of headroom and effort is better spent on detection.

## Data

| Source | Contents |
|---|---|
| LIDC-IDRI images | 1010 patient folders, `E:/MiniProject/DATASET_ct/lidc_idri` |
| Annotations | pylidc's bundled database (derived from the LIDC XML) |
| Raw XML | `E:/MiniProject/LIDC-XML-only`, 1319 files — reference only |

Nodules are scored against the 50% consensus of up to four radiologists. That
consensus is a construct, not absolute truth, so `n_annotations` is carried on
every result row.

## Environment note

pylidc predates Python 3.12 and NumPy 2.x. `src/compat.py` restores
`configparser.SafeConfigParser` (removed in 3.12) and the `np.int` family
(removed in NumPy 1.24). Both were pure renames, so the patches change no
behaviour. Without them pylidc lists scans happily but cannot load a single
image or find a single nodule.

## Layout

```
configs/config.yaml     paths, window, ROI size, thresholds
external/MedSAM2/       vendored upstream model code
src/compat.py           pylidc shims for modern Python/NumPy
src/data/               LIDC parsing, HU windowing, ROI cropping
src/prompts/            box from mask (dev) or from click (tool)
src/models/medsam2.py   frozen MedSAM2 wrapper + dummy segmenter
src/evaluate/metrics.py Dice, IoU, HD95, volume, diameter, HU stats
src/report/density.py   HU profile + provisional density class
src/report/sizing.py    measurement summary + confidence flags
scripts/                00 verify → 01 setup → 02 prepare → 04 sanity
                        → 03 baseline → 05 calibrate
app/app.py              Streamlit tool (single process, local)
services/               Phase 3 service stack
  common/               env settings, case storage, PNG rendering, phantom
  api/                  FastAPI: upload, enqueue, poll, overlay
  worker/               RQ worker; the only process that loads the model
  ui/                   Streamlit client, talks only to the API
docker/                 one Dockerfile per service (+ a CUDA worker variant)
docker-compose.yml      full stack, CPU, runs anywhere
docker-compose.gpu.yml  overlay: swaps the worker onto CUDA
tests/                  43 unit tests (28 research + 15 service)
docs/hld.md             high-level design
docs/model_card.md      model version, metrics, limitations
docs/phase3_deployment.md  running, scaling and hosting the stack
```

## Run the full stack

```bash
cp .env.example .env
docker compose up --build
# UI  http://localhost:8501     API  http://localhost:8000/docs
```

Needs **no dataset and no GPU**, but it does need the checkpoint: put
`MedSAM2_CTLesion.pt` in `checkpoints/` (`python scripts/download_checkpoint.py`).
The weights are research-and-education-use only, so they are not baked into the
image; without them the worker refuses to start and the log says why. Real
inference is the default — set `DRY_RUN=1` only to demonstrate the plumbing, and
every number it then produces is labelled as invented.

`DEVICE=auto` picks the GPU when one is visible and falls back to CPU when it is
not. For GPU, add `-f docker-compose.gpu.yml`. Upload a case or create a phantom
in the sidebar, then press **Segment** — the prompt box is proposed for you.
Details and hosting notes in
[`docs/phase3_deployment.md`](docs/phase3_deployment.md).

## Known limitations

- **Detection is not automatic.** The tool proposes a prompt box inside an ROI
  crop, which is enough for one button in the UI, but it does not find nodules
  in a whole scan. It assumes the nodule is roughly central and discards
  anything touching the edge of the crop. All metrics remain conditional on the
  box being right, which is why every result carries a confidence flag.
- **Density classification is unvalidated.** On 30 nodules it agreed with
  radiologist texture ratings 60% of the time, and every error was a
  reader-rated *solid* nodule that attenuates like ground glass. HU and
  perceptual texture are not the same quantity. The tool reports HU statistics
  and marks any density label provisional. See `docs/hld.md` §7.
- **Ground-glass and part-solid nodules are hardest** regardless of
  architecture — faint boundaries. Both are confidence-flagged by default.
- Single dataset, single reader population. Generalisation untested.
- MedSAM2 weights are licensed for research and education only.

## References

- Ma et al., *MedSAM2: Segment Anything in 3D Medical Images and Videos*,
  arXiv:2504.03600. Weights: https://huggingface.co/wanglab/MedSAM2
- Armato et al., *The Lung Image Database Consortium (LIDC) and Image Database
  Resource Initiative (IDRI)*, Medical Physics, 2011.
