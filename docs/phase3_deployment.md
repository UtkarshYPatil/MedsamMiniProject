# Phase 3 — full-stack deployment

Five containers. The model runs on the server, never on the user's machine,
which is why "anyone can use this" and "it needs a GPU" are not in tension.

```
browser
   │  HTTP
   ▼
ui        Streamlit. No torch, no weights. Talks only to the API.
   │
   ▼
api       FastAPI. Moves bytes and rows. Enqueues; never segments.
   │                 │
   │ enqueue         └──────► db      Postgres: cases + measurements
   ▼
redis     RQ queue
   │
   ▼
worker    The only container that loads MedSAM2. Scale with --scale worker=N
```

## Run it

```bash
cp .env.example .env          # Windows: copy .env.example .env
docker compose up --build
```

Before the first build, check that it will work:

```bash
python scripts/10_preflight.py          # add --gpu for the GPU overlay
```

It verifies the Docker daemon, both compose files, every Dockerfile `COPY`
source, the ports, free disk, and the checkpoint -- and for `--gpu`, that the
host driver is actually reachable from a container. Exit 0 means go. Every
check in it corresponds to something that has failed here for real.

| | |
|---|---|
| UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/healthz |

It needs **no dataset and no GPU**, but it does need the checkpoint:

1. Put `MedSAM2_CTLesion.pt` in `./checkpoints/`
   (`python scripts/download_checkpoint.py`)
2. `docker compose up --build`

The weights are not in the image — they are research-and-education-use only, so
redistributing them is not ours to do. Without them the worker **refuses to
start** and the log says what to do. That is deliberate: real inference is the
default, and a tool that reports millimetres has no business inventing them.

To demonstrate the stack with no checkpoint at all, set `DRY_RUN=1`. The dummy
segmenter stands in, and every number it produces is labelled a dry run in the
API response and behind a banner in the UI, because it is made up.

### Device

`DEVICE=auto` is the default: the worker asks torch at start-up whether a card
is actually visible and falls back to CPU when it is not, so the same image
runs on a laptop and on a GPU host unchanged.

Only the worker can answer this — it is the only container with torch — so it
publishes what it resolved to Redis and `/healthz` reports that under `worker`,
alongside `workers_alive` from RQ's own registry. The API's `config.device` is
its own setting and is not evidence about the GPU; an earlier build reported
the API's value and said `cpu` while the worker was running on CUDA.

On CPU expect **15–40 s per nodule**; on GPU about **1 s**. The CPU path is
usable for a demonstration, not for routine reading.

### GPU

Check the host first:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If that prints your card, overlay the GPU file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The overlay changes only the worker — same code, CUDA base image, one GPU
reserved. On Windows this works through Docker Desktop with the WSL2 backend.

## Tests

```bash
pip install -r services/requirements-dev.txt
python -m pytest tests/ -q
```

Two files, split by what they can prove:

- `tests/test_services.py` — the modules. Storage round-trip, PNG rendering,
  the worker task end to end against the dummy segmenter, device resolution,
  the auto-prompt (including a volume containing nothing but chest wall, where
  the correct answer is to refuse), the shape-plausibility checks, and a test
  that imports every module in the project.
- `tests/test_api_http.py` — the routes, over real HTTP, with Postgres swapped
  for SQLite and Redis for fakeredis. Status codes, the upload cap, the
  suggest→segment handoff, and `/healthz` reporting the worker's device rather
  than its own. No Docker, no services, runs anywhere.

The HTTP file skips cleanly if the dev requirements are not installed; the rest
of the suite still runs.

Two bugs in this project were invisible to module tests and found by the other
kinds: a handler handed the browser `http://api:8000/...`, which only resolves
inside the Docker network, and a docstring closed one line early in
`src/report/sizing.py`, which parses but cannot import. `compileall` on
`services/` alone missed the second one, which is why the import test walks
`src/` too.

## Sharing it

Someone else can run the stack without the repository, without Python and
without a build. They need Docker, one file, and the checkpoint.

**The weights are never published.** `MedSAM2_CTLesion.pt` is mounted from the
host at runtime and is absent from every image, because MedSAM2's weights are
research-and-education-use only. `scripts/publish_images.ps1` greps each built
image for them and refuses to push if it finds any.

### Publish once

```powershell
.\scripts\publish_images.ps1              # GHCR, tag "latest"
.\scripts\publish_images.ps1 -Tag v1.0    # a fixed version
.\scripts\publish_images.ps1 -Gpu         # also the CUDA worker
```

GHCR packages are **private by default**, and that is not only an access
question -- it is a quota. GitHub Pro includes 2 GB of storage for private
packages; public packages are exempt. The worker image alone is larger than
that allowance, so it cannot be published privately at all.

Over the limit the failure is misleading: the blobs upload normally and only
the manifest is refused with `denied`, so a large image fails after minutes of
visible progress while a smaller one pushed a moment earlier succeeded. It
reads like a permissions problem and is not one.

Set each package to public at `https://github.com/UtkarshYPatil?tab=packages`,
which also lets anyone pull without `docker login`. A package has to exist
before its visibility can be changed, so if a push never completed, create it
with a placeholder first:

```powershell
docker tag hello-world ghcr.io/utkarshypatil/nodule-worker:init
docker push ghcr.io/utkarshypatil/nodule-worker:init
```

Set it public, push `:latest`, then delete the `init` version.

### What they do

1. Put `MedSAM2_CTLesion.pt` in a `./checkpoints` folder beside the compose
   file — from <https://huggingface.co/wanglab/MedSAM2>, 156 MB.
2. `docker compose -f docker-compose.pull.yml up`
3. Open <http://localhost:8501>

With no checkpoint at all, `DRY_RUN=1` brings the stack up on a dummy
segmenter so the plumbing can be seen. Every number it produces is labelled
invented, in the API response and behind a banner in the UI.

For GPU: `WORKER_TAG=latest-gpu`, `DEVICE=cuda`, and uncomment the `deploy`
block under `worker`.

### Choosing how to share

| | recipient needs | size to move | when |
|---|---|---|---|
| **Registry** | Docker + 1 file + checkpoint | they pull ~3 GB | showing it to someone |
| **`docker save`** | Docker + checkpoint | you move 4–6 GB | they are offline |
| **The repo** | Docker + Python + checkpoint | a clone | they will change the code |

A tarball, if you need one:

```powershell
docker save ghcr.io/utkarshypatil/nodule-api:latest `
            ghcr.io/utkarshypatil/nodule-worker:latest `
            ghcr.io/utkarshypatil/nodule-ui:latest `
            postgres:16-alpine redis:7-alpine -o nodule-stack.tar
# they run: docker load -i nodule-stack.tar
```

### One caveat

Images are built for the architecture you build on. An image built on your
x86_64 Windows machine will not run natively on an Apple Silicon Mac — Docker
emulates it, slowly, and torch sometimes does not survive the translation. If
the recipient is on an M-series Mac, have them use the repo and build locally.

## DICOM upload

`POST /api/cases/dicom` takes a **.zip of one DICOM series**. The UI exposes it
as the first tab under "Add a case".

Four things have to be right or every millimetre afterwards is wrong, and three
of them fail silently. `services/common/dicomseries.py` handles each, and
`tests/test_dicom.py` tests each against a synthetic series:

| | what goes wrong if skipped |
|---|---|
| **Rescale to HU** | `HU = slope*raw + intercept`, per slice. Skip it and a −600 HU nodule reads as +424; the density classifier inverts and nothing looks broken. |
| **Slice order** | Files arrive in zip order. Sorting by name puts IM10 before IM2. Order comes from `ImagePositionPatient`. |
| **Slice spacing** | `SliceThickness` is what was reconstructed; the gap between slice centres is what the volume has. They differ on overlapping or gapped reconstructions, and volume scales directly with it. Measured from consecutive positions, with thickness as fallback. |
| **Axis order** | `pixel_array` is (rows, cols) = (Y, X), so stacking gives (Z, Y, X) and spacing is `(dz, PixelSpacing[0], PixelSpacing[1])`. Transposing this still looks like a chest. |

Refused with a readable reason: a non-CT series (HU only exist on CT), fewer
than 3 slices, a series whose slices all share one position, a zip containing
no DICOM images, and anything that is not a zip. A series whose minimum value
is above −500 HU is flagged, because lung air sits near −1000 and anything
higher means the rescale never happened.

Several series in one zip is normal — exports carry scouts and dose reports —
so the largest CT series wins and the response says so.

**Patient data.** Only pixels and geometry are read. No name, ID, birth date,
accession or institution is returned or stored; the case note carries only
geometry. This is research and education software on public de-identified data
and real patient scans do not belong in it.

### The nodule is not located for you

A whole series gets **no automatic prompt**. `/suggest` scores candidates by
how central they are, which holds for an ROI cut around a centroid and is
meaningless on a full chest, where the centre is mediastinum. It returns
`box: null` with `source: "unavailable"` and the UI asks the radiologist to
find the nodule and place the box. A confident box on the heart would be worse
than no box at all.

This is the tool's stated boundary, not a gap: **it measures what it is pointed
at; it does not detect nodules.**

### Cropping to the validated geometry

Every accuracy figure in this project was measured on 128×128×33 crops
(`configs/config.yaml`: `roi_size 128`, `roi_depth 16` — slices *each side*, so
2d+1 = 33). A series is 512×512×~250.

So the worker crops that same geometry around whatever box it is given before
segmenting, and pastes the mask back into full-volume coordinates. The model
sees what it saw during evaluation, the propagation does not have hundreds of
slices to drift through, and results stay comparable to `outputs/results.csv`.
An uploaded ROI is already that size and passes through untouched — the result
reports `cropped_to_roi` either way.

## Why the API never segments inline

Inference takes seconds to a minute. A request that waits for it times out
behind any reverse proxy, and the browser gives up first. So:

```
POST /api/cases                   201  {case_id}        upload an ROI (.npz)
POST /api/cases/phantom           201  {case_id}        synthetic, for demos
POST /api/cases/dicom             201  {case_id}        upload a CT series (.zip)
GET  /api/cases/{id}/suggest      200  {slice, box}     proposed prompt
GET  /api/cases/{id}/slice/{z}.png                      windowed slice
POST /api/cases/{id}/segment      202  {job_id}         body: {slice, box}
GET  /api/jobs/{job_id}           200  {status, result} poll this
GET  /api/jobs/{job_id}/overlay.png                     mask drawn on the slice
```

Uploads are capped at `MAX_UPLOAD_MB` (64 by default) and the cap is enforced
while reading the body, not after it is buffered. `CORS_ORIGINS` defaults to
`*`, which is right for localhost and wrong for anything on a network.

`/suggest` is what makes the UI one button. It is a **proposal, not a finding**:
it assumes the nodule is near the middle of the crop, which holds for an ROI
cut around a centroid and not for a whole scan, and it returns an `info` object
saying how it arrived at the box so the proposal can be questioned rather than
trusted. Components touching the edge of the crop are discarded, which is what
keeps it off the chest wall.

Load becomes latency instead of failure, and scaling becomes a flag.

## Scaling

```bash
docker compose up --scale worker=4
```

Four workers drain one queue with no coordination and no code change. That is
the whole reason the model is not called from the request handler.

**One GPU is further than it sounds.** At ~1 s per nodule a single card does
~86,000 nodules/day. A thousand radiologists at 20 scans/day and ~2 nodules per
scan is ~40,000/day — under half a card.

What breaks first is **bandwidth**, not compute: a full CT series is ~200 MB,
so 1000 users is ~4 TB/day. Two decisions already in the design address that:

- **A case is an ROI, not a scan.** The model consumes a 128×128×33 crop —
  about 1 MB against 200 MB. That is a ~200× reduction in bytes crossing the
  wire, and it keeps the full series inside the hospital, which is the better
  answer on privacy as well as cost.
- **Uploads can go straight to object storage.** Swap the `data` volume for
  S3/R2/MinIO and have the browser PUT to a presigned URL, so the API never
  touches the bytes.

## Hosting

Split the always-on parts from the GPU. A GPU idle 99% of the time is the
entire cost problem.

| Layer | Where | Cost |
|---|---|---|
| ui, api, db, redis | Fly.io / Railway / small VPS / Azure free services | ~$0–10/month |
| worker | serverless GPU, scale to zero | per second |

At 10,000 inferences/month that is roughly **$10–15** against **$250–360** for
an always-on T4.

With a GitHub Student Developer Pack: Camber gives 5 GPU hours/month, Azure
gives $100 of credit, Heroku gives $13/month for 24 months, and GHCR plus
Actions build and host the images free for a public repo. Set a spending alert
the day you create anything on Azure credit.

**Cold starts.** Scale-to-zero means the first request after idle pays for
container start, torch import and loading a 156 MB checkpoint — 20–40 s. Keep
one worker warm during a demo, load the checkpoint from a mounted volume
rather than from inside the image, and tell the user it is warming up. Silence
reads as broken.

## Boundaries

**The model licence does not permit deployment.** The MedSAM2 model card tags
`cc-by-sa-4.0` but its License section says the weights *"can only be used for
research and education purposes."* Those conflict, and the explicit sentence
is the one to obey. This stack is a research and education tool.

That is also why the weights are **not baked into any image** — they are
mounted at `/app/checkpoints`. Redistributing them inside a container is not
ours to do.

**Only LIDC-IDRI data.** Public, de-identified, consented for research. Never
a real patient scan.

**This is not a medical device.** Software that measures nodules and informs
follow-up is regulated (FDA, CE, CDSCO). Nothing here has been through that,
and the UI says so on every page.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `failed to resolve source metadata` | no network to the registry, or a proxy without Docker Hub allowed |
| API up, worker idle, jobs stay `queued` | worker crashed at boot — `docker compose logs worker`; usually a missing checkpoint with `DRY_RUN=0` |
| `job is failed` with a torch traceback | on Windows, usually the missing Visual C++ runtime on the host, not a container problem |
| Build context is enormous | `.dockerignore` missing or edited — it must exclude `checkpoints/`, `external/`, `outputs/masks/` |
| UI says the API is unreachable | the api container is still starting; it has a healthcheck, give it ~20 s |

## What is verified, and what is not

Verified: all 43 tests pass, including 15 that drive the storage round-trip,
the PNG rendering and the full worker task against the dummy segmenter;
both compose files validate.

**Not verified: the images have never been built.** Docker Hub was unreachable
from the environment these files were written in, so `docker compose build`
has not run end to end. Expect to fix something on the first build — a pinned
base-image tag is the likeliest candidate. The Python the containers run is
tested; the containers themselves are not yet.
