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

| | |
|---|---|
| UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/healthz |

It comes up with **no dataset and no GPU**. `DRY_RUN=1` is the default, so the
dummy segmenter stands in for the model and the UI shows a warning banner
saying so. Create a phantom case from the sidebar and press Segment to
exercise the whole path.

### Real inference

1. Put `MedSAM2_CTLesion.pt` in `./checkpoints/`
   (`python scripts/download_checkpoint.py`)
2. Set `DRY_RUN=0` in `.env`
3. `docker compose up --build`

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

## Why the API never segments inline

Inference takes seconds to a minute. A request that waits for it times out
behind any reverse proxy, and the browser gives up first. So:

```
POST /api/cases                   201  {case_id}        upload an ROI (.npz)
POST /api/cases/phantom           201  {case_id}        synthetic, for demos
POST /api/cases/{id}/segment      202  {job_id}         body: {slice, box}
GET  /api/jobs/{job_id}           200  {status, result} poll this
GET  /api/jobs/{job_id}/overlay.png                     mask drawn on the slice
```

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
