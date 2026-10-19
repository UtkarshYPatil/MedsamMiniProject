# Worker, CPU build. Runs on any machine -- this is the one that makes
# `docker compose up` work without a GPU.
#
# CPU torch wheels are ~200 MB against ~2.8 GB for the CUDA build, so this
# image is about a quarter the size of the GPU one. Inference is roughly
# 15-40 s per nodule instead of 1 s: usable for a demo, not for a clinic.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cpu

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

COPY services/requirements-worker.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# sam2 comes from the MedSAM2 repo, which is gitignored and so not in the
# build context. --no-build-isolation stops pip fetching its own torch.
ARG MEDSAM2_REF=main
RUN git clone --depth 1 --branch "${MEDSAM2_REF}" \
      https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2 \
 && pip install --no-cache-dir -e /opt/MedSAM2 --no-build-isolation \
 || echo "sam2 install failed - the worker will still run with DRY_RUN=1"

COPY src/      /app/src/
COPY configs/  /app/configs/
COPY services/ /app/services/

# Model weights are NOT baked in. They are mounted at /app/checkpoints.
# MedSAM2's weights are research-and-education-use only, so redistributing
# them inside an image is not ours to do. See docs/phase3_deployment.md.
VOLUME ["/app/checkpoints"]

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/cases /data/masks /app/checkpoints \
 && chown -R app:app /app /data
USER app

CMD ["python", "-m", "services.worker.run"]
