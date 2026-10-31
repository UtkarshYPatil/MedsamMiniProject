# Worker, CUDA build. Same code as worker.Dockerfile, different base.
#
# Needs the NVIDIA Container Toolkit on the host. Verify before building:
#     docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
#
# 'runtime' rather than 'devel' -- about 3 GB smaller and nothing here compiles
# CUDA kernels.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cuda

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY services/requirements-worker.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

ARG MEDSAM2_REF=main
RUN git clone --depth 1 --branch "${MEDSAM2_REF}" \
      https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2 \
 && pip install --no-cache-dir -e /opt/MedSAM2 --no-build-isolation

COPY src/      /app/src/
COPY configs/  /app/configs/
COPY services/ /app/services/

VOLUME ["/app/checkpoints"]

# Same uid as the API and the CPU worker. /data is a volume SHARED with the
# API, so the two must write as the same user or whatever one creates the
# other cannot touch.
#
# This image used to run as root. Everything worked until the volume happened
# to be initialised by this container rather than the API -- then /data/cases
# belonged to root and every upload failed with
#   PermissionError: [Errno 13] Permission denied: '/data/cases/<id>.npz'
# which looks like a code bug and is a uid mismatch. Whether it broke depended
# on container start order, so it worked for days and then did not.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/cases /data/masks /app/checkpoints \
 && chown -R app:app /app /data
USER app

CMD ["python", "-m", "services.worker.run"]
