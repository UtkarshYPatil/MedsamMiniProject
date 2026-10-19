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

RUN mkdir -p /data/cases /data/masks /app/checkpoints

CMD ["python", "-m", "services.worker.run"]
