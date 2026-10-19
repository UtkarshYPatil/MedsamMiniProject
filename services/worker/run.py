"""
Worker entry point.

    python -m services.worker.run

Scale by running more of these -- `docker compose up --scale worker=4`. They
share one Redis queue and need no coordination, which is the whole reason the
API does not call the model directly.
"""
import logging
import sys

from redis import Redis
from rq import Queue, SimpleWorker

from services.common import settings


def main():
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("worker")

    settings.ensure_dirs()
    log.info("config: %s", settings.summary())
    if settings.DRY_RUN:
        log.warning("DRY_RUN is on - the dummy segmenter is in use and masks are "
                    "meaningless. Unset DRY_RUN for real inference.")

    conn = Redis.from_url(settings.REDIS_URL)
    conn.ping()
    log.info("connected to %s, queue %r", settings.REDIS_URL, settings.QUEUE_NAME)

    from services.worker.tasks import device_detail, resolve_device
    dev = resolve_device()
    log.info("device: %s", device_detail())
    if settings.DEVICE == "auto" and dev == "cpu":
        log.warning("no usable GPU visible - falling back to CPU. Inference will "
                    "take 15-40 s per nodule instead of about 1 s. For GPU, add "
                    "-f docker-compose.gpu.yml and check the host with: "
                    "docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi")

    from services.worker.tasks import checkpoint_present, publish_status
    if not settings.DRY_RUN:
        if not checkpoint_present():
            # Refuse to start rather than accept jobs it cannot serve. The
            # alternative -- a traceback on every job -- hides the one fact
            # that matters behind a per-request error.
            log.error("no checkpoint at %s", settings.CHECKPOINT)
            log.error("Place MedSAM2_CTLesion.pt in ./checkpoints (it is not "
                      "shipped in the image: the weights are research-and-"
                      "education-use only), or set DRY_RUN=1 to run the dummy "
                      "segmenter, whose numbers are meaningless.")
            sys.exit(2)

        # Pay the model load now, at boot, rather than on the first user's job.
        from services.worker.tasks import get_segmenter
        log.info("loading the model...")
        get_segmenter()
        log.info("model ready on %s", dev)

    # Tell /healthz what this worker actually resolved, so the UI can stop
    # guessing from the API's own environment.
    publish_status(conn)

    # SimpleWorker runs jobs in-process instead of forking. Forking per job
    # would discard the loaded model every time, which on CPU costs more than
    # the inference itself.
    SimpleWorker([Queue(settings.QUEUE_NAME, connection=conn)],
                 connection=conn).work(logging_level="INFO")


if __name__ == "__main__":
    main()
