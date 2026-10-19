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

    if not settings.DRY_RUN:
        # Pay the model load now, at boot, rather than on the first user's job.
        from services.worker.tasks import get_segmenter
        log.info("loading the model...")
        get_segmenter()
        log.info("model ready on %s", settings.DEVICE)

    # SimpleWorker runs jobs in-process instead of forking. Forking per job
    # would discard the loaded model every time, which on CPU costs more than
    # the inference itself.
    SimpleWorker([Queue(settings.QUEUE_NAME, connection=conn)],
                 connection=conn).work(logging_level="INFO")


if __name__ == "__main__":
    main()
