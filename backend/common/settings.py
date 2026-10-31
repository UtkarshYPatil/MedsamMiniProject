"""
Runtime configuration for the service stack, from the environment.

configs/config.yaml holds the research settings and carries absolute Windows
paths (E:/MiniProject/...) that do not exist inside a container. The services
therefore take their paths from the environment and borrow only the
path-independent values -- window level/width, prompt padding -- from CFG.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _flag(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ paths
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CASES_DIR = DATA_DIR / "cases"
MASKS_DIR = DATA_DIR / "masks"

# ------------------------------------------------------------ infrastructure
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://nodule:nodule@db:5432/nodule")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "segmentation")
API_URL = os.environ.get("API_URL", "http://api:8000")

# Where the worker publishes what it resolved (device, checkpoint) and the API
# reads it for /healthz. It lives here because it is a contract between two
# containers: hardcoding the string at both ends means a rename silently stops
# the API from ever seeing a worker again, with no error anywhere.
WORKER_STATUS_KEY = "worker:status"

# Comma-separated origins, or "*". Fine as "*" on localhost; narrow it for
# anything reachable from a network.
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

# An ROI is about 1 MB. This is generous for a crop and still refuses a whole
# series, which would be a 200 MB upload the model cannot consume anyway.
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 64)

# A zipped CT series is far larger than an ROI -- a few hundred 512x512 slices.
# Separate cap so raising it does not also let someone post a 500 MB "ROI".
MAX_DICOM_MB = _int("MAX_DICOM_MB", 600)

# The caps above bound COMPRESSED bytes. Zeros compress ~1000:1, so a 0.2 MB
# .npz expands to a 210 MB array. This bounds what may be materialised.
# A full CT series is ~250 MB uncompressed, so 2 GB is generous for real data
# and still refuses a bomb.
MAX_DECOMPRESSED_MB = _int("MAX_DECOMPRESSED_MB", 2048)

# ------------------------------------------------------------------- model
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/MedSAM2_CTLesion.pt")
MODEL_CFG = os.environ.get("MODEL_CFG", "configs/sam2.1_hiera_t512.yaml")

# "auto" (the default), "cuda" or "cpu".
#
# auto asks torch at worker start-up whether a usable GPU is visible and falls
# back to CPU when it is not, so the same image runs on a laptop without a card
# and on a GPU host without changing anything. Only the worker can resolve this
# -- the api and ui containers have no torch, which is deliberate -- so
# backend.worker.tasks.resolve_device() does the work and reports what it
# picked in every job result.
DEVICE = os.environ.get("DEVICE", "auto").strip().lower()

# DRY_RUN swaps in DummySegmenter: no weights, no GPU, meaningless masks.
# It exists so the whole stack can be brought up and demonstrated on a machine
# that has neither. Any result produced this way is labelled as such.
DRY_RUN = _flag("DRY_RUN", False)

JOB_TIMEOUT = _int("JOB_TIMEOUT", 600)

# --------------------------------------------------- research settings (CFG)
try:
    from src.config import CFG
    WINDOW_LEVEL = _int("WINDOW_LEVEL", CFG.preprocess.window_level)
    WINDOW_WIDTH = _int("WINDOW_WIDTH", CFG.preprocess.window_width)
    BBOX_PADDING = _int("BBOX_PADDING", CFG.prompt.bbox_padding_px)
    PROPAGATE = _flag("PROPAGATE", CFG.prompt.propagate)
    ROI_SIZE = _int("ROI_SIZE", CFG.preprocess.roi_size)
    ROI_DEPTH = _int("ROI_DEPTH", CFG.preprocess.roi_depth)
except Exception:
    # The API and UI containers do not ship configs/, and do not need to.
    WINDOW_LEVEL = _int("WINDOW_LEVEL", -600)
    WINDOW_WIDTH = _int("WINDOW_WIDTH", 1500)
    BBOX_PADDING = _int("BBOX_PADDING", 0)
    PROPAGATE = _flag("PROPAGATE", True)
    # Must match configs/config.yaml: the geometry every reported number was
    # measured on. roi_depth is slices EACH SIDE, so the crop is 2d+1 deep.
    ROI_SIZE = _int("ROI_SIZE", 128)
    ROI_DEPTH = _int("ROI_DEPTH", 16)


def ensure_dirs():
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    return {
        "device": DEVICE,
        "dry_run": DRY_RUN,
        "window": f"{WINDOW_LEVEL}/{WINDOW_WIDTH} HU",
        "bbox_padding_px": BBOX_PADDING,
        "roi": f"{2 * ROI_DEPTH + 1}x{ROI_SIZE}x{ROI_SIZE}",
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_dicom_mb": MAX_DICOM_MB,
        "propagate": PROPAGATE,
        "data_dir": str(DATA_DIR),
    }
