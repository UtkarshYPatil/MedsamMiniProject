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

# ------------------------------------------------------------------- model
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/MedSAM2_CTLesion.pt")
MODEL_CFG = os.environ.get("MODEL_CFG", "configs/sam2.1_hiera_t512.yaml")
DEVICE = os.environ.get("DEVICE", "cpu")

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
except Exception:
    # The API and UI containers do not ship configs/, and do not need to.
    WINDOW_LEVEL = _int("WINDOW_LEVEL", -600)
    WINDOW_WIDTH = _int("WINDOW_WIDTH", 1500)
    BBOX_PADDING = _int("BBOX_PADDING", 0)
    PROPAGATE = _flag("PROPAGATE", True)


def ensure_dirs():
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    return {
        "device": DEVICE,
        "dry_run": DRY_RUN,
        "window": f"{WINDOW_LEVEL}/{WINDOW_WIDTH} HU",
        "bbox_padding_px": BBOX_PADDING,
        "propagate": PROPAGATE,
        "data_dir": str(DATA_DIR),
    }
