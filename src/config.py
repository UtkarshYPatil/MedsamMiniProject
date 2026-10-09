"""Load configs/config.yaml into a dot-accessible object."""
from pathlib import Path
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def load(path: str | Path = None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return _ns(yaml.safe_load(f))


CFG = load()
