"""Config loader: YAML -> dot access."""
from pathlib import Path
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) for k, v in d.items()}) if isinstance(d, dict) else d


def load(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        return _ns(yaml.safe_load(f))


CFG = load()
