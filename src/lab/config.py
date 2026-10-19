"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config.yaml (created if missing)."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
