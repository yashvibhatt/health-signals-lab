"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_env(env_file: Path = ROOT / ".env") -> list[str]:
    """Load KEY=VALUE lines from .env into os.environ, overriding existing variables.

    Values in .env win over the shell environment so a rotated key takes effect without
    unsetting old exports. Empty values are skipped. Returns the names that were set.
    """
    if not env_file.exists():
        return []
    loaded = []
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config.yaml (created if missing)."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
