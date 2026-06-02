"""Project-local environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_ENV_LOADED = False


def _load_env_fallback(path: Path, *, override: bool) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value[:1] == value[-1:] and value[:1] in {"\"", "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_project_env(*, override: bool = False) -> None:
    """Load environment variables from the repository-local .env file once."""
    global _ENV_LOADED

    if _ENV_LOADED and not override:
        return
    if not _PROJECT_ENV_PATH.exists():
        _ENV_LOADED = True
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_fallback(_PROJECT_ENV_PATH, override=override)
    else:
        load_dotenv(_PROJECT_ENV_PATH, override=override)

    _ENV_LOADED = True