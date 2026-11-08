"""Shared pytest fixtures and path setup (English only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `pipeline` importable when running pytest from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_base(tmp_path: Path):
    outbox = tmp_path / "outbox"
    for d in ("pending", "uploaded", "failed"):
        (outbox / d).mkdir(parents=True, exist_ok=True)
    return tmp_path
