"""Tests for pipeline.validate env checks (no network)."""

from __future__ import annotations

import pytest

from pipeline.validate import REQUIRED, check_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ["OPENROUTER_API_KEY", "TEACHER_MODEL", "HF_TOKEN", "HF_BUCKET"]:
        monkeypatch.delenv(k, raising=False)
    yield


def test_check_env_fails_when_missing(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        check_env()
    assert exc.value.code == 2


def test_check_env_ok(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    monkeypatch.setenv("TEACHER_MODEL", "provider/model")
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("HF_BUCKET", "user/bucket")
    cfg = check_env()
    assert cfg["TEACHER_MODEL"] == "provider/model"
    assert cfg["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_required_list():
    assert set(REQUIRED) == {"OPENROUTER_API_KEY", "TEACHER_MODEL", "HF_TOKEN", "HF_BUCKET"}
