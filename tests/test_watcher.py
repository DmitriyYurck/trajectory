"""Tests for pipeline.watcher (mocked upload, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pipeline.watcher import run_once


def _cfg(tmp_path: Path) -> dict:
    base = tmp_path
    (base / "outbox" / "pending").mkdir(parents=True, exist_ok=True)
    (base / "outbox" / "uploaded").mkdir(parents=True, exist_ok=True)
    (base / "outbox" / "failed").mkdir(parents=True, exist_ok=True)
    return {
        "teacher": {"temperature": 0.2},
        "pipeline": {
            "pipeline_log": str(base / "pipeline.log"),
            "outbox": "outbox",
            "batch_size": 50,
            "retries": 1,
            "poll_interval_s": 1,
            "watch_dirs": {"tbench": "jobs", "swesmith": "runs/swesmith", "r2e": "traj"},
        },
        "bucket": {},
    }


def test_run_once_no_new(tmp_path: Path):
    res = run_once(_cfg(tmp_path), tmp_path)
    assert res["status"] == "no_new" and res["converted"] == 0


def test_run_once_converts_and_uploads(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEACHER_MODEL", "provider/model")
    monkeypatch.setenv("HF_BUCKET", "user/bucket")
    cfg = _cfg(tmp_path)
    job = tmp_path / "jobs" / "job1"
    job.mkdir(parents=True)
    (job / "trial1.json").write_text(
        json.dumps({"job_id": "j1", "task_id": "t1", "model": "m", "reward": 1.0}), encoding="utf-8"
    )
    with patch("pipeline.watcher.upload_batch", return_value={"status": "ok", "uploaded": 1, "Uploaded": 1}) as mock_up:
        res = run_once(cfg, tmp_path)
    assert res["converted"] == 1 and res["Uploaded"] == 1
    mock_up.assert_called_once()
    # Parquet staged in outbox/pending.
    pend = list((tmp_path / "outbox" / "pending").glob("*.parquet"))
    assert len(pend) == 1


def test_env_fallback_teacher_model(monkeypatch):
    # Watcher must resolve teacher model from env, not config.
    monkeypatch.setenv("TEACHER_MODEL", "env-model")
    monkeypatch.setenv("HF_BUCKET", "env-bucket")
    from pipeline.watcher import _resolve_bucket, _resolve_teacher_model

    assert _resolve_teacher_model({"teacher": {}}) == "env-model"
    assert _resolve_bucket({"bucket": {}}) == "env-bucket"
