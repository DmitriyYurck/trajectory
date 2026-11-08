"""Tests for pipeline.upload (mocked HF Bucket, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pipeline.upload import _atomic_stage, bucket_path, upload_batch


class _Info:
    def __init__(self, path, size):
        self.path = path
        self.size = size


def _setup(tmp_path: Path):
    base = tmp_path / "outbox"
    for d in ("pending", "uploaded", "failed"):
        (base / d).mkdir(parents=True, exist_ok=True)
    log = tmp_path / "pipeline.log"
    f = tmp_path / "r1-0.parquet"
    f.write_bytes(b"parquet-bytes")
    return base, log, f


def test_bucket_path_shape():
    p = bucket_path("tbench", "mod", "run1", 2)
    assert p.startswith("parquet/source=tbench/teacher=mod/date=")
    assert p.endswith("run1-2.parquet")


def test_atomic_stage(tmp_path: Path):
    src = tmp_path / "a.parquet"
    src.write_bytes(b"data")
    pending = tmp_path / "pending" / "a.parquet"
    out = _atomic_stage(src, pending)
    assert out.exists() and out.read_bytes() == b"data"
    assert not pending.with_name(pending.name + ".tmp").exists()


def test_upload_success_moves_to_uploaded(tmp_path: Path):
    base, log, f = _setup(tmp_path)
    remote = bucket_path("tbench", "m", "r1", 0)
    info = _Info(remote, f.stat().st_size)
    with (
        patch("huggingface_hub.batch_bucket_files") as mock_batch,
        patch("huggingface_hub.get_bucket_paths_info", side_effect=[[], [info]]),
    ):
        res = upload_batch("user/bucket", [(f, remote)], log, retries=1, outbox_base=base)
    mock_batch.assert_called_once()
    assert res["status"] == "ok" and res["Uploaded"] == 1
    assert (base / "uploaded" / Path(remote).name).exists()


def test_upload_idempotent_skip(tmp_path: Path):
    base, log, f = _setup(tmp_path)
    remote = bucket_path("tbench", "m", "r1", 0)
    info = _Info(remote, 999)
    with (
        patch("huggingface_hub.batch_bucket_files") as mock_batch,
        patch("huggingface_hub.get_bucket_paths_info", return_value=[info]),
    ):
        res = upload_batch("user/bucket", [(f, remote)], log, outbox_base=base)
    mock_batch.assert_not_called()
    assert res["status"] == "skipped"


def test_upload_failure_moves_to_failed(tmp_path: Path):
    base, log, f = _setup(tmp_path)
    remote = bucket_path("tbench", "m", "r1", 0)
    with (
        patch("huggingface_hub.batch_bucket_files", side_effect=RuntimeError("boom")),
        patch("huggingface_hub.get_bucket_paths_info", return_value=[]),
        patch("pipeline.upload.time.sleep", return_value=None),
    ):
        res = upload_batch("user/bucket", [(f, remote)], log, retries=2, outbox_base=base)
    assert res["status"] == "failed"
    assert (base / "failed" / Path(remote).name).exists()
    events = [json.loads(x) for x in log.read_text().splitlines()]
    assert any(e.get("event") == "failed" for e in events)


def test_upload_missing_file_raises(tmp_path: Path):
    base, log, _ = _setup(tmp_path)
    try:
        with patch("huggingface_hub.get_bucket_paths_info", return_value=[]):
            upload_batch("user/bucket", [(tmp_path / "nope.parquet", "x")], log, outbox_base=base)
        raise AssertionError("should have raised")
    except FileNotFoundError:
        pass
