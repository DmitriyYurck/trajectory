"""Tests for pipeline.convert (no network, no secrets, no Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.convert import (
    COLUMNS,
    empty_row,
    make_row,
    parse_harbor_job,
    parse_r2e_traj,
    parse_swesmith_record,
    redact,
    tar_zst,
    write_parquet,
)


def test_empty_row_shape():
    row = empty_row("tbench", "run1", "task1")
    assert list(row.keys()) == COLUMNS
    assert row["source"] == "tbench"
    assert row["messages"] == "[]"
    assert row["error"] is None


def test_make_row_redacts_and_passes_assert():
    row = make_row(source="tbench", run_id="r1", task_id="t1", messages="api_key=sk-12345678901234567890 hello")
    assert "sk-" not in row["messages"]
    assert "[REDACTED]" in row["messages"]


def test_redact_hf_token():
    assert redact("x hf_abcdefghijklmnop12345678 y") == "x HF_TOKEN y"


def test_write_parquet_roundtrip(tmp_path: Path):
    row = make_row(
        source="tbench",
        run_id="r1",
        task_id="t1",
        teacher_model="m",
        messages=json.dumps([{"role": "user", "content": "hi"}]),
        reward=1.0,
        tests_passed=["t_pass"],
        tests_failed=[],
        patch="diff",
    )
    out = tmp_path / "r1-0.parquet"
    write_parquet([row], out)
    assert out.exists()
    table = pq.read_table(out)
    assert table.column_names == COLUMNS
    assert table.num_rows == 1
    assert table.column("run_id").to_pylist() == ["r1"]
    assert table.column("reward").to_pylist() == [1.0]


def test_write_parquet_atomic_no_tmp(tmp_path: Path):
    row = make_row(source="tbench", run_id="r1", task_id="t1")
    out = tmp_path / "sub" / "r1-0.parquet"
    write_parquet([row], out)
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_utc_iso_and_error_none():
    row = make_row(source="tbench", run_id="r", task_id="t", error=None)
    assert row["error"] is None
    assert row["started_at"].endswith("Z")


def test_parse_swesmith_minimal():
    rec = {"instance_id": "iid", "repo": "o/r", "llm_name": "m", "trajectory": []}
    row = parse_swesmith_record(rec, raw_log_path="/tmp/l.jsonl")
    assert row["source"] == "swesmith"
    assert row["task_id"] == "iid"
    assert row["raw_log_path"] == "/tmp/l.jsonl"


def test_parse_harbor_job(tmp_path: Path):
    job = tmp_path / "job1"
    job.mkdir()
    (job / "trial1.json").write_text(
        json.dumps(
            {
                "job_id": "j1",
                "task_id": "t1",
                "model": "m",
                "agent": "harbor",
                "reward": 0.5,
                "messages": [{"role": "user", "content": "go"}],
                "attempt": 0,
            }
        ),
        encoding="utf-8",
    )
    rows = parse_harbor_job(job, teacher_model="ignored")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "j1"
    assert rows[0]["reward"] == 0.5


def test_parse_r2e_traj(tmp_path: Path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"run_id": "r2", "task_id": "env1", "llm_name": "m", "reward": 1.0}), encoding="utf-8")
    row = parse_r2e_traj(p, teacher_model="fallback")
    assert row["source"] == "r2e"
    assert row["teacher_model"] == "m"


def test_tar_zst_fallback(tmp_path: Path):
    src = tmp_path / "log.txt"
    src.write_text("hello", encoding="utf-8")
    out = tmp_path / "log.tar.zst"
    tar_zst(src, out)
    assert out.exists() and out.stat().st_size > 0
