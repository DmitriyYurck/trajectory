"""Convert native logs -> unified Parquet schema. One row = one attempt."""

from __future__ import annotations

import json
import os
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS: list[str] = [
    "run_id",
    "task_id",
    "source",
    "repo",
    "commit",
    "docker_image",
    "teacher_model",
    "harness",
    "harness_version",
    "attempt",
    "messages",
    "reward",
    "tests_passed",
    "tests_failed",
    "patch",
    "n_steps",
    "n_input_tokens",
    "n_output_tokens",
    "cost_usd",
    "wall_time_s",
    "error",
    "started_at",
    "finished_at",
    "raw_log_path",
]

_SECRET_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(hf_[A-Za-z0-9]{20,})"), "HF_TOKEN"),
    (re.compile(r"(?i)(sk-(?:ant|proj|live)?-?[A-Za-z0-9\-_]{16,})"), "API_KEY"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;]+"), r"\1=[REDACTED]"),
]


def redact(obj: Any) -> Any:
    if isinstance(obj, str):
        s = obj
        for rx, repl in _SECRET_RES:
            s = rx.sub(repl, s)
        # normalize replacement marker
        s = re.sub(r"\[REDACTED\]", "[REDACTED]", s)
        return s
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def assert_no_secrets(obj: Any, where: str = "") -> None:
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    # Redacted placeholders are safe by construction: drop "key=[REDACTED]"
    # assignments before scanning so the generic pattern does not re-trigger.
    s = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\[REDACTED\]", "", s)
    for rx, _ in _SECRET_RES:
        if rx.search(s):
            raise ValueError(f"secret pattern matched in {where}: {rx.pattern}")


def utc_iso(dt: str | datetime | None) -> str:
    if dt is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return dt


def empty_row(source: str, run_id: str, task_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "source": source,
        "repo": "",
        "commit": "",
        "docker_image": "",
        "teacher_model": "",
        "harness": "",
        "harness_version": "",
        "attempt": 0,
        "messages": "[]",
        "reward": 0.0,
        "tests_passed": [],
        "tests_failed": [],
        "patch": "",
        "n_steps": 0,
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "cost_usd": 0.0,
        "wall_time_s": 0.0,
        "error": None,
        "started_at": utc_iso(None),
        "finished_at": utc_iso(None),
        "raw_log_path": "",
    }


def make_row(**fields: Any) -> dict[str, Any]:
    row = empty_row(fields.get("source", "tbench"), fields.get("run_id", ""), fields.get("task_id", ""))
    for k, v in fields.items():
        if k in COLUMNS:
            row[k] = v
    row = redact(row)
    assert_no_secrets(row, "row")
    return row


def write_parquet(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("task_id", pa.string()),
            ("source", pa.string()),
            ("repo", pa.string()),
            ("commit", pa.string()),
            ("docker_image", pa.string()),
            ("teacher_model", pa.string()),
            ("harness", pa.string()),
            ("harness_version", pa.string()),
            ("attempt", pa.int32()),
            ("messages", pa.string()),
            ("reward", pa.float64()),
            ("tests_passed", pa.list_(pa.string())),
            ("tests_failed", pa.list_(pa.string())),
            ("patch", pa.string()),
            ("n_steps", pa.int32()),
            ("n_input_tokens", pa.int64()),
            ("n_output_tokens", pa.int64()),
            ("cost_usd", pa.float64()),
            ("wall_time_s", pa.float64()),
            ("error", pa.string()),
            ("started_at", pa.string()),
            ("finished_at", pa.string()),
            ("raw_log_path", pa.string()),
        ]
    )
    cols: dict[str, list[Any]] = {c: [r.get(c, None) for r in rows] for c in COLUMNS}
    table = pa.table(cols, schema=schema)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)  # atomic rename
    return out_path


# --- Per-source parsers (defensive: accept real logs once envs exist; tests use synthetic) ---


def parse_harbor_job(job_dir: str | Path, teacher_model: str = "") -> list[dict[str, Any]]:
    """Harbor jobs dir: jobs/<job-id>/trials/... + config. Returns one row per trial/attempt."""
    rows: list[dict[str, Any]] = []
    job_dir = Path(job_dir)
    # Support single job dir or parent containing job dirs.
    trial_dirs = list(job_dir.rglob("trial*.json")) or list(job_dir.rglob("*.json"))
    for p in trial_dirs:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            make_row(
                run_id=data.get("job_id", job_dir.name),
                task_id=data.get("task_id", p.stem),
                source="tbench",
                repo=data.get("repo", ""),
                commit=data.get("commit", ""),
                docker_image=data.get("docker_image", ""),
                teacher_model=data.get("model", teacher_model),
                harness=data.get("agent", "harbor"),
                harness_version=data.get("harness_version", ""),
                attempt=int(data.get("attempt", 0)),
                messages=json.dumps(data.get("messages", data.get("trajectory", [])), ensure_ascii=False),
                reward=float(data.get("reward", 0.0) or 0.0),
                tests_passed=data.get("tests_passed", []),
                tests_failed=data.get("tests_failed", []),
                patch=data.get("patch", "") or "",
                n_steps=int(data.get("n_steps", 0) or 0),
                n_input_tokens=int(data.get("n_input_tokens", 0) or 0),
                n_output_tokens=int(data.get("n_output_tokens", 0) or 0),
                cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
                wall_time_s=float(data.get("wall_time_s", 0.0) or 0.0),
                error=data.get("error"),
                started_at=utc_iso(data.get("started_at")),
                finished_at=utc_iso(data.get("finished_at")),
                raw_log_path=str(p),
            )
        )
    return rows


def parse_swesmith_record(record: dict[str, Any], raw_log_path: str = "") -> dict[str, Any]:
    return make_row(
        run_id=record.get("run_id", record.get("instance_id", "")),
        task_id=record.get("instance_id", record.get("task_id", "")),
        source="swesmith",
        repo=record.get("repo", record.get("repo_name", "")),
        commit=record.get("base_commit", record.get("commit", "")),
        docker_image=record.get("docker_image", ""),
        teacher_model=record.get("model", record.get("llm_name", "")),
        harness=record.get("harness", "swe-agent"),
        harness_version=record.get("harness_version", ""),
        attempt=int(record.get("attempt", 0) or 0),
        messages=json.dumps(record.get("trajectory", record.get("messages", [])), ensure_ascii=False),
        reward=float(record.get("reward", 0.0) or 0.0),
        tests_passed=record.get("passed", []) or [],
        tests_failed=record.get("failed", record.get("tests_failed", [])) or [],
        patch=record.get("patch", "") or "",
        n_steps=int(record.get("steps", record.get("n_steps", 0)) or 0),
        n_input_tokens=int(record.get("n_input_tokens", 0) or 0),
        n_output_tokens=int(record.get("n_output_tokens", 0) or 0),
        cost_usd=float(record.get("cost_usd", 0.0) or 0.0),
        wall_time_s=float(record.get("wall_time_s", 0.0) or 0.0),
        error=record.get("error"),
        started_at=utc_iso(record.get("started_at")),
        finished_at=utc_iso(record.get("finished_at")),
        raw_log_path=raw_log_path,
    )


def parse_r2e_traj(traj_path: str | Path, teacher_model: str = "") -> dict[str, Any]:
    p = Path(traj_path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        # .traj / dir: try to load trajectory.json inside
        cand = p / "trajectory.json" if p.is_dir() else p
        data = json.loads(cand.read_text(encoding="utf-8")) if cand.exists() else {}
    stats = data.get("stats", data)
    return make_row(
        run_id=str(stats.get("run_id", p.stem)),
        task_id=str(stats.get("task_id", stats.get("env_id", p.stem))),
        source="r2e",
        repo=str(stats.get("repo", stats.get("repo_name", ""))),
        commit=str(stats.get("commit", stats.get("gt_commit", ""))),
        docker_image=str(stats.get("docker_image", "")),
        teacher_model=str(stats.get("llm_name", teacher_model)),
        harness=str(stats.get("harness", "r2e-edit")),
        harness_version=str(stats.get("harness_version", "")),
        attempt=int(stats.get("attempt", 0) or 0),
        messages=json.dumps(stats.get("trajectory", stats.get("messages", [])), ensure_ascii=False),
        reward=float(stats.get("reward", 0.0) or 0.0),
        tests_passed=stats.get("tests_passed", []) or [],
        tests_failed=stats.get("tests_failed", []) or [],
        patch=str(stats.get("patch", "") or ""),
        n_steps=int(stats.get("n_steps", stats.get("max_steps", 0)) or 0),
        n_input_tokens=int(stats.get("n_input_tokens", 0) or 0),
        n_output_tokens=int(stats.get("n_output_tokens", 0) or 0),
        cost_usd=float(stats.get("cost_usd", 0.0) or 0.0),
        wall_time_s=float(stats.get("wall_time_s", 0.0) or 0.0),
        error=stats.get("error"),
        started_at=utc_iso(stats.get("started_at")),
        finished_at=utc_iso(stats.get("finished_at")),
        raw_log_path=str(p),
    )


def tar_zst(path: str | Path, out: str | Path) -> Path:
    """Compress raw log dir/file into .tar.zst (falls back to .tar if zstd missing)."""
    src = Path(path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import zstandard as zstd

        tar_bytes = _make_tar_bytes(src)
        with open(out, "wb") as fh:
            c = zstd.ZstdCompressor(level=3)
            fh.write(c.compress(tar_bytes))
    except ImportError:
        import io

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            if src.is_dir():
                tf.add(src, arcname=src.name)
            else:
                tf.add(src, arcname=src.name)
        out.write_bytes(buf.getvalue())
    return out


def _make_tar_bytes(src: Path) -> bytes:
    import io

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        if src.is_dir():
            tf.add(src, arcname=src.name)
        else:
            tf.add(src, arcname=src.name)
    return buf.getvalue()
