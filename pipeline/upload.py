"""Upload Parquet + raw logs to HF Bucket with retries, idempotency, atomic outbox."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _log(log_path: Path, event: dict[str, Any]) -> None:
    event = {"ts": datetime.now(UTC).isoformat(), **event}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_stage(src: Path, pending: Path) -> Path:
    """Atomically move a prepared file into outbox/pending/."""
    src = Path(src)
    pending = Path(pending)
    pending.parent.mkdir(parents=True, exist_ok=True)
    # Stage via .tmp in same dir, then rename.
    tmp = pending.with_name(pending.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, pending)
    return pending


def bucket_path(source: str, teacher: str, run_id: str, attempt: int, kind: str = "parquet") -> str:
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    fname = f"{run_id}-{attempt}.parquet" if kind == "parquet" else f"{run_id}-{attempt}.tar.zst"
    if kind == "parquet":
        return f"parquet/source={source}/teacher={teacher}/date={date}/{fname}"
    return f"raw/parquet/source={source}/teacher={teacher}/date={date}/{fname}"


def upload_batch(
    bucket_id: str,
    files: list[tuple[Path, str]],  # (local, remote)
    log_path: Path,
    retries: int = 5,
    base_delay_s: float = 2.0,
    outbox_base: Path | None = None,
) -> dict[str, Any]:
    """Batch upload via batch_bucket_files; idempotent existence check; exp-backoff >=5; verify."""
    base_dir = Path(outbox_base) if outbox_base is not None else Path("outbox")
    pending_dir = base_dir / "pending"
    uploaded_dir = base_dir / "uploaded"
    failed_dir = base_dir / "failed"
    pending_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import (
        batch_bucket_files,
        get_bucket_paths_info,
    )

    if not bucket_id or not bucket_id.strip():
        raise ValueError("bucket_id is empty — set HF_BUCKET in .env")
    added: list[tuple[str, str]] = []  # (local_str, remote)
    for local, remote in files:
        local = Path(local)
        if not local.exists():
            raise FileNotFoundError(local)
        # Idempotency: skip if exists in bucket (verify path metadata without download).
        try:
            infos = list(get_bucket_paths_info(bucket_id, [remote]))
            if infos and getattr(infos[0], "path", None):
                _log(log_path, {"event": "exists_skip", "remote": remote, "size": getattr(infos[0], "size", None)})
                continue  # do not overwrite; conflict with same name + different content requires manual decision
        except Exception as e:
            _log(log_path, {"event": "exists_check_failed", "remote": remote, "error": str(e)})
        staged = _atomic_stage(local, pending_dir / Path(remote).name)
        added.append((str(staged), remote))
    if not added:
        return {"status": "skipped", "Uploaded": 0}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            batch_bucket_files(bucket_id, add=[(p, r) for p, r in added])
            # Verify size/hash remotely where possible (size at minimum).
            for p, r in added:
                local_size = Path(p).stat().st_size
                infos = list(get_bucket_paths_info(bucket_id, [r]))
                if not infos:
                    raise RuntimeError(f"verify failed, remote not found: {r}")
                remote_size = getattr(infos[0], "size", None)
                if remote_size is not None and remote_size != local_size:
                    raise RuntimeError(f"verify size mismatch {r}: local={local_size} remote={remote_size}")
                # success: move to uploaded
                dest = uploaded_dir / Path(r).name
                shutil.move(str(p), str(dest))
                _log(log_path, {"event": "uploaded", "remote": r, "local_size": local_size, "sha256": _sha256(dest)})
            return {"status": "ok", "Uploaded": len(added)}
        except Exception as e:
            last_exc = e
            _log(log_path, {"event": "upload_attempt_failed", "attempt": attempt, "error": str(e)})
            if attempt >= retries:
                break
            time.sleep(base_delay_s * (2 ** (attempt - 1)))
    # Final failure: move to failed/
    for p, r in added:
        dest = failed_dir / Path(r).name
        try:
            shutil.move(str(p), str(dest))
        except FileNotFoundError:
            pass
        _log(log_path, {"event": "failed", "remote": r, "error": str(last_exc)})
    return {"status": "failed", "Uploaded": 0, "error": str(last_exc)}
