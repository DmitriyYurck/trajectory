"""Poll watch_dirs, convert new logs, batch-upload; network failure never stops agent runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from pipeline.convert import parse_harbor_job, parse_r2e_traj, parse_swesmith_record, write_parquet
from pipeline.upload import bucket_path, upload_batch


def _seen(log: Path) -> set[str]:
    if not log.exists():
        return set()
    s: set[str] = set()
    for line in log.read_text().splitlines():
        try:
            s.add(json.loads(line).get("raw_log_path", ""))
        except Exception:
            continue
    return s


def _resolve_teacher_model(config: dict) -> str:
    m = os.getenv("TEACHER_MODEL") or config.get("teacher", {}).get("model") or ""
    return m


def _resolve_bucket(config: dict) -> str:
    b = os.getenv("HF_BUCKET") or config.get("bucket", {}).get("bucket_id") or ""
    return b


def run_once(config: dict, base: Path) -> dict:
    teacher_model = _resolve_teacher_model(config)
    bucket_id = _resolve_bucket(config)
    outbox = base / config["pipeline"].get("outbox", "./outbox")
    log = base / config["pipeline"]["pipeline_log"]
    names: list[tuple[Path, str]] = []
    seen = _seen(log)
    for src, watch in config["pipeline"]["watch_dirs"].items():
        w = base / watch
        if not w.exists():
            continue
        if src == "tbench":
            for job in w.glob("*"):
                if job.is_dir():
                    for row in parse_harbor_job(job, teacher_model):
                        if row.get("raw_log_path") in seen:
                            continue
                        remote = bucket_path("tbench", teacher_model, row["run_id"], row["attempt"], "parquet")
                        tmp = outbox / "pending" / Path(remote).name
                        write_parquet([row], tmp)
                        names.append((tmp, remote))
        elif src == "r2e":
            for p in list(w.rglob("*.json")) + list(w.rglob("*.traj")):
                if str(p) in seen:
                    continue
                try:
                    row = parse_r2e_traj(p, teacher_model)
                except Exception:
                    continue
                remote = bucket_path("r2e", teacher_model, row["run_id"], row["attempt"], "parquet")
                tmp = outbox / "pending" / Path(remote).name
                write_parquet([row], tmp)
                names.append((tmp, remote))
        elif src == "swesmith":
            for p in w.glob("*.jsonl"):
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    row = parse_swesmith_record(rec, str(p))
                    remote = bucket_path("swesmith", teacher_model, row["run_id"], row["attempt"], "parquet")
                    tmp = outbox / "pending" / Path(remote).name
                    write_parquet([row], tmp)
                    names.append((tmp, remote))
    if not names:
        return {"status": "no_new", "converted": 0, "Uploaded": 0}
    batch = config["pipeline"].get("batch_size", 50)
    total_up = 0
    for i in range(0, len(names), batch):
        chunk = names[i : i + batch]
        try:
            res = upload_batch(bucket_id, chunk, log, retries=config["pipeline"].get("retries", 5), outbox_base=outbox)
            total_up += res.get("Uploaded", 0)
        except Exception as e:
            with open(log, "a") as fh:
                fh.write(json.dumps({"event": "watcher_error", "error": str(e)}) + "\n")
    return {"status": "ok", "converted": len(names), "Uploaded": total_up}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    base = Path(".")
    cfg = yaml.safe_load((base / args.config).read_text())
    if args.once:
        print(json.dumps(run_once(cfg, base), ensure_ascii=False, indent=2))
        return
    while True:
        try:
            run_once(cfg, base)
        except Exception as e:
            print(f"[watcher] error: {e}")
        time.sleep(cfg["pipeline"].get("poll_interval_s", 2))


if __name__ == "__main__":
    main()
