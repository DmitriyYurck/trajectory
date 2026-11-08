"""Fail-fast preflight: env presence, cheap OpenRouter call, HF Bucket probe.

Usage: python -m pipeline.validate --config config.yaml
All secrets come ONLY from .env / environment. Never logged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

REQUIRED = ["OPENROUTER_API_KEY", "TEACHER_MODEL", "HF_TOKEN", "HF_BUCKET"]


def fail(msg: str) -> NoReturn:
    print(f"[validate] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def check_env() -> dict:
    base = Path(__file__).resolve().parent.parent
    load_dotenv(base / ".env")
    missing = [k for k in REQUIRED if not os.getenv(k, "").strip()]
    if missing:
        fail(f"missing required env vars: {', '.join(missing)}. Fill .env from .env.example.")
    return {
        "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
        "TEACHER_MODEL": os.environ["TEACHER_MODEL"],
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_BUCKET": os.environ["HF_BUCKET"],
        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "MAX_CONCURRENCY": os.getenv("MAX_CONCURRENCY", "8"),
    }


def check_openrouter(cfg: dict) -> None:
    import requests

    key, model, base = cfg["OPENROUTER_API_KEY"], cfg["TEACHER_MODEL"], cfg["OPENROUTER_BASE_URL"].rstrip("/")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Model existence + tool-calling support.
    r = requests.get(f"{base}/models", headers=h, timeout=30)
    if r.status_code != 200:
        fail(f"OpenRouter /models failed: HTTP {r.status_code}")
    data = r.json().get("data", [])
    ids = {m.get("id") for m in data}
    if model not in ids:
        # Single-model lookup (aliases resolve; 404 if unknown).
        r2 = requests.get(f"{base}/model/{model}", headers=h, timeout=30)
        if r2.status_code != 200:
            fail(f"TEACHER_MODEL does not exist on OpenRouter: {model} (HTTP {r2.status_code})")
        entry = r2.json().get("data", {})
    else:
        entry = next(m for m in data if m.get("id") == model)
    params = entry.get("supported_parameters") or []
    if "tools" not in params:
        print("[validate] WARN: TEACHER_MODEL does not list 'tools' support; fn-calling harnesses may fail.")
    # One cheap chat call; read usage/cost.
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 1}
    r3 = requests.post(f"{base}/chat/completions", headers=h, json=payload, timeout=60)
    if r3.status_code != 200:
        fail(f"OpenRouter chat probe failed: HTTP {r3.status_code}: {r3.text[:200]}")
    usage = r3.json().get("usage") or {}
    print(f"[validate] OpenRouter OK model={model} usage_prompt={usage.get('prompt_tokens')} cost={usage.get('cost')}")


def check_bucket(cfg: dict) -> None:
    from huggingface_hub import batch_bucket_files, bucket_info

    bucket = cfg["HF_BUCKET"]
    try:
        info = bucket_info(bucket)
        print(f"[validate] Bucket OK id={info.id} private={info.private} size={info.size}")
    except Exception as e:
        fail(f"HF bucket_info failed for {bucket}: {e}")
    probe = f"_probe/probe-{os.getpid()}.txt"
    try:
        batch_bucket_files(bucket, add=[(b"probe", probe)])
        batch_bucket_files(bucket, delete=[probe])
    except Exception as e:
        fail(f"HF Bucket write/delete probe failed for {bucket}: {e}")
    print("[validate] Bucket write/delete probe OK")


def check_disk(config_path: str = "config.yaml") -> float:
    import shutil

    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    pre = cfg.get("preflight", {})
    threshold = float(pre.get("min_free_gb", 300))
    free_gb = shutil.disk_usage(Path(".").resolve()).free / (1024**3)
    print(f"[validate] free disk: {free_gb:.1f} GB (threshold {threshold:.0f} GB)")
    if free_gb < threshold:
        fail(f"free disk {free_gb:.1f} GB < required {threshold:.0f} GB. Stop and report.")
    return free_gb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-network", action="store_true", help="Skip OpenRouter/Bucket probes (offline).")
    args = ap.parse_args()
    check_disk(args.config)
    cfg = check_env()
    print(f"[validate] env OK bucket={cfg['HF_BUCKET']} model={cfg['TEACHER_MODEL']}")
    if args.skip_network:
        print("[validate] network probes skipped (--skip-network).")
    else:
        check_openrouter(cfg)
        check_bucket(cfg)
    print("[validate] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
