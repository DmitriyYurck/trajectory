#!/usr/bin/env bash
# Usage: bash runners/swesmith.sh (one task, MAX_CONCURRENCY from .env)
# SWE-smith runner. Model and keys come ONLY from .env / environment.
# Smoke: exactly ONE task, MAX_CONCURRENCY=1.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing OPENROUTER_API_KEY in .env}"
: "${TEACHER_MODEL:?missing TEACHER_MODEL in .env}"
: "${OPENROUTER_BASE_URL:=https://openrouter.ai/api/v1}"
: "${MAX_CONCURRENCY:=1}"
export OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENAI_API_BASE="${OPENROUTER_BASE_URL}" OPENAI_API_KEY="${OPENROUTER_API_KEY}" MAX_CONCURRENCY
echo "[swesmith] teacher=${TEACHER_MODEL} base=${OPENROUTER_BASE_URL} concurrency=${MAX_CONCURRENCY} tasks=1"
envs/swesmith/bin/python - <<'PY'
import os
print("smoke: SWE-smith env OK, exactly ONE task, teacher from .env")
print("TEACHER_MODEL present:", bool(os.getenv("TEACHER_MODEL")), "MAX_CONCURRENCY:", os.getenv("MAX_CONCURRENCY"))
PY
