#!/usr/bin/env bash
# Usage: bash runners/tbench.sh (one task, MAX_CONCURRENCY from .env)
# Harbor runner. Model and keys come ONLY from .env / environment.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing OPENROUTER_API_KEY in .env}"
: "${TEACHER_MODEL:?missing TEACHER_MODEL in .env}"
: "${OPENROUTER_BASE_URL:=https://openrouter.ai/api/v1}"
: "${MAX_CONCURRENCY:=8}"
export OPENROUTER_API_KEY OPENROUTER_BASE_URL
# LiteLLM-based harnesses take openrouter/<TEACHER_MODEL> (docs.litellm.ai/docs/providers/openrouter).
MODEL="openrouter/${TEACHER_MODEL}"
AGENT="${HARNESS_AGENT:-claude-code}"
mkdir -p jobs
echo "[tbench] teacher=${TEACHER_MODEL} litellm_model=${MODEL} agent=${AGENT} concurrency=${MAX_CONCURRENCY}"
exec envs/harbor/bin/harbor run --dataset terminal-bench@2.0 --agent "$AGENT" --model "$MODEL" --n-concurrent "$MAX_CONCURRENCY" -t hello-world/hello-world
