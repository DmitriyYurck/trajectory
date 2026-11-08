#!/usr/bin/env bash
# Usage: bash runners/r2e.sh (one task, MAX_CONCURRENCY from .env)
# R2E-Gym runner. Model and keys come ONLY from .env / environment.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${OPENROUTER_API_KEY:?missing OPENROUTER_API_KEY in .env}"
: "${TEACHER_MODEL:?missing TEACHER_MODEL in .env}"
: "${OPENROUTER_BASE_URL:=https://openrouter.ai/api/v1}"
: "${MAX_CONCURRENCY:=8}"
export OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENAI_API_BASE="${OPENROUTER_BASE_URL}" OPENAI_API_KEY="${OPENROUTER_API_KEY}"
echo "[r2e] teacher=${TEACHER_MODEL} workers=${MAX_CONCURRENCY}"
cd envs/r2egym/main
exec uv run python src/r2egym/agenthub/run/edit.py runagent_multiple --traj_dir ./traj --max_workers "$MAX_CONCURRENCY" --start_idx 0 --k 1 --dataset "R2E-Gym/R2E-Gym-Lite" --split train --llm_name "$TEACHER_MODEL" --use_fn_calling True --exp_name smoke --temperature 0.2 --max_steps 40
