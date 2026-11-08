# trajectory

Converts agent run logs from Terminal-Bench (Harbor), SWE-smith, and R2E-Gym into one Parquet row per attempt and uploads the Parquet file plus the raw log to a Hugging Face Bucket. Resumable and idempotent. `make test` covers deterministic file names and upload skip; live crash recovery is untested (no secrets here).

Install and test (from the repo root; each line run):

```bash
git clone https://github.com/DmitriyYurck/trajectory.git
cd trajectory
uv sync --extra dev
make lint && make test
```

Set `MAX_CONCURRENCY=1` for one task per tool. Monitor disk with `df -h`.

## Status

| Component | Status | Tested how | Limits |
|---|---|---|---|
| Pipeline (`pipeline/`: convert, upload, watcher, validate) | verified | `make test` (22 tests: Parquet round-trip, 3 parsers, mocked upload, tmp dirs) | Synthetic logs and mocked upload only; no real harness logs in repo |
| Terminal-Bench via Harbor, SWE-smith, R2E-Gym | untested | Not run here | Harbor 0.25.0, Python >=3.12; 300-500 MB per gym image |
| OpenRouter and HF Bucket live probes | untested | Not run here | Need `OPENROUTER_API_KEY` + `TEACHER_MODEL` and `HF_TOKEN` + `HF_BUCKET` |

## Requirements

- Linux, Python 3.12+ and `uv`. Docker for harness runs. Free disk: 300 GB for full runs (`config.yaml:preflight`). Accounts: OpenRouter and Hugging Face only.

## Configuration

Copy the template, then fill the four required variables. Teacher model is read from `TEACHER_MODEL` in `runners/*.sh`.

```bash
cp .env.example .env && chmod 600 .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Teacher key |
| `TEACHER_MODEL` | yes | OpenRouter slug, e.g. `provider/model-name` |
| `HF_TOKEN` | yes | Write scope on `HF_BUCKET` |
| `HF_BUCKET` | yes | Bucket id, e.g. `user/bucket-name` |
| `OPENROUTER_BASE_URL` | no | Default `https://openrouter.ai/api/v1` |
| `MAX_CONCURRENCY` | no | Default 8; use 1 for a smoke run |

`config.yaml` sets the disk gate (`preflight.min_free_gb` 300), `pipeline.batch_size` (50), `pipeline.retries` (5), `pipeline.outbox` / `pipeline_log`, and `pipeline.watch_dirs`. Runner mapping: Harbor `--model openrouter/<TEACHER_MODEL>`; SWE-smith and R2E-Gym `llm_name = TEACHER_MODEL`. Runners list the missing variable and stop.

## Data format

One Parquet row = one attempt. Columns from `pipeline/convert.py:COLUMNS`: `run_id`, `task_id`, `source` (`tbench`/`swesmith`/`r2e`), `repo`, `commit`, `docker_image`, `teacher_model`, `harness`, `harness_version`, `attempt`, `messages` (JSON string of `{role, content, tool_calls, tool_call_id}`), `reward`, `tests_passed`, `tests_failed`, `patch`, `n_steps`, `n_input_tokens`, `n_output_tokens`, `cost_usd`, `wall_time_s`, `error`, `started_at`, `finished_at`, `raw_log_path`. Bucket paths (see `pipeline/upload.py:bucket_path`):

```
parquet/source=<source>/teacher=<model>/date=<YYYY-MM-DD>/<run_id>-<attempt>.parquet
raw/parquet/source=<source>/teacher=<model>/date=<YYYY-MM-DD>/<run_id>-<attempt>.tar.zst
```

Redaction is covered by `make test` (`test_make_row_redacts_and_passes_assert`). The native log is uploaded compressed next to the Parquet.

## Running at scale

Manual steps (not run here):

```bash
PYTHONPATH=. python -m pipeline.validate --config config.yaml
bash runners/tbench.sh
bash runners/swesmith.sh
bash runners/r2e.sh
PYTHONPATH=. python -m pipeline.watcher --config config.yaml --once
hf buckets info "$HF_BUCKET" && hf buckets list "$HF_BUCKET" -R
python -c "import pandas as pd; print(pd.read_parquet('hf://buckets/$HF_BUCKET/parquet/...').head())"
```

Each billed OpenRouter call adds cost; each harness image adds disk. The watcher retries uploads with backoff, verifies size in the bucket, then moves files to `outbox/uploaded/` or `outbox/failed/` and logs JSONL to `pipeline.log`. Re-run skip was tested offline with mocked upload (`test_upload_idempotent_skip`); live dedup is untested. Names are deterministic (`<run_id>-<attempt>.parquet`), existence is checked with `get_bucket_paths_info`, and writes are staged via `.tmp`.

## Development

`make install` (uv sync), `make lint` (ruff plus non-ASCII grep over `*.md`), `make test` (22 offline tests), `make validate`. Run link check (lychee or markdown-link-check) on a networked machine before publishing; mark first-push badge targets as `pending first push`. Add a sanitized real log fixture per harness parser.

## License

MIT. See `LICENSE`.
