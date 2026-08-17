# tests/ — data quality · AI eval

`unit/` is the automated suite — run it in Docker with `docker compose run --rm test`
(pytest against `tfm_demo/runs.py` and the off-CML registry degradation).

## What verifies this accelerator today

| Check | Where | What it asserts |
|-------|-------|-----------------|
| Connectivity probes | `infra/vast_probe.py` · `infra/nexus_probe.py` | VAST S3 and NEXUS endpoints answer. |
| Engine round trip | manual (UI or `/api`) | A raw transaction tokenizes → embeds → scores in REAL mode; DEMO-FALLBACK works without the checkpoint. |
| Data path | `pipelines/prepare_data.py` output | Impala external tables readable (see `docs/impala-vast-s3-config.md`). |

## What goes here (as suites are added)

| Path | Contents |
|------|----------|
| `unit/` | Budget-schedule + run-history invariants; `/api/registry` off-CML behavior. |
| `data-quality/` *(add as needed)* | Split-schema and row-count assertions on the prepared Parquet. |
| `ai-eval/` *(add as needed)* | Golden transactions with expected score ranges per head; lift-regression bounds. |
