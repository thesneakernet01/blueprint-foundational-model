# data/ — the Lakehouse layer

This accelerator's Lakehouse is **VAST object storage + Impala**: transaction splits are
written as Parquet by the data-preparation job and exposed as Impala external tables.
Nothing is stored in this folder — it documents the layer.

| What | Where | Automated by |
|------|-------|--------------|
| Transaction splits (Parquet) | VAST/S3 buckets | `pipelines/prepare_data.py` |
| External tables + DDL | Impala | `pipelines/prepare_data.py` (config in [`../docs/impala-vast-s3-config.md`](../docs/impala-vast-s3-config.md)) |
| Model checkpoint | `models/` at the repo root (gitignored; CWD-relative contract, [ADR-001](../docs/architecture/ADR-001-root-packages.md)) | `pipelines/fetch_model.py` |
| Local scratch during preparation | managed by `pipelines/prepare_data.py` | — |

## Conventions

- VAST credentials only in the untracked `.vast.env`.
- Re-preparation is idempotent and also triggerable from the UI (`tfm_demo/jobs.py`).
