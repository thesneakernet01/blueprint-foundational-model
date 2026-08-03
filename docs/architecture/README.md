# docs/architecture/ — reference architecture & ADRs

## Architecture

This accelerator realizes the **build standard** (§5 of the operating model):

```
Ingest (pipelines/prepare_data.py → VAST/S3 Parquet + Impala DDL) →
Lakehouse (VAST object storage · Impala external tables) →
Process (AMP job chain: fetch model → prepare data) →
AI (GPU tokenizer → TFM decoder → embeddings → PCA-64 → XGBoost heads [+ NEXUS LTM]) →
Serve (FastAPI + React cockpit on one GPU container)
```

Deep-dives: [`../impala-vast-s3-config.md`](../impala-vast-s3-config.md) and
[`../nexus-ltm-design.md`](../nexus-ltm-design.md).

- ADRs record any deviation from the standard stack:
  - **[ADR-001](./ADR-001-root-packages.md)** — `src/`, `tfm_demo/`, `app.py`, and
    `models/` stay at the repo root (import + CWD contracts).
