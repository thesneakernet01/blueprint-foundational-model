# Worked example — FSI / transaction foundation model

This repository is a **complete, running accelerator**:
`cloudera-forge-fsi-foundational-model` — a live fraud-inference cockpit proving the
embedding lift of NVIDIA's Transaction Foundation Model over raw features.

## The data flow

```
        docs/business-case/scorecard.md   (greenfield, 🟢 greenlight)
                       │
   pipelines/fetch_model.py      TFM checkpoint → models/ · blueprint src/ staged
   pipelines/prepare_data.py     transaction splits → VAST/S3 Parquet + Impala DDL
                       │
                       ▼
        raw transaction (UI or API)
                       │
                       ▼
   src/tokenizer → TFM decoder (GPU) → 512-d embedding → PCA-64
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
    XGBoost: raw   XGBoost: embed   XGBoost: combined    (+ optional NEXUS LTM head)
          └────────────┴───────┬────────┘
                               ▼
   tfm_demo/engine.py — live AUC/AP lift · UMAP · scoring
                               │
                               ▼
   app/serve_app.py — uvicorn (127.0.0.1:$BACKEND_PORT) behind
   Vite preview on $CDSW_APP_PORT → app/frontend (React SPA)
```

## Layer by layer

| Layer | Where | What it does |
|-------|-------|--------------|
| Business case | [`docs/business-case/scorecard.md`](./business-case/scorecard.md) | Type = **greenfield**; KPI = fraud detection lift (AUC/AP) |
| Ingest | [`pipelines/prepare_data.py`](../pipelines/prepare_data.py) | Splits → VAST/S3 + Impala |
| Lakehouse | VAST + Impala | External Parquet tables ([config](./impala-vast-s3-config.md)) |
| Process | [`pipelines/`](../pipelines/) | AMP job chain |
| AI | `src/` + `tfm_demo/` (root, [ADR-001](./architecture/ADR-001-root-packages.md)) | Tokenize → embed → score |
| Serve | [`app/`](../app/) | The cockpit Application |
| Governance | [`governance/`](../governance/) | TFM + heads model card |
| Ship | [`.cicd/`](../.cicd/) | build → test → deploy |

## Try it

```bash
python deploy/install_deps.py && python deploy/build_frontend.py
python pipelines/fetch_model.py && python pipelines/prepare_data.py
python app/serve_app.py
# No GPU/checkpoint? The engine's DEMO-FALLBACK mode still drives the UI.
```

## Make it yours

```bash
cd ../demo-design-template && make new VERTICAL=<vertical> USECASE=<usecase>
```
