# ai/ — the AI layer

This folder documents the model/inference layer of the blueprint; the implementation
itself lives in the root packages, not here (see
[ADR-001](../docs/architecture/ADR-001-root-packages.md) for why).

## Pipeline

A transaction is scored through five stages:

1. **Tokenizer** — GPU-side (RAPIDS/cuDF) feature tokenization of the raw transaction.
2. **TFM decoder** — NVIDIA's Transaction Foundation Model produces a 512-dimension
   last-token embedding.
3. **PCA-64** — the embedding is reduced to 64 dimensions.
4. **Three XGBoost heads** — trained in parallel on raw features, on embeddings, and on
   both combined, so the lift from the foundation model is a measured number.
5. **Optional NEXUS LTM head** — an external long-term-memory endpoint, off by default.

## Where each piece lives

| Component | Code | Notes |
|-----------|------|-------|
| Tokenizer + decoder inference (blueprint) | `src/` | Staged by `pipelines/fetch_model.py`, not hand-written here |
| Inference engine | `tfm_demo/engine.py` | Runs in REAL mode (GPU + checkpoint present) or DEMO-FALLBACK |
| FastAPI app factory + `/api` routes | `tfm_demo/app.py` | Entrypoint is the root `app.py` |
| NEXUS LTM integration | `tfm_demo/nexus.py` | See [design doc](../docs/nexus-ltm-design.md) |
| Run history + progressive training budget | `tfm_demo/runs.py` | Backs the lifecycle dashboard |
| Model Registry + CML Model deploy | `tfm_demo/registry.py` | Falls back to `tfm_demo/registry_predict.py` when registry deploy isn't available |
| Checkpoint | `models/` | Root-level, gitignored — fetched, not committed |

### Notebook

`notebooks/tfm_training_walkthrough.ipynb` walks a data scientist through the full
pipeline end to end — data → embeddings → PCA → heads → registry — as an executable
notebook rather than a slide deck. Off-GPU, it degrades honestly to a labelled synthetic
sample instead of failing or faking real numbers.

`inference/` *(reserved — add here if a standalone inference doc/script is needed)*.

## Conventions to keep in mind

- The model card lives at
  [`../governance/model-cards/tfm_fraud_heads.md`](../governance/model-cards/tfm_fraud_heads.md)
  — update it alongside any change to what the heads are trained on or how they're
  evaluated.
- **DEMO-FALLBACK must always keep working** without a GPU or checkpoint present — it's
  the de-risked path the demo relies on when hardware isn't available, so don't let it
  bit-rot silently.
