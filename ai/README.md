# ai/ — the AI layer

The inference stack: GPU tokenizer + **Transaction Foundation Model decoder** → 512-d
embeddings → PCA-64 → three **XGBoost heads** (raw / embeddings / combined), plus an
optional **NEXUS LTM** head. The implementation lives in the root packages (see
[ADR-001](../docs/architecture/ADR-001-root-packages.md)); this folder documents the layer.

## Where the AI layer actually lives

| Component | Code |
|-----------|------|
| Tokenizer + decoder inference (blueprint) | `src/` (staged by `pipelines/fetch_model.py`) |
| Inference engine (REAL + DEMO-FALLBACK modes) | `tfm_demo/engine.py` |
| FastAPI app factory + /api routes | `tfm_demo/app.py` (entrypoint: root `app.py`) |
| NEXUS LTM integration | `tfm_demo/nexus.py` ([design](../docs/nexus-ltm-design.md)) |
| Checkpoint | `models/` (root, gitignored) |

`notebooks/` and `inference/` *(add as needed)*.

## Conventions

- The model card is
  [`../governance/model-cards/tfm_fraud_heads.md`](../governance/model-cards/tfm_fraud_heads.md).
- **DEMO-FALLBACK must keep working** without GPU/checkpoint — it is the de-risked demo
  path.
