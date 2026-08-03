# Model card — TFM Fraud Inference Ensemble

- **Accelerator:** `cloudera-forge-fsi-foundational-model`
- **Serving:** in-process on one GPU container (`tfm_demo/engine.py`), REAL mode with the
  fetched checkpoint or DEMO-FALLBACK without it
- **Components:** NVIDIA **Transaction Foundation Model** decoder (fetched by
  `pipelines/fetch_model.py`) → 512-d embeddings → PCA-64 → three **XGBoost** heads
  (raw features / embeddings / combined) · optional **NEXUS LTM** head
- **NIM model ID:** _n/a (checkpoint-based)_ — pin the TFM checkpoint version here.

## Intended use

Demonstrating **embedding lift** for card-fraud scoring: the combined head's AUC/AP gain
over the raw-features head, live per transaction. Demo/accelerator use on public/synthetic
transaction data — not a production fraud decision system.

## Training / source data

- TFM: pretrained by NVIDIA (consumed as a checkpoint).
- Heads: trained in-app on the prepared transaction splits
  (`pipelines/prepare_data.py` → VAST/Impala).
- PCA-64 fitted on the embedding set; artifacts under `models/` (gitignored).

## Evaluation

- Live AUC / average-precision per head in the UI; UMAP for qualitative separation.
- DEMO-FALLBACK mode substitutes deterministic outputs — clearly not a model evaluation.

## Limitations & risks

- Lift measured on demo data — not evidence of production lift on an issuer's portfolio.
- The GPU path requires the pinned runtime; checkpoint license/redistribution terms apply.
- NEXUS head is optional/experimental ([design](../../docs/nexus-ltm-design.md)).

## Owner

- Forge Engineering · <team-contact>
