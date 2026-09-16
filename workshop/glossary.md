# Glossary — Transaction Foundation Models on Cloudera AI (handout)

**Transaction Foundation Model (TFM)** — NVIDIA's decoder-only transformer (LLaMA-style,
8 layers, hidden size 512, ~56 MB) pretrained on payment-transaction sequences. Reads
transactions as tokens the way a language model reads words.

**TabFormer** — IBM's public synthetic credit-card dataset used here: ~24 million
transactions across 20,000 simulated cardholders (~2.4 GB), with fraud labels.

**Tokenization** — mapping raw transaction fields (amount, merchant, MCC, city, time,
channel…) into a 6,251-token vocabulary; sequences up to 128 tokens; merchant names
hashed into 2,000 bins. Runs on GPU via RAPIDS cuDF.

**512-D embedding** — the decoder's last-token hidden state: a 512-number "behavioral
fingerprint" of the transaction in its sequence context.

**PCA-64** — principal-component compression of the 512-d embedding to 64 features
(`pca_0..pca_63`); fit on train only. Defines the deployed endpoint's input contract.

**Multi-head benchmark** — three XGBoost heads trained on identical temporal splits with
frozen hyperparameters: **raw** (hand-crafted features = legacy baseline), **embed**
(PCA-64 only), **combined** (raw ⊕ PCA-64 = production candidate).

**Temporal split** — train on the first 80% of the timeline, validate on the next 10%,
test on the final 10% — the model is always evaluated on the future.

**ROC-AUC / Average Precision (AP)** — the two quality metrics reported per head on the
held-out test split. AP is the headline: with ~1% fraud prevalence it tracks the
precision an analyst review queue actually experiences.

**Lift** — percentage improvement of the embed/combined heads over the raw baseline
(e.g. `combined_ap_pct`); the demo's central, live-recomputed number.

**Training budget ladder** — successive runs earn bigger budgets: 4K → 8K → 12K → 16K →
20K embedded rows per split at 35→55→75→90→100% boosting rounds, making the
compute-vs-quality trade visible run over run.

**Hot reload** — on run completion the in-process engine reloads the new artifacts; the
next scored transaction uses the new model with zero service disruption.

**UMAP embedding map** — cuML-fitted 2-D projection of held-out test embeddings; live
transactions appear as a dot placed by the same projection, next to a dashed ring
showing the batch pipeline's position for that row (the consistency check).

**RAPIDS** — NVIDIA's GPU data-science stack used throughout: **cuDF** (dataframes /
tokenizer input), **cuML** (UMAP), **RMM** (shared GPU memory pool), plus
XGBoost-on-CUDA and PyTorch for the decoder. Sized to fit one 24 GB NVIDIA L4.

**AMP (Applied ML Prototype)** — Cloudera AI's one-click deployment: this app's
`.project-metadata.yaml` declares five tasks (deps → SPA build → checkpoint fetch →
data prep → application) on an NVIDIA GPU runtime.

**Cloudera AI Model Registry** — platform registry where the run is registered as
`tfm-fraud-combined` (MLflow experiment `tfm-fraud-demo` carries params/metrics
lineage), then deployed as a governed CML Model REST endpoint (2 CPU / 4 GB).

**Registered scope** — the deployable bundle: preprocessor + PCA + combined XGBoost
head. It runs on CPU and expects rows carrying raw features plus `pca_0..pca_63`; the
GPU embedding stage runs upstream and scales independently.

**DEMO-FALLBACK mode** — the app's clearly-labelled degraded mode (no GPU / checkpoint /
artifacts): the full UI works with synthetic scores, so the demo never blanks.

**NEXUS LTM head** — optional fourth benchmark card calling an external foundation
model (Fundamental's Large Tabular Model) on raw features over a governed endpoint;
ships in `off`/`stub` modes (design: `docs/nexus-ltm-design.md`).

**Run history / audit trail** — `.runs_history.json`: per-run id, timestamps, duration,
budget, per-head AUC/AP, lift, diagnostics, and registry status — the recorded evidence
behind every performance claim.
