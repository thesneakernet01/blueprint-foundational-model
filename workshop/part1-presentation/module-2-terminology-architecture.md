# Module 2 — Core Terminology & Architecture (15 min)

**Goal:** attendees can whiteboard the inference path — *raw transaction → GPU tokenizer
→ TFM decoder → 512-d embedding → PCA-64 → three XGBoost heads → fraud probabilities* —
and defend the multi-head benchmarking design to a skeptical customer data scientist.

---

## Slide 2.1 — Tokenization & 512-D embeddings (4 min)

**On the slide**

```
Raw transaction (14 TabFormer fields)
   Amount · Merchant Name · Merchant City/State · ZIP · MCC
   Year/Month/Day/Time · Use Chip (channel) · User · Card
        │  FinancialTabularTokenizer  (GPU, cuDF)
        ▼
Token sequence (max_length = 128, vocab = 6,251, merchant hash size = 2,000)
        │  TFM decoder (8-layer LLaMA-style, hidden_size = 512)
        ▼
512-dimensional last-token embedding  ← the "behavioral fingerprint"
```

**Speaker notes**

- Walk one field through: `Amount "$132.40"` is bucketed and tokenized; `Merchant Name`
  is hashed into 2,000 bins (cardinality control); time fields become temporal tokens.
  The tokenizer runs on **cuDF dataframes on the GPU** — no CPU detour.
- "Last-token pooling": like asking a language model "having read this whole sequence,
  what's your state now?" — the final hidden state (512 numbers) *is* the embedding.
- Code anchors: tokenizer settings in `tfm_demo/config.py` (`MAX_LENGTH = 128`,
  `MERCHANT_HASH_SIZE = 2000`); single-transaction path in
  `tfm_demo/engine.py::_embed_one`; the fetched NVIDIA blueprint package lives in
  `src/`.

---

## Slide 2.2 — Dimensionality reduction: PCA 512 → 64 (3 min)

**On the slide**

- 512-d embeddings are compressed to **64 principal components** before hitting XGBoost.
- Why: downstream efficiency and generalization —
  - 8× fewer columns → faster tree training and smaller registered artifacts
  - decorrelated components suit tree splits better than raw correlated dims
  - the 64-d vector (`pca_0 … pca_63`) is the **contract** for the deployed endpoint
- Fit on the training split only; val/test are *transformed* — no leakage.

**Speaker notes**

- Keep it honest: PCA here is sklearn (`PCA(n_components=64, random_state=42)`), fit in
  seconds because the input is only tens of thousands of 512-d rows — the GPU budget is
  spent where it pays (embedding generation and XGBoost). RAPIDS cuML is used where it
  matters most visually: UMAP (Module 3).
- Foreshadow MLOps: when we register the model later, the registered bundle is
  *preprocessor + PCA + combined XGBoost head* — the endpoint expects `pca_0..pca_63`
  columns. Say it now so the lab's endpoint contract isn't a surprise.
- Code anchor: `tfm_demo/export.py` (PCA fit/transform), `PCA_DIM = 64` in
  `tfm_demo/config.py`.

---

## Slide 2.3 — The multi-head benchmarking strategy (5 min)

**On the slide**

Three XGBoost heads trained in parallel on identical splits:

| Head | Input | Role |
| --- | --- | --- |
| **Raw** | 13 hand-crafted transaction features (ordinal-encoded) | Legacy baseline — "your current model" |
| **Embed** | PCA-64 foundation-model embedding only | Pure representation-power test |
| **Combined** | raw features ⊕ PCA-64 (hstack) | Production candidate — best of both |

- All heads: `tree_method="hist"`, `device="cuda"`, early stopping (20 rounds) on a
  held-out validation split, `eval_metric="auc"`.
- Hyperparameters are fixed per head (tuned once in the blueprint notebooks, copied
  verbatim) — the comparison varies **only the representation**.

**Speaker notes**

- This is the credibility slide for data-science audiences. The design answers the
  obvious objection ("you just tuned the embedding model harder") — no: same splits,
  same training procedure, same metric, hyperparameters frozen per head. The only
  variable is what the trees get to see.
- The **combined** head is the realistic adoption path: nobody throws away their
  features on day one — you *add* the embedding columns to the existing model and
  measure the delta. That is exactly what this demo operationalizes.
- Optional 4th head tease: the app scaffolds a **NEXUS long-term-memory head**
  (Fundamental's Large Tabular Model as an external endpoint) as a fourth card — a
  pattern for composing external foundation models into the same benchmark. It ships in
  `off`/`stub` modes today (`docs/nexus-ltm-design.md`).
- Code anchor: `tfm_demo/export.py` — `XGB_PARAMS_RAW / _EMBED / _COMBINED`.

---

## Slide 2.4 — Quantifying the lift (3 min)

**On the slide**

- Metrics per head on the **held-out temporal test split**: **ROC-AUC** and
  **Average Precision (AP)**.
- The UI leads with AP — the right headline for ~1% fraud prevalence.
- Lift is computed **relative to the raw baseline** and shown as a percentage:
  `embed_ap_pct`, `combined_ap_pct` (and the AUC equivalents).
- Expected story: **double-digit AP lift** for the embedding-fed heads over hand-crafted
  features — recomputed live every time you retrain.

**Speaker notes**

- Explain "temporal test split" precisely (it matters to FSI buyers): train on the first
  80% of the timeline, validate on the next 10%, test on the final 10% — the model is
  always evaluated on *the future*, which is how fraud models actually live.
- Translate the metric for business stakeholders: at a fixed review budget, higher AP
  means more real fraud per 100 analyst reviews — fewer false positives annoying good
  customers, more fraud dollars caught.
- Everything on this slide is on screen in the lab: the `MetricsStrip` headline numbers
  and "Foundation-model lift" chip, and per-run history in the Model Lifecycle tab.
- Transition: "Numbers need a lifecycle: where does the data live, what runs on the GPU,
  how does a retrain reach production? Module 3."
