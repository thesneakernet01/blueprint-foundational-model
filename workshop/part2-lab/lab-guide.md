# Hands-On Lab: Live Scoring, GPU Retraining & Endpoint Deployment (60 min)

You are working with a live fraud-inference application running **natively on Cloudera
AI**, with the full NVIDIA GPU stack behind it: cuDF tokenizer → TFM decoder (512-d
embeddings) → PCA-64 → three XGBoost heads → cuML UMAP.

**Your instance URL:** ______________________ (provided by the instructor)

| Exercise | Time | You will |
| --- | --- | --- |
| 1. Score live transactions | 12 min | Run the real GPU inference path and read the three heads |
| 2. Trigger a GPU training run | 18 min | Launch a budgeted retrain, watch the pipeline + GPU, see hot reload |
| 3. Analyze diagnostics & clusters | 12 min | Read validation curves, separation, importance, and the lift trend |
| 4. Register & deploy an endpoint | 15 min | Push to the Cloudera AI Model Registry, deploy + `curl` a REST endpoint |
| 5. Stretch goals | spare time | Data dialog, NEXUS 4th head, raw API |

> ✅ **Checkpoint** boxes mark what you should see before moving on. If you don't, flag
> the instructor rather than debugging solo — lab time is short.

---

## Exercise 1 — Score live transactions (12 min)

### 1.1 Verify you're on the real GPU path

Open your instance URL. In the header, find the status badge.

> ✅ Badge reads **`live · GPU`** with a lit CUDA dot. If it says `demo fallback`, tell
> the instructor — your scores would be synthetic.

### 1.2 Score a known fraud

On the **Inference** tab:

1. In the transaction composer, load the example **"Real fraud (test set)"** — a real
   row from the held-out test split.
2. Click score, and read the three heads:
   - **classic ml** (raw features only — the legacy baseline)
   - **foundation-fed xgboost** (combined: raw ⊕ embedding — the production candidate)
   - **foundation model** (embedding only)
3. Note the token strip above the scores: those are the actual sequence tokens the GPU
   tokenizer produced from your fields (max 128, vocab 6,251).

> ✅ At least one embedding-fed head flags the transaction (score ≥ 0.5 → `FLAG ·
> REVIEW`). Often the raw head is *less* confident than the embedding heads — that gap
> is the paradigm shift, rendered as two bars.

### 1.3 Watch it land on the embedding map

Below the scores, the **embedding map** shows a 2-D cuML UMAP projection of up to 1,500
held-out transactions (grey = legit, red = fraud). Your scored transaction appears as an
**orange live dot**; the dashed ring is where the batch pipeline placed this exact row
during training.

> ✅ For the loaded test-set example, the live dot sits on or near its dashed ring —
> proof the live single-transaction path (tokenizer → decoder → PCA → UMAP transform)
> reproduces the batch pipeline.

### 1.4 Perturb the transaction

Now break it. Starting from **"Real legitimate (test set)"**, change one field at a
time and re-score:

- Set **Channel** to `Online`
- Move **Time** to 03:00
- Multiply **Amount** by 10
- Change **Merchant State** to somewhere implausible

Watch which heads move, by how much, and where the dot travels on the map.

**Question for the room:** which single change moved the *embedding* head most? Why
might sequence pretraining make it sensitive to that field when a rule engine isn't?

---

## Exercise 2 — Trigger a live GPU training run (18 min)

### 2.1 Check your training budget

Switch to the **Model Lifecycle** tab. The **Train next run** button shows the budget
the next run has earned, e.g. `4,000 rows · 35% rounds`.

The budget ladder (each completed run advances one tier):

| Run | Rows embedded / split | Boosting rounds |
| --- | --- | --- |
| 1 | 4,000 | 35% |
| 2 | 8,000 | 55% |
| 3 | 12,000 | 75% |
| 4 | 16,000 | 90% |
| 5+ | 20,000 | 100% |

### 2.2 Launch and observe

Click **Train next run**. In the build dialog, follow along:

1. **Stage chips** light in order: `check → embed → pca → train → artifacts`, then
   `reload → done`. Each chip is wired to real pipeline events, not an animation.
2. The **streaming log** narrates: split reads from your governed storage backend
   (Impala / S3) straight into cuDF, embedding batches of 512 through the TFM decoder,
   PCA fit, three XGBoost heads training with `device="cuda"` and early stopping.
3. The **resource meters** show live `nvidia-smi` GPU utilization and VRAM alongside
   CPU/memory. Watch VRAM during `embed` and GPU utilization spike during `train`.

While it runs (~3–6 min at tier 1), find in the log: the GPU stack version line
(cuDF/RMM/torch), and the per-split row counts read from storage.

> ✅ Run completes: state `done`, the dialog shows per-head test **ROC-AUC** and
> **Average Precision** plus the **lift %** over the raw baseline.

### 2.3 Prove the hot reload

The final `reload` stage re-loaded the new artifacts into the *running* engine — no
restart, no redeploy.

1. Go back to **Inference**. The metrics strip now shows your run's numbers.
2. Re-score the same fraud example from Exercise 1. The scores are now produced by the
   model **you** just trained; the UMAP background is your run's projection.

> ✅ Metrics strip and map changed without the app restarting. This is the
> zero-disruption artifact-refresh story from Module 3.

### 2.4 Run tier 2 (start it now, analyze while it runs)

Click **Train next run** again — note the budget doubled (8,000 rows · 55% rounds).
Let it run in the background while you do Exercise 3.

---

## Exercise 3 — Analyze diagnostics & embedding clusters (12 min)

Stay on **Model Lifecycle**. Every run recorded its diagnostics; explore your run 1
while run 2 trains.

### 3.1 Validation curve

The eval-curve panel plots per-boosting-round validation AUC with the early-stopping
**best round** marked.

**Read it:** did the model stop early (curve flattened) or hit the round cap (still
climbing)? At 35% rounds, tier 1 usually leaves quality on the table — that's the
point of the ladder.

### 3.2 Class separation

The separation panel is a histogram of the **combined** head's test scores, split into
legit vs fraud distributions.

**Read it:** the further apart the two masses, the cleaner an operating threshold you
can set. Remember it for comparison after run 2.

### 3.3 Feature importance — the money chart

The importance panel shows the combined head's top-15 features, each tagged **raw** or
**embedding** (`pca_0..pca_63`).

**Question for the room:** how many of the top 15 are embedding components? This chart
is the customer-facing answer to "do the embeddings actually add signal, or is the
model just using amount and MCC?"

### 3.4 Run-over-run lift

When run 2 finishes (hot-reloads automatically):

1. **Metric trend** — toggle ROC-AUC / Avg precision; run 2 should sit above run 1.
2. **Budget chart** — rows-embedded bars, boosting-rounds line, and run duration: the
   compute-vs-quality trade, charted.
3. **Runs table** — the audit trail: run id, finish time, rows/split, rounds, AUC, AP,
   duration. This history persists in `.runs_history.json`, outside the artifact
   directory, so it survives every retrain.

> ✅ You can point at the run history and say: "budget went up, AP went up, and every
> claim has a recorded run behind it."

---

## Exercise 4 — Register & deploy a governed endpoint (15 min)

> Registry actions need the app running inside CML (workload API key). `GET
> /api/registry` explains itself if anything is missing.

### 4.1 Register your best run

In the **Registry panel**, click **Register**. Behind the button
(`tfm_demo/registry.py`):

- an MLflow run logs your budget params and per-head AUC/AP metrics to experiment
  `tfm-fraud-demo`;
- the artifact bundle — **combined XGBoost head + preprocessor + PCA** — is registered
  as **`tfm-fraud-combined`** in the **Cloudera AI Model Registry**.

> ✅ A version number appears in the panel, and a **version marker** appears on the
> metric-trend chart pinned to the run it came from — lineage you can see.

### 4.2 Deploy it as a REST endpoint

Click **Deploy**. The app creates (or reuses) a CML Model, builds it from the
registered version, and deploys it at 2 CPU / 4 GB — this is a **CPU** deployment,
because the registered scope is the classical head bundle; the GPU embedding stage
stays upstream. Build + deploy takes ~5–10 minutes; the panel polls status.

### 4.3 Score the governed endpoint

When it shows `deployed`, grab the endpoint URL + access key from the panel (or the CML
Models page) and score it from a terminal. The contract: rows carry the raw feature
columns **plus** `pca_0..pca_63`, and the response is fraud probabilities:

```bash
curl -s -X POST "<ENDPOINT_URL>" \
  -H 'Content-Type: application/json' \
  -d '{"request": {"rows": [{"Amount": "$132.40", "Use Chip": "Online",
        "Merchant City": "Rome", "Merchant State": "Italy", "MCC": 5812,
        "...": "...", "pca_0": 0.113, "pca_1": -0.042, "...": "...",
        "pca_63": 0.007}]}}'
# → {"probability": [0.87]}
```

(Fastest way to get a valid row: ask the instructor for the prepared sample, or pull
one from `demo_artifacts/examples.json` in your project.)

> ✅ You have gone from "click train" to a **governed, versioned REST endpoint with
> registry lineage** without leaving the platform. That is the full Cloudera AI story.

---

## Exercise 5 — Stretch goals (spare time)

- **Governed data plumbing:** open the **Data** dialog. Inspect the backend config
  (Impala connection/database, or S3 endpoint/bucket/prefix), hit **Save & test
  connection**, and find the three split tables (`train`, `val_eval`, `test_eval`) in
  your warehouse or bucket. Note the secret is write-only (`(unchanged)` placeholder).
- **Fourth head — external foundation model:** in the build dialog, switch **NEXUS** to
  `stub` and retrain: a fourth card (Fundamental's Large Tabular Model, long-term-memory
  head) joins the benchmark with clearly-labelled canned metrics. Design:
  `docs/nexus-ltm-design.md`.
- **Raw API:** the SPA is just a client. From a terminal in your project:
  `curl -s localhost:7100/api/status`, `/api/summary`, `/api/runs`, and
  `POST /api/score` with a transaction JSON. Everything you clicked is a governed API
  call you could script.
- **Keep climbing the ladder:** run tiers 3–5 and watch the metric trend — where does
  quality saturate relative to compute spent? That curve is a customer conversation.

---

## Wrap-up: what you can now say to a customer

1. "Foundation-model embeddings gave a **measured double-digit AP lift** over
   hand-crafted features — here's the run history that proves it."
2. "The whole loop — governed lakehouse data, GPU training, hot-reloaded scoring,
   registry, governed endpoint — ran on **Cloudera AI with one NVIDIA L4**."
3. "Swap the dataset and tokenizer, and this same harness benchmarks AML, credit risk,
   payments abuse, or churn."
