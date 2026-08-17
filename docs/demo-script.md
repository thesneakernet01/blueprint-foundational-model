# Demo Script — Transaction Foundation Model on Cloudera AI Workbench

A ~10-minute presenter talk-track. Each beat has **Say** (the words), **Do**
(the clicks), and **Under the hood** (where in this repo the thing actually
happens — useful when a technical viewer asks "is that real?").

Pre-demo checklist is in the appendix.

---

## 1 · Cold open — the inference cockpit (~1 min)

**Say:** "This is a live fraud-detection application running on Cloudera AI
Workbench. The model behind it is a decoder-style *transaction foundation
model* — the same idea as a language model, but its vocabulary is card
transactions. It was pre-trained on the TabFormer dataset: 24 million
transactions from 20,000 simulated cardholders."

Point at the status badge: "This badge is honest — REAL means every score you
see comes from the actual model on the GPU behind this page. If the heavy
stack were missing it would say DEMO-FALLBACK, clearly labelled."

**Do:** Point out the header badge (REAL · GPU) and the three metric cards.

**Under the hood:** `tfm_demo/engine.py::warmup` loads the exported artifacts
and decides REAL vs DEMO-FALLBACK; the cards read `demo_artifacts/summary.json`
via `GET /api/summary`.

## 2 · Score a live transaction (~2 min)

**Say:** "Let's score a real transaction from the held-out test set — one the
model has never trained on. Watch what happens: the raw fields are tokenized,
the foundation model turns them into a 512-dimensional embedding, PCA
compresses that to 64 dimensions, and three XGBoost heads score it in
parallel — one on hand-crafted features only, one on embeddings only, one on
both."

"And on the right, the map: that's the embedding space itself, projected to
2-D. Fraud clusters. The orange dot is *this* transaction landing in that
space, live."

**Do:** Load "Real fraud (test set)" in the composer → Run inference → walk
the three (or four) score bars → point at the live dot on the embedding map.

**Under the hood:** `tfm_demo/engine.py::score` runs tokenize → embed → PCA →
heads; the map background and the UMAP projector come from the export
(`tfm_demo/export.py`, UMAP block). The hollow ring is a built-in diagnostic:
where the batch pipeline embedded this exact row — live dot on the ring means
the live path agrees with the training path.

## 3 · The paradigm story (~1 min)

**Say:** "The point of the three cards: same XGBoost, same data — the only
change is the representation. Foundation-model embeddings lift average
precision by double digits over hand-crafted features. That lift number is
computed from the real test set at every training run, not a slide."

**Do:** Open **Compare paradigms** from the metrics strip; close it.

**Under the hood:** the lift block is computed at the end of
`tfm_demo/export.py::run_export` and stored in `summary.json`.

## 4 · Train it, live — the Model Lifecycle dashboard (~2 min)

**Say:** "Now the part demos usually fake: let's retrain the model, here, on
the cluster. This tab is the model's lifecycle — and this pipeline diagram is
not an animation loop, it's wired to the actual training job."

**Do:** Switch to the **Model Lifecycle** tab → **Train next run** → launch
the build → narrate the pipeline stages as they light up:

1. *Load splits* — temporal train/val/test read from governed storage
   (Impala tables or S3 objects — `tfm_demo/storage.py`).
2. *FM embeddings* — the foundation model embeds each transaction
   (`tfm_demo/export.py::run_export`, notebook-04 step, cached per budget).
3. *PCA 512→64*, *Train XGBoost heads* — the notebook-05 step, on-GPU.
4. *UMAP + artifacts* — the map, examples, and model bundle are written.
5. *Hot reload* — `tfm_demo/jobs.py::ExportManager` re-warms the live engine.
   No restart, no redeploy of the app: the next scored transaction uses the
   new model.

**Under the hood:** the stage chips follow real stage events emitted by
`run_export(on_stage=...)`; the log streams from the same job.

## 5 · Improvement over time (~2 min)

**Say:** "Here's the operational story. Every training run is granted a bigger
budget than the last — more transactions embedded, more boosting rounds. So
run over run, you watch the same pipeline get better, and the chart is the
audit trail: every point is a real test-set number from a recorded run."

"More data plus more compute equals better fraud detection — and because data,
GPUs, and the app live on one platform, that loop is a button, not a ticket."

**Do:** Walk the **Model quality per training run** chart (toggle AUC/AP),
show the **Training budget per run** bars beneath it, and the run-history
table. If replaying for a new audience: **Reset demo** starts the curve over
(cached embeddings make replays fast).

**Under the hood:** the budget schedule and run history live in
`tfm_demo/runs.py` (`.runs_history.json`); each export appends its record at
the end of `run_export`.

## 6 · Register and deploy (~1.5 min)

**Say:** "A model that only lives in a notebook is a liability. One click
registers this run's model — the trained head with its preprocessing — as a
new version in the Cloudera Model Registry, with its metrics and training
budget logged. A second click deploys that registered version as a governed
REST endpoint with its own resources and auth. Watch the version marker land
on the trend chart — that's the lineage: which training run, which budget,
which metrics, which endpoint."

**Do:** In the registry panel: **Register latest run** → version appears in
the panel and as a marker on the trend chart → **Deploy vN** → endpoint status
goes live on the lifecycle strip. (Off-CML the panel explains itself and stays
disabled — nothing here is smoke and mirrors.)

**Under the hood:** `tfm_demo/registry.py` — MLflow-backed registration into
the workspace registry, cmlapi for the Model build + deployment;
`tfm_demo/registry_predict.py` is the fallback scoring entry point.

## 7 · Close — why Cloudera AI Workbench (~1 min, no competitor names)

**Say:** "Step back and count the platforms you just saw: zero besides this
one.

- **The data never moved.** Training splits sit in governed storage with the
  same access controls as the rest of the business's data; the model came to
  the data, not the other way around.
- **GPU training, jobs, the registry, the endpoint, and this application are
  one platform** — the lifecycle you just watched (develop → train → register
  → deploy → observe) is the platform's native shape, not glue code between
  point tools.
- **It's reproducible by manifest.** This entire demo — dependencies, data
  prep, model fetch, the app itself — deploys from one project manifest
  (`.project-metadata.yaml`) as an accelerator: one click, same result, any
  workspace.
- **Sovereignty by default.** Model weights, embeddings, and transaction data
  never leave the environment — the inference you saw ran on GPUs the
  business owns and governs.

That's the difference between a model demo and an operating model."

---

## Appendix

### Pre-demo checklist

- [ ] App running on CML (or `docker compose up` for the UI-only walkthrough —
      badge will honestly read DEMO-FALLBACK).
- [ ] Data prepared (Data dialog shows row counts for all three splits).
- [ ] Model checkpoint fetched (`pipelines/fetch_model.py`).
- [ ] At least one prior training run recorded if you want the trend chart
      populated before the live run (or reset and start from run #1).
- [ ] Registry panel enabled (on CML with APIv2 key injected) if beat 6 is in
      scope; otherwise skip beat 6 or show the disabled panel honestly.

### Technical deep-dive

For a data-scientist audience, open `ai/notebooks/tfm_training_walkthrough.ipynb` — the
same pipeline as beats 4–6, one executable stage at a time, with the training-budget
economics spelled out for business stakeholders.

### Failure-mode talking points

- **DEMO-FALLBACK badge:** "the UI runs anywhere; scores are synthetic and
  labelled as such — on the cluster this is live." Keep going.
- **Registry panel disabled:** the panel states the reason (off-CML, no APIv2
  key). Show it — an honest boundary beats a faked one.
- **Training run in progress when you arrive:** the pipeline diagram picks up
  the running job automatically; narrate from wherever it is.
- **A tier regresses slightly:** improvement is driven by real training on a
  bigger budget, not scripted numbers — point at the budget chart and let the
  honesty be the feature.
