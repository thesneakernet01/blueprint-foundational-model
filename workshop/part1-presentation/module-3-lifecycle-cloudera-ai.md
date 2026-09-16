# Module 3 — End-to-End Model Lifecycle on Cloudera AI (15 min)

**Goal:** attendees can narrate the full loop — governed data in the lakehouse → GPU
training run → hot-reloaded scoring engine → registered model version → governed REST
endpoint — and name the Cloudera component that owns each step.

---

## Slide 3.1 — Governed data pipelines (3 min)

**On the slide**

- TabFormer temporal splits (**train / val_eval / test_eval**) live in **enterprise
  storage**, not in files shipped with the app:
  - **Impala tables** (Parquet-backed, in a governed CDW database), or
  - **S3-compatible object storage** (VAST/MinIO — one Parquet object per split)
- Training reads splits **directly from the lakehouse** into GPU memory: Impala/S3 →
  pandas chunk → **cuDF on the GPU**.
- Ingestion is an AMP job (`pipelines/prepare_data.py`): downloads the ~2.4 GB TabFormer
  archive, computes the 80/10/10 **temporal** cutoffs, keeps *all* fraud rows in train
  (capped at 1M rows total), samples ~100K rows each for val/test at natural fraud rate.
- Storage is configured from the app's **Data dialog** — backend picker, connection
  test, and a "Load TabFormer" button that runs the whole pipeline with a streaming log.

**Speaker notes**

- The governance point: the training data is a **queryable, governed asset** — the same
  Impala tables your SQL analysts and BI tools see, with lineage in the platform — not a
  CSV someone scp'd into a notebook. That is the Cloudera differentiation sentence.
- Deterministic and idempotent: fixed seed (42), reproducible cutoffs, and re-ingest
  automatically invalidates the embedding cache so nothing goes stale silently.
- Code anchors: `tfm_demo/storage.py` (backend dispatch), `tfm_demo/impala.py`,
  `tfm_demo/vast.py`; config persisted with 0600 perms via `tfm_demo/settings.py`.

---

## Slide 3.2 — NVIDIA GPU pipeline execution (4 min)

**On the slide**

One training click runs six stages, all on a single GPU:

```
check → embed → pca → train → (nexus) → artifacts        [then: reload → done]
```

- **embed** — cuDF tokenizer frames → TFM decoder (torch, batch 512) → 512-d embeddings
- **pca** — 512 → 64 compression
- **train** — three XGBoost heads with `device="cuda"` (GPU hist tree method)
- **artifacts** — plus **cuML UMAP** fits a 2-D projection of up to 8,000 test
  embeddings for the live map
- Fits a **24 GB NVIDIA L4** by design: RMM pool allocator (1 GiB, growable), cuDF
  spilling enabled, PyTorch expandable segments, one-split-at-a-time streaming.

**Speaker notes**

- Emphasize the "one commodity GPU" story: the blueprint upstream assumes an 80 GB
  A100/H100; this app was deliberately engineered (`tfm_demo/gpu.py`) to run the entire
  end-to-end workload on an L4 — the GPU customers can actually get approved.
- RAPIDS name-drop map: **cuDF** (data frames + tokenizer input), **RMM** (shared
  memory pool across cuDF/CuPy/XGBoost), **cuML** (UMAP), **XGBoost-on-CUDA** (heads),
  **torch cu121** (decoder). All pinned in `requirements-gpu.txt`.
- Live proof point in the lab: the export dialog shows real GPU utilization/VRAM meters
  from `nvidia-smi` while the run executes.
- War story if asked about robustness: the stack routes numba through NVIDIA's CUDA
  bindings and runs a "host-copy canary" subprocess before every export so a driver
  incompatibility fails with a readable error instead of crashing the server.

---

## Slide 3.3 — Hot reloading & the training budget ladder (4 min)

**On the slide**

**Hot reload** — when a run finishes, the in-process engine reloads the new artifacts
(`reload` stage) and the *next scored transaction uses the new model*. No restart, no
redeploy, no service disruption.

**Budget ladder** — each successive run earns a bigger budget:

| Run | Rows embedded / split | Boosting rounds |
| --- | --- | --- |
| 1 | 4,000 | 35% |
| 2 | 8,000 | 55% |
| 3 | 12,000 | 75% |
| 4 | 16,000 | 90% |
| 5+ | 20,000 | 100% |

**Audit trail** — every run appends to a persistent history: run id, timestamps,
duration, budget, per-head AUC/AP, lift, diagnostics, registry status.

**Speaker notes**

- The ladder is the demo's narrative engine: it turns "retrain" into a visible
  *invest-more-compute → get-more-quality* curve, charted per run in the UI
  (`BudgetChart`, `MetricTrend`). It mirrors the real conversation with customers about
  compute budgeting for model refreshes.
- Every run also emits diagnostics rendered in the UI: per-round **validation AUC
  curves** (with the early-stopping best round), a **class-separation histogram** of
  combined-head scores on test, and **top-15 feature importance** where each feature is
  tagged *raw* vs *embedding* — the killer chart: embedding components claim top slots.
- Audit framing: run history lives in `.runs_history.json` (survives artifact
  overwrites) — recorded, replayable performance gains, not anecdotes. "Reset demo"
  clears it so the story replays from run 1.
- Code anchors: `tfm_demo/runs.py` (`_TIERS`), `tfm_demo/jobs.py::ExportManager`
  (train → `engine.warmup()` hot reload).

---

## Slide 3.4 — Cloudera enterprise MLOps & governance (4 min)

**On the slide**

- **Register**: one click logs the run to MLflow (params, per-head AUC/AP metrics) and
  registers the model as **`tfm-fraud-combined`** in the **Cloudera AI Model Registry**
  — on Cloudera AI, the MLflow tracking server *is* the registry, so experiment lineage
  and the registered version are one governed record.
- **Deploy**: one click builds and deploys a **CML Model REST endpoint** (2 CPU / 4 GB)
  from the registered version — governed, versioned, with full lineage back to the run
  and the data that trained it.
- **Registered scope**: preprocessor + PCA + combined XGBoost head. The endpoint scores
  rows carrying raw features + `pca_0..pca_63`; the TFM embedding stage runs upstream.
- Registry versions appear as **markers on the metric-trend chart** — you can see which
  run each production version came from.

**Speaker notes**

- This is where Cloudera pulls away from "a GPU box with notebooks": experiment
  tracking, model registry, build, deployment, and endpoint auth are platform
  primitives, driven here through `cmlapi`/MLflow from inside the app — no console
  hopping.
- Be crisp on the endpoint contract (customers will ask): the registered asset is the
  classical head bundle, deployable on CPU; the foundation-model embedding service
  scales independently on GPU. That separation *is* a production architecture pattern,
  not a demo shortcut.
- Governance collateral in-repo: model cards under `governance/model-cards/` and the
  design docs under `docs/` — point customers at them for their model-risk teams.
- Code anchor: `tfm_demo/registry.py` (`register_latest`, `deploy_latest`).
