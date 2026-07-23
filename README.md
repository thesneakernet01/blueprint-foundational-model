# Transaction Foundation Model — Live Fraud Inference Cockpit

A single-screen demo app for **Cloudera AI (CML)** that runs live fraud inference
against the real NVIDIA **Transaction Foundation Model (TFM)** checkpoint. You
compose (or load) a card transaction and watch it flow through the actual
blueprint pipeline:

```
raw txn → GPU tokenizer → decoder foundation model → 512-d embedding
        → PCA-64 → 3 XGBoost heads (raw / embeddings / combined)
        → fraud probabilities + AUC/AP lift + UMAP position
```

The headline: the **embeddings** and **combined** heads score the same
transaction next to a **raw-features baseline**, with the held-out test-set
AUC/AP lift shown up top — "no hand-crafted features; the model learned this
from raw transaction sequences." An optional fourth head (**NEXUS**,
Fundamental's Large Tabular Model) extends the story to
"classic GBM vs. TFM embeddings vs. Large Tabular Model."

- **Backend:** FastAPI (`tfm_demo/`), runs the NVIDIA GPU stack in-process.
- **Frontend:** React/Vite SPA (`frontend/`), pure client, calls same-origin `/api/*`.
- **Deployment shape on CML:** one Application, one GPU container —
  `scripts/serve_app.py` supervises uvicorn (private on `127.0.0.1:$BACKEND_PORT`)
  and a Vite preview server (public on `$CDSW_APP_PORT`) that proxies `/api/*` inward.

Companion docs: [`README_DEMO.md`](README_DEMO.md) (modes, storage backends,
customising), [`docs/demo-runbook-fsi.md`](docs/demo-runbook-fsi.md) (presenter
run-of-show and talk tracks), [`docs/nexus-ltm-design.md`](docs/nexus-ltm-design.md)
(NEXUS integration design).

---

## Deploying on Cloudera AI

The repo is a CML **AMP** — [`.project-metadata.yaml`](.project-metadata.yaml)
declares the whole deployment.

### Prerequisites

- A CML workspace with **GPU nodes**. A single **24 GB L4** is enough — the
  checkpoint is small (~56 MB) and `tfm_demo/gpu.py` configures an RMM pool +
  cuDF spill so the export fits (the upstream blueprint assumes an 80 GB
  A100/H100; this app does not).
- A **Python 3.12 / Nvidia GPU edition** ML Runtime (JupyterLab editor). CUDA
  must be present in the runtime; the setup job installs only Python wheels.
- **No Spark.** The AMP intentionally declares no runtime add-ons — don't
  attach one.
- Somewhere to keep the training splits (choose one, configured after deploy):
  - an **S3-compatible object store** (MinIO or VAST; the app calls this
    backend "VAST" for historical reasons) — endpoint, bucket, and keys; or
  - a **CDW Impala Virtual Warehouse**, reachable as a CML data connection.

### 1. Create the project from git

**Projects → New Project → Import from Git**, point at this repo. CML reads
`.project-metadata.yaml` and offers to run the AMP steps. Accept the defaults —
the env vars can all stay empty (everything is configurable in-app afterwards).

The AMP runs four one-shot Jobs, then starts the Application:

| Step | Script | What it does |
|---|---|---|
| Install dependencies | `scripts/install_deps.py` | `requirements-demo.txt` (web layer) + `requirements-gpu.txt` (RAPIDS cu12, torch cu121, transformers, xgboost — carefully pinned, see below). Slow: multi-GB wheels. |
| Build frontend SPA | `scripts/build_frontend.py` | Installs Node user-locally (`~/.local/node`, no root) and runs `npm ci && npm run build` → `frontend/dist`. |
| Fetch model checkpoint | `scripts/fetch_model.py` | Downloads the decoder checkpoint into `models/` and the blueprint's `src/` package (tokenizer + inference code) over plain HTTPS — no git-lfs needed. |
| Prepare TabFormer data | `scripts/prepare_data.py` | Downloads TabFormer (~2.4 GB from IBM Box) and writes the temporal splits. **Exits 0 with guidance if storage isn't configured yet** — expected on a fresh deploy; you re-run it from the UI in step 3. |
| **TFM Fraud Demo** (Application) | `scripts/serve_app.py` | Backend + UI in one GPU container; `bypass_authentication: true`, so the URL is publicly reachable. |

### 2. Open the app and configure storage

Open the Application URL. The header badge tells you the mode:

- **REAL** — checkpoint + GPU + artifacts loaded; every score is a live forward
  pass. This is what you demo.
- **DEMO-FALLBACK** — GPU, checkpoint, or artifacts missing; the UI runs with
  clearly-labelled synthetic scores. Expected until you finish step 3.

Click the header **Data** button and pick a backend:

- **VAST S3** (recommended; despite the name, any S3-compatible store — it
  currently targets a MinIO bucket): enter endpoint, bucket, folder, access +
  secret key, then **Save & test**. Path-style addressing and sigv4 are
  hard-wired; region defaults to `us-east-1` (`$VAST_REGION` overrides); TLS
  verification is off unless `$VAST_VERIFY_SSL=1` or a CA bundle path. Splits
  are written as Parquet objects directly with boto3.
- **Impala (CDW)**: enter the CML data connection name and database. (Outside
  CML, `$IMPALA_HOST` + `_PORT/_USER/_PASSWORD/_AUTH/_SSL/_HTTP` drive impyla
  directly.)

Settings persist to `.data_settings.json` in the project (mode 0600 — it can
hold the S3 secret key). `$VAST_*` / `$IMPALA_*` env vars only seed the dialog
defaults.

> Don't route the splits through Impala-on-VAST s3a — the Java s3a client 400s
> against the VAST endpoint (boto3 does not). Background:
> `docs/impala-vast-s3-config.md`.

### 3. Load data and build artifacts (in-app)

1. **Load TabFormer** (Data dialog) — downloads ~2.4 GB, reproduces the
   blueprint's temporal split in chunked pandas (train capped at 1M rows,
   `$PREP_TRAIN_CAP`), and writes `train` / `val_eval` / `test_eval` to your
   backend. Minutes to tens of minutes; idempotent, and the UI button forces a
   re-ingest.
2. **Build artifacts** — reads the splits back into cuDF on the GPU, generates
   embeddings through the decoder (cached per data-target under
   `data/embeddings/`; the *first* build is the slow one), trains the three
   XGBoost heads + PCA + UMAP, and writes `demo_artifacts/`. Progress is polled
   live in the UI.
3. Refresh — the badge should read **REAL**.

**Pre-stage both of these the day before a live demo** so the embedding cache
is warm; see `docs/demo-runbook-fsi.md` for the full run-of-show.

---

## Environment variables (all optional)

| Variable | Purpose |
|---|---|
| `BACKEND_PORT` | Private in-container FastAPI port (default `7100`). |
| `SKIP_GPU_DEPS=1` | Install web layer only → DEMO-FALLBACK mode (UI preview without a GPU). |
| `VITE_API_BASE` | Leave empty for the standard single-app deploy; set only for a separately-hosted backend. |
| `VAST_ENDPOINT/BUCKET/PATH/ACCESS_KEY/SECRET_KEY` | Seed the Data dialog's S3 defaults. |
| `VAST_UPLOAD_PART_MB` / `VAST_UPLOAD_CONCURRENCY` / `VAST_UPLOAD_RETRIES` | Upload tuning (defaults 64 / 8 / 2). |
| `IMPALA_CONNECTION_NAME` / `IMPALA_DATABASE` | Seed the Data dialog's Impala defaults. |
| `PREP_SCRATCH` / `PREP_TRAIN_CAP` / `PREP_FORCE` | Data-prep scratch dir, train row cap (1M), force re-ingest. |
| `NEXUS_MODE` + `NEXUS_*` | Fourth-head config — but prefer the in-UI selector (below). |

## The NEXUS fourth head (optional)

The **Build artifacts** dialog has an **off / stub / live** selector for the
NEXUS (Fundamental Large Tabular Model) card — persists to
`.nexus_settings.json`, applies immediately, wins over `$NEXUS_MODE`:

- **off** (default) — the card is hidden.
- **stub** — demos the full 4-model path today with deterministic fake scores
  clearly tagged `stub`. Nothing extra needed.
- **live** — scores via a pre-deployed SageMaker endpoint
  (`$NEXUS_ENDPOINT_NAME`, `$NEXUS_S3_BUCKET`, optional
  `$NEXUS_REGION/_S3_PREFIX/_SCORE_TIMEOUT_S/_TRAIN_MAX`), then re-run the
  export. **The endpoint is ~$60+/hr (`ml.p5en.48xlarge`)** — deploy right
  before the demo, delete right after; this app never creates or deletes
  endpoints. The live transport ships intentionally disabled until Fundamental
  access lands — day-one checklist in `docs/nexus-ltm-design.md`, verification
  via `scripts/nexus_probe.py`.

NEXUS degrades on its own (2.5 s score budget; a timeout shows `—`) and REAL
mode never depends on it.

## GPU notes (why the pins look the way they do)

Handled automatically — listed so nobody "fixes" them backwards:

- `requirements-gpu.txt` pins the **RAPIDS 24.12 cu12** matrix, `torch
  2.5.1+cu121`, `transformers 4.53.*` (the checkpoint's saved version), and
  `cuda-python==12.6.0` exactly. `install_deps.py` purges stray CUDA-13
  packages from earlier unpinned installs so re-runs converge.
- `tfm_demo/gpu.py` sets `NUMBA_CUDA_USE_NVIDIA_BINDING=1` at import — on
  r580/CUDA-13 drivers, numba-cuda's ctypes shim segfaults every cuDF
  device→host copy (NVIDIA/numba-cuda#173) and no in-matrix pin avoids it.
  The export preflights the copy path in a subprocess so a broken stack fails
  the build readably instead of killing the server.
- It also configures an **RMM pool + cuDF spill** so the export fits a 24 GB
  card. Opt-outs: `DEMO_RMM_POOL/DEMO_CUDF_SPILL/DEMO_TORCH_EXPANDABLE/DEMO_NUMBA_NV_BINDING=0`.

## Troubleshooting

- **Badge stuck on DEMO-FALLBACK** — check, in order: the fetch-model job ran
  (`models/` and `src/` exist), the app has a GPU, and **Build artifacts** has
  completed (`demo_artifacts/` exists).
- **"No module named src"** during export — re-run the *Fetch model checkpoint*
  job; it downloads the blueprint `src/` package as well as the checkpoint.
- **Prepare-data job "succeeded" but no data** — that's the unconfigured-storage
  guard. Configure the Data dialog, then click **Load TabFormer** in the UI.
- **S3 uploads stall or fail** — run `scripts/vast_upload_probe.py` and check
  the wire-config banner; tune `$VAST_UPLOAD_*` before touching code. If the
  store can't parse trailing checksums, set `$VAST_TRAILING_CHECKSUMS=0`
  (and/or `$VAST_EXPECT_100=0`). Also confirm the dialog's *folder* field is
  set — an empty prefix drops objects at the bucket root.
- **Export SIGSEGV on device→host copies** — verify
  `NUMBA_CUDA_USE_NVIDIA_BINDING=1` wasn't disabled (see GPU notes).
- **Changed the data target and scores look stale** — re-ingesting clears the
  embedding cache automatically; the cache is keyed by row position per target,
  so never hand-edit splits in place.

## Local UI preview (no Cloudera, no GPU)

```bash
docker compose up --build     # then open http://localhost:8100
```

Real SPA + real FastAPI backend, but no GPU stack — the engine serves
DEMO-FALLBACK with labelled synthetic scores. For UI work only; live inference
needs the GPU, model, and data in CML.

## Repo layout

```
.project-metadata.yaml   # CML AMP spec — jobs + the single Application
scripts/                 # AMP job entrypoints + probes (vast_*, nexus_probe)
tfm_demo/                # FastAPI backend: engine, export, jobs, storage (vast/impala), gpu, nexus
frontend/                # React/Vite SPA (dist/ committed; rebuilt in-cluster)
src/                     # blueprint tokenizer + decoder inference (fetched by fetch_model.py)
models/                  # decoder checkpoint (fetched by fetch_model.py)
docs/                    # runbook, NEXUS design, Impala/VAST s3a notes
docker*, docker-compose.yml  # off-GPU UI preview
```

Built on the NVIDIA AI Blueprint *Transaction Foundation Model* (Apache-2.0).
