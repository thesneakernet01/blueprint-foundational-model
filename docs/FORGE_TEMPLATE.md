# Cloudera Forge — FSI Transaction Foundation Model Accelerator

> **The standard deliverable.** This repository is a Cloudera Forge AI accelerator
> (`cloudera-forge-fsi-foundational-model`) following the reference structure every
> accelerator uses: a **live fraud-inference cockpit** for NVIDIA's Transaction Foundation
> Model on Cloudera AI + VAST — raw transaction → GPU tokenizer → TFM decoder → 512-d
> embedding → PCA-64 → three XGBoost heads (raw / embeddings / combined, + optional NEXUS
> LTM), with AUC/AP lift and UMAP views in a React SPA.

**New to the team? Read [`docs/OPERATING_MODEL.md`](./OPERATING_MODEL.md) first.**
Then walk this accelerator layer by layer in [`docs/EXAMPLE.md`](./EXAMPLE.md), and
use [`docs/GATES.md`](./GATES.md) to know what "done" means at each phase.

> **Building or running this app? The authoritative build/run guide is
> [`docs/APP_GUIDE.md`](./APP_GUIDE.md)** (formerly this repo's root README), with the
> demo path in [`docs/presentations/README_DEMO.md`](./presentations/README_DEMO.md)
> and the FSI runbook in
> [`docs/presentations/demo-runbook-fsi.md`](./presentations/demo-runbook-fsi.md).

---

## The standard repo

```
cloudera-forge-fsi-foundational-model/   (folder name: demo-foundational-model)
├── docs/          APP_GUIDE · impala-vast-s3-config · nexus-ltm-design · business case
├── deploy/         install_deps.py · build_frontend.py · VAST/NEXUS probes · docker/
├── data/          (transaction splits live on VAST/S3 + Impala — see data/README.md)
├── pipelines/     fetch_model.py · prepare_data.py             → AMP jobs
├── ai/            (inference lives in src/ + tfm_demo/ — ADR-001)
├── app/           serve_app.py · frontend/ (React SPA)         → the CML Application
├── governance/    model cards (TFM + XGBoost heads)
├── tests/         conventions
├── .cicd/         build → test → deploy
├── src/           blueprint package (tokenizer · decoder inference)   → root: import contract
├── tfm_demo/      the backend package · app.py entrypoint            → root: import contract
├── models/        fetched TFM checkpoint                             → root: CWD-relative, gitignored
├── requirements-*.txt · docker-compose.yml · export_for_demo.py      → root exceptions (see ADR-001)
└── .vast.env                                                         → VAST credentials (untracked)
```

---

## The four standard components

| Component | What it is | Read |
|-----------|------------|------|
| **A way to choose** | Sourcing + weighted scoring (≥ 4.0 / 5 advances) | [§1–2](./OPERATING_MODEL.md#1-how-we-source-use-cases) · [scorecard](./business-case/scorecard.md) |
| **A build process** | 6-phase stage-gate (Discover → Publish) | [§3–4](./OPERATING_MODEL.md#3-the-development-process) |
| **A build standard** | One reference stack: Ingest → Lakehouse → Process → AI → Serve | [§5–6](./OPERATING_MODEL.md#5-the-build-standard) · [architecture](./architecture/README.md) |
| **A standard deliverable** | This repo — a working accelerator, deployable as a CML AMP | [§7](./OPERATING_MODEL.md#7-the-standard-deliverable-this-repo) |

---

## Quickstart

Deploys as a **CML AMP** (`.project-metadata.yaml`, five tasks, GPU runtime) or runs
manually:

```bash
python deploy/install_deps.py       # pinned Python deps (GPU-aware)
python deploy/build_frontend.py     # user-local Node + Vite build → app/frontend/dist
python pipelines/fetch_model.py    # TFM checkpoint + blueprint src/ (idempotent)
python pipelines/prepare_data.py   # transaction splits → VAST/S3 Parquet + Impala DDL
python app/serve_app.py            # uvicorn (127.0.0.1:$BACKEND_PORT) behind Vite preview on $CDSW_APP_PORT

# Connectivity diagnostics:
python deploy/vast_probe.py         # VAST S3 metadata calls
python deploy/nexus_probe.py        # NEXUS LTM endpoint
```

VAST credentials live in the **untracked** `.vast.env`; Impala/S3a specifics in
[`docs/impala-vast-s3-config.md`](./impala-vast-s3-config.md).

---

## The ~8-week lifecycle

| Week | wk 0 | wk 1 | wk 2 | wk 3–6 | wk 7 | wk 8 |
|------|------|------|------|--------|------|------|
| Phase | Discover | Qualify | Architect | **Build** | Harden | Publish → **Deployed** |

---

## What's already in this accelerator

- **`tfm_demo/` + `src/`** — the inference engine: GPU tokenizer + TFM decoder →
  embeddings → PCA-64 → three XGBoost heads with live AUC/AP lift, UMAP, Impala/VAST
  readers, background jobs, and an optional NEXUS long-term-memory head
  ([design](./nexus-ltm-design.md)).
- **`app/frontend/`** — the React SPA cockpit (served via Vite preview in front of the
  API).
- **`pipelines/`** — the AMP job chain (model fetch, data preparation to VAST/Impala).
- **`deploy/`** — pinned installs, the frontend toolchain, VAST/NEXUS connectivity probes,
  and the Docker path (`deploy/docker/` + root compose).

© 2025 Cloudera, Inc. · Internal
