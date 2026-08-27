# Cloudera Forge — Operating Model

> How our team identifies, builds, and ships AI accelerators on Cloudera.
>
> This document is the text version every new developer reads on day one. The repository
> you are looking at *is* the standard deliverable — this copy is instantiated as the
> **FSI / transaction-foundation-model** accelerator.

---

## The operating model in one line

**Four standard components, applied to every accelerator:**

| Component | What it is | Where it lives |
|-----------|------------|----------------|
| **A way to choose** | Sourcing + a weighted-scoring method to pick the right use cases | [Sourcing](#1-how-we-source-use-cases) · [Scoring](#2-how-we-choose-what-to-build) |
| **A build process** | A stage-gate path that moves a use case from idea to deployed | [Process](#3-the-development-process) · [RACI](#4-who-does-what) |
| **A build standard** | One reference architecture every accelerator conforms to | [Build standard](#5-the-build-standard) · [Automation](#6-how-we-automate-the-build) |
| **A standard deliverable** | Every solution ships as a single, deployable repository | [The repo](#7-the-standard-deliverable-this-repo) |

---

## 1. How we source use cases

Five inputs feed a candidate backlog. Each candidate is **typed** for the right build path.

| Input | What we look for |
|-------|------------------|
| Lost-deal & RFP analysis | Where we couldn't say yes fast enough |
| Support tickets & field patterns | The same hard ask, repeated across accounts |
| Vertical value-chain mapping | Where data and AI move a real business KPI |
| Competitive gap analysis | White space vs. other vendors' accelerators |
| Analyst & regulatory drivers | Mandates that create net-new demand |

**Each candidate is typed** — and the type drives the build path:

| Type | Build path |
|------|-----------|
| Net-new / unsolved | **greenfield** |
| Enhancement of existing | **extend** |
| Modernization / migration | **migrate** |

This accelerator is typed **greenfield**: a live fraud-inference cockpit for NVIDIA's
Transaction Foundation Model (TFM) on Cloudera AI + VAST — see the
[scorecard](./business-case/scorecard.md).

---

## 2. How we choose what to build

A weighted scorecard ranks every candidate. **Only a score ≥ 4.0 / 5 advances.**

| Band | Score | Action |
|------|-------|--------|
| 🟢 **Greenlight** | ≥ 4.0 | Advance to build |
| 🟡 **Watchlist** | 3.0 – 3.9 | Refine & re-score |
| 🔴 **Park** | < 3.0 | Decline / revisit |

> **Knockout rule:** any candidate scoring **1 on data availability or technical
> feasibility is parked**, regardless of total score.

---

## 3. The development process

How a use case moves through the team — idea to deployed, with a **gate at every handoff**.
Only high-value, repeatable use cases advance.

```
  IDENTIFY ──────────────►   BUILD ──────────────►   SHIP
 ┌──────────┬──────────┐ ┌──────────┬──────────┐ ┌──────────┬──────────┐
 │ 1        │ 2        │ │ 3        │ 4        │ │ 5        │ 6        │
 │ Discover │ Qualify  │ │Architect │ Build    │ │ Harden   │ Publish  │
 │ Field +  │ Product +│ │ Forge    │ Forge    │ │ Eng +    │ Enable + │
 │ vertical │ vertical │ │architects│ eng.     │ │ prof.    │ partners │
 │ SMEs     │ leads    │ │          │          │ │ services │          │
 └──────────┴──────────┘ └──────────┴──────────┘ └──────────┴──────────┘
        ▲                                                      │
        └──────── ↻ Field feedback continuously refreshes the catalog ────────┘
```

| # | Phase | Owner |
|---|-------|-------|
| 1 | Discover | Field + vertical SMEs |
| 2 | Qualify | Product + vertical leads |
| 3 | Architect | Forge architects |
| 4 | Build | Forge engineering |
| 5 | Harden | Eng + professional services |
| 6 | Publish | Enablement + partners |

---

## 4. Who does what

Responsibility at each phase. **R** = responsible · **A** = accountable · **C** = consulted · **—** = not involved.

| Phase | Field & SME | Product | Forge Eng | Prof. Svc | Enable & Partner |
|-----------|:-----------:|:-------:|:---------:|:---------:|:----------------:|
| Discover  | A · R | C | — | — | — |
| Qualify   | C | A · R | C | — | — |
| Architect | — | C | A · R | C | — |
| Build     | — | — | A · R | C | — |
| Harden    | — | — | R | A | C |
| Publish   | — | C | — | — | A · R |

Each phase has exactly **one accountable owner**; the feedback loop keeps the catalog current.

---

## 5. The build standard

Every accelerator conforms to this stack, so builds stay **consistent, governed, and reusable**.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Cloudera platform — hybrid: on-prem · cloud · edge                  │
├─────────┬───────────┬───────────┬──────────────────────┬────────────┤
│ Ingest  │ Lakehouse │ Process   │ AI layer             │ Serve      │
│ DataFlow│ Apache    │ Data Eng  │ Cloudera AI ·        │ Data Viz · │
│ (NiFi)  │ Iceberg   │ Data      │ Inference ·          │ APIs ·     │
│ stream  │ + Ozone   │ Warehouse │ Agent Studio         │ embedded   │
│ + batch │           │           │                      │ apps       │
├─────────┴───────────┴───────────┴──────────────────────┴────────────┤
│  Cloudera SDX — governance · security · lineage across every layer   │
└─────────────────────────────────────────────────────────────────────┘
```

In this accelerator the layers are realized on **Cloudera AI (CML) + VAST**: Ingest =
`pipelines/prepare_data.py` staging transaction splits to VAST/S3 (Impala-readable
Parquet), Lakehouse = VAST object storage + Impala external tables, Process = the AMP job
chain (fetch model → prepare data), AI = the GPU tokenizer + TFM decoder → 512-d
embeddings → PCA-64 → three XGBoost heads (+ optional NEXUS LTM) in `src/` + `tfm_demo/`,
Serve = the FastAPI + React cockpit (`app/serve_app.py`). Deviations are ADRs in
[`architecture/`](./architecture/).

---

## 6. How we automate the build

Each accelerator deploys onto the **running platform** through native service APIs — **one Git-driven pipeline**.

```
 1. Commit to Git  ──►  2. Build & test  ──►  3. Deploy to platform
    source + config        CDP CLI               service APIs
```

> **Platform provisioning is one-time:** the CML workspace and (optionally) a GPU runtime
> are stood up once (see [`../deploy/README.md`](../deploy/README.md)); every deploy after
> that is declarative via the AMP manifest.

| Layer | What it automates | APIs & tools (this accelerator) |
|-------|-------------------|--------------------------------|
| **Ingest** | Transaction splits → VAST/S3 Parquet (+ Impala DDL) | `pipelines/prepare_data.py` (AMP job; also run from the UI via `tfm_demo/jobs.py`) |
| **Process** | Model + blueprint code staging | `pipelines/fetch_model.py` (AMP job) |
| **AI** | TFM embeddings → XGBoost heads → live scoring | `src/` (tokenizer, decoder) · `tfm_demo/engine.py` · GPU runtime |
| **Serve** | The inference cockpit Application | `app/serve_app.py` (uvicorn behind Vite preview on `CDSW_APP_PORT`) |
| **Registry** | Model versioning + endpoint deploy (UI-triggered) | `tfm_demo/registry.py` (MLflow-backed Model Registry + cmlapi APIv2; run history in `tfm_demo/runs.py`) |
| **Governance** | Model cards | [`../governance/`](../governance/) |

---

## 7. The standard deliverable (this repo)

Every team produces the **same structure**, so any solution deploys clean from the repo alone.

```
cloudera-forge-fsi-foundational-model/   (folder name: demo-foundational-model)
├── docs/          APP_GUIDE · runbooks · architecture · business case
├── deploy/         install_deps · build_frontend · VAST/NEXUS probes · docker/
├── data/          layer docs — data lives on VAST/S3 + Impala (see data/README.md)
├── pipelines/     fetch_model.py · prepare_data.py (AMP jobs)
├── ai/            layer docs — inference lives in src/ + tfm_demo/ (ADR-001)
├── app/           serve_app.py · frontend/ (React SPA)
├── governance/    model cards (TFM + XGBoost heads)
├── tests/         conventions
├── .cicd/         build → test → deploy
├── src/           blueprint package: tokenizer · decoder inference (root: import contract)
├── tfm_demo/      the backend package (root: import contract) + app.py entrypoint
└── models/        fetched TFM checkpoint (root: CWD-relative contract)
```

**`.project-metadata.yaml` wires the whole solution into CML as an AMP — five declared
tasks from install to the running Application.**

What makes this the standard:

- **Provisioning scripted** — the environment is stood up from `deploy/`
- **Pipelines & AI included** — fetch, prepare, embed, score, compare — wired
- **Governance packaged** — model cards travel with the repo
- **The Phase-4 gate** — runs clean from the repo alone

### Directory map

| Directory | Build-standard layer | README |
|-----------|----------------------|--------|
| `deploy/` | Platform provisioning (one-time) | [deploy/README.md](../deploy/README.md) |
| `data/` | Ingest + Lakehouse | [data/README.md](../data/README.md) |
| `pipelines/` | Process (AMP tasks) | [pipelines/README.md](../pipelines/README.md) |
| `ai/` | AI layer (agent · model serving) | [ai/README.md](../ai/README.md) |
| `app/` | Serve (the fraud dashboard) | [app/README.md](../app/README.md) |
| `governance/` | Model governance | [governance/README.md](../governance/README.md) |
| `tests/` | Harden gate (data quality · AI eval) | [tests/README.md](../tests/README.md) |
| `.cicd/` | Build → test → deploy pipeline | [.cicd/README.md](../.cicd/README.md) |
| `docs/` | App guide · architecture · business case | [docs/README.md](./README.md) |

---

## 8. Lifecycle of one accelerator

From selected use case to a deployed, governed solution — **≈ 8 weeks**. The Build phase
scales with solution complexity.

| Week | Phase |
|------|-------|
| wk 0 | Discover |
| wk 1 | Qualify |
| wk 2 | Architect |
| wk 3–6 | **Build** |
| wk 7 | Harden |
| wk 8 | Publish → **Deployed** |

---

## New developer: start here

1. **Read this document** end to end — it is the whole operating model.
2. **Walk the worked example:** [EXAMPLE.md](./EXAMPLE.md) traces this foundation-model
   accelerator across every layer (Ingest → Serve).
3. **Read [APP_GUIDE.md](./APP_GUIDE.md)** — the authoritative build/run guide, including
   VAST/Impala configuration and GPU runtime notes.
4. **Skim each directory's `README.md`** — every one tells you what goes there, the naming
   convention, and which Cloudera API/tool automates it.
5. **Run the app:** the Quickstart in the [root README](../README.md#quickstart).
6. **Know the bar:** [GATES.md](./GATES.md) is the definition of done at each handoff.
7. **Ship it:** deploy the repo as a CML AMP (`.project-metadata.yaml`).

© 2025 Cloudera, Inc. · Internal
