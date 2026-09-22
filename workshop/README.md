# SE Workshop: Transaction Foundation Models on Cloudera AI with NVIDIA GPU Acceleration

A 2-hour enablement workshop for technical Sales Engineers, built directly on this
repository's live fraud-inference application. The narrative: **Cloudera AI is the
unified platform** (governed data, in-app training, model registry, governed REST
endpoints) and **NVIDIA GPUs are the compute engine** (tokenizer → TFM decoder →
RAPIDS/cuDF → XGBoost-on-CUDA → cuML UMAP, all on a single commodity L4).

> **Audience variants:** this Part 1 deck is pitched at technical SEs (300-level).
> For a business-technical room there is a 200-level rewrite of the same deck in
> [`evolve-nyc/`](evolve-nyc/) (built for Evolve NYC), with its own speaker guide.

## Session structure

| Segment | Duration | Materials |
| --- | --- | --- |
| **Part 1 — Presentation & Technical Deep-Dive** | 50 min | [`part1-presentation/`](part1-presentation/) |
| Break | 10 min | — |
| **Part 2 — Hands-On Implementation Lab** | 60 min | [`part2-lab/`](part2-lab/) |

## Part 1 modules (50 min)

| Module | Time | File |
| --- | --- | --- |
| 1. The Paradigm Shift in Fraud Detection | 10 min | [`module-1-paradigm-shift.md`](part1-presentation/module-1-paradigm-shift.md) |
| 2. Core Terminology & Architecture | 15 min | [`module-2-terminology-architecture.md`](part1-presentation/module-2-terminology-architecture.md) |
| 3. End-to-End Model Lifecycle on Cloudera AI | 15 min | [`module-3-lifecycle-cloudera-ai.md`](part1-presentation/module-3-lifecycle-cloudera-ai.md) |
| 4. Platform Architecture & Lab Preview | 10 min | [`module-4-platform-lab-preview.md`](part1-presentation/module-4-platform-lab-preview.md) |

Each module file is a slide-by-slide outline with speaker notes, the exact numbers to
quote, and pointers into the codebase so presenters can answer "show me" questions.

## Part 2 lab (60 min)

| File | Purpose |
| --- | --- |
| [`00-instructor-setup.md`](part2-lab/00-instructor-setup.md) | Environment prep — do this **before** the session |
| [`lab-guide.md`](part2-lab/lab-guide.md) | The attendee-facing lab: live scoring, GPU retraining, diagnostics, registry + endpoint deployment |

## Support materials

| File | Purpose |
| --- | --- |
| [`facilitator-guide.md`](facilitator-guide.md) | Run-of-show, timing checkpoints, anticipated Q&A / objection handling |
| [`glossary.md`](glossary.md) | One-page terminology reference (handout) |

## Source-of-truth pointers

Everything in these materials is grounded in the application itself:

- App build/run guide: [`../docs/APP_GUIDE.md`](../docs/APP_GUIDE.md)
- 7-beat presenter demo script: [`../docs/demo-script.md`](../docs/demo-script.md)
- AMP manifest (the five one-click tasks): [`../.project-metadata.yaml`](../.project-metadata.yaml)
- Training/export pipeline: [`../tfm_demo/export.py`](../tfm_demo/export.py)
- GPU memory strategy for the 24 GB L4: [`../tfm_demo/gpu.py`](../tfm_demo/gpu.py)
- Model Registry + endpoint deployment: [`../tfm_demo/registry.py`](../tfm_demo/registry.py)
- Model card: [`../governance/model-cards/tfm_fraud_heads.md`](../governance/model-cards/tfm_fraud_heads.md)
