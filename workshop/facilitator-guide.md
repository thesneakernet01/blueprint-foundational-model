# Facilitator Guide — Run of Show, Checkpoints & Q&A

## Run of show (2h00)

| Clock | Segment | Facilitator cues |
| --- | --- | --- |
| 0:00 | Welcome + Module 1 (10') | Hook with the rules-plateau pain; promise a *measured* lift by 1:50 |
| 0:10 | Module 2 (15') | Whiteboard the inference path once, slowly; the multi-head slide is the credibility anchor |
| 0:25 | Module 3 (15') | Optionally flash the live app during 3.2/3.3 (stage chips, budget chart) — 60 seconds max, don't start the lab early |
| 0:40 | Module 4 (10') | End with lab logistics: instance URLs, lab-guide location, pairing |
| 0:50 | **Break (10')** | Start a tier-1 training run on YOUR instance now — you'll have fresh diagnostics to reference during the lab |
| 1:00 | Lab Ex 1 — live scoring (12') | Everyone must hit the `live · GPU` checkpoint in the first 3 minutes |
| 1:12 | Lab Ex 2 — GPU retrain (18') | Get all runs *launched* by 1:16; narrate stages/GPU meters over the room while they wait |
| 1:30 | Lab Ex 3 — diagnostics (12') | Attendees start Ex 3 while their tier-2 run trains — this overlap is designed, call it out |
| 1:42 | Lab Ex 4 — register + deploy (15') | Have everyone click **Deploy** by 1:47 (build takes ~5–10'); do the `curl` against your pre-deployed endpoint if builds lag |
| 1:57 | Wrap (3') | The three customer-facing sentences at the end of the lab guide; where the materials live |

## Hard checkpoints (do not sail past a red one)

1. **0:50** — every instance shows `live · GPU` (check before the break, not after).
2. **1:16** — every attendee has a training run *running*. A run not launched by then
   means Ex 2.3's hot-reload proof slips; have them pair up.
3. **1:47** — every deploying attendee has clicked **Deploy**. Later than that, switch
   to the shared pre-deployed endpoint for the `curl` step.

## Numbers to have memorized

- TabFormer: ~24M transactions, 20K cardholders, ~2.4 GB; splits 80/10/10 **temporal**;
  train capped at 1M rows (all fraud kept), val/test ~100K each.
- Model: 512-d last-token embedding → PCA-64; checkpoint ~56 MB (8 layers, hidden 512,
  vocab 6,251); tokenizer max_length 128.
- Heads: raw / embed / combined XGBoost, `device="cuda"`, early stopping 20 on val AUC.
- Budget ladder: 4K/8K/12K/16K/20K rows per split at 35/55/75/90/100% rounds.
- Metrics: held-out test **ROC-AUC** and **Average Precision**; lift % vs raw baseline.
- Hardware: everything on one **24 GB NVIDIA L4**; registered endpoint runs 2 CPU/4 GB.
- Registry: model `tfm-fraud-combined`, experiment `tfm-fraud-demo`.

## Anticipated Q&A / objection handling

**"Is the lift real or did you tune the embedding heads harder?"**
Hyperparameters are frozen per head (copied verbatim from the blueprint's tuning
notebooks), identical splits and training procedure; the only variable is the
representation. Show `XGB_PARAMS_*` in `tfm_demo/export.py` if pressed.

**"Why is the test split temporal instead of random?"**
Because production fraud models score the future. Random splits leak future behavior
into training and inflate metrics; the temporal 80/10/10 is the honest evaluation.

**"Why PCA on the embeddings? Aren't you throwing away signal?"**
64 components keep the bulk of variance, decorrelate inputs for tree splits, cut
training cost 8×, and define a compact endpoint contract. The benchmark's lift is
measured *after* PCA — the reported number already pays that cost.

**"Do I need the GPU at inference time in production?"**
The registered/deployed asset (preprocessor + PCA + combined head) runs on CPU. The
GPU serves the embedding stage, which scales independently — the demo runs both in one
container for simplicity; production separates them (see the model card's registered
scope).

**"Can this run on our data / our use case?"**
The harness is dataset-agnostic in design: governed splits, frozen-head benchmarking,
budget ladder, registry lineage. Swapping in customer data means a tokenizer/schema
mapping exercise — position it as the follow-on workshop. AML, credit risk, payments
abuse, and churn map directly.

**"What about model risk / governance sign-off?"**
Point at the in-repo model cards (`governance/model-cards/`), the persisted run history
(budgets, metrics, timestamps per run), MLflow experiment lineage, and registry
versioning — the artifacts a model-risk team asks for exist on day one.

**"What GPUs do we need to buy?"**
Demo: one L4-class (24 GB). Production guidance in the README: 1–2 A10G/L40S-class
(24–48 GB) for concurrent inference + in-app training. The checkpoint is ~56 MB — this
is not an H100 conversation.

**"What is the NEXUS head?"**
Optional scaffolding for a fourth benchmark card calling an external foundation model
(Fundamental's Large Tabular Model) over a governed endpoint. Ships `off`/`stub` today;
`docs/nexus-ltm-design.md` has the design and cost runbook. Use it to show the harness
composes *any* FM, not just NVIDIA's.

## Failure playbook

| Symptom | Move |
| --- | --- |
| Badge shows `demo fallback` | `GET /api/status` gives the reason (checkpoint/artifacts/GPU). If unfixable in 2 min, pair the attendee up |
| Export fails at `check` stage | The GPU host-copy canary caught a broken stack — this is the readable-failure design working; use your instance, debug later |
| Storage connection fails | Data dialog → Save & test shows the probe error; `deploy/vast_probe.py` from a terminal for detail |
| Endpoint build slow/stuck | Fall back to the instructor's pre-deployed endpoint for the `curl` step |
| Whole room blocked on GPUs | Docker DEMO-FALLBACK path (localhost:8100): Ex 1–3 walkable with labelled synthetic scores; drop Ex 4 claims |

## After the session

- Share `workshop/` and the deeper docs: `docs/APP_GUIDE.md`, `docs/demo-script.md`
  (7-beat, ~10-min customer demo track), `docs/presentations/demo-runbook-fsi.md`.
- Suggested follow-up assignment: each SE runs the 10-minute `docs/demo-script.md` demo
  solo within a week and records their run history reaching tier 3+.
