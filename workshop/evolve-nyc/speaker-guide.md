# Evolve NYC · 200-Level Speaker Guide

## Session abstract (as published)

> Participants will get hands-on with a fraud-detection application on Cloudera AI.
> You'll learn the modeling approach behind it — why sequence-based foundation models
> outperform traditional point-in-time features, and how to benchmark them against a
> classic baseline. You'll score live transactions, retrain a model on GPU and see
> results update in real time, review diagnostics, and deploy a production-ready
> REST endpoint. By the end, you'll have walked the full lifecycle from data to
> deployment — and see how the same pattern applies to AML, credit risk, and
> customer churn.

Every sentence maps to a deliverable: live scoring → Lab Ex 1 · GPU retrain with
real-time updates → Lab Ex 2 + slide 15 · diagnostics → Lab Ex 3 · REST endpoint →
Lab Ex 4 + slide 16 · AML / credit risk / churn → slide 20.

**Deck:** [`TFM-Workshop-EvolveNYC-200.pptx`](TFM-Workshop-EvolveNYC-200.pptx)
**Audience:** business-technical — comfortable with concepts like "model", "API",
"GPU", but not practitioners. Sophomore (200) level: they know *what* these things
are; today they learn *how they fit together* and *why it matters commercially*.
**Length:** 50 min presentation + 10 min break + 60 min guided hands-on.

This is the 200-level variant of the SE workshop in
[`../part1-presentation/`](../part1-presentation/). Same 22-slide skeleton, same live
demo, same Miro diagrams — the language is rewritten for a business-technical room.

## Level rules (how this differs from the SE deck)

- **Say:** foundation model, behavioral fingerprint, governed data, one mainstream
  GPU, versioned API, tested on the future.
- **Don't say (unless asked):** decoder-only transformer, last-token pooling, cuDF,
  RMM, PCA, XGBoost hyperparameters, vocab sizes. The SE module files
  (`../part1-presentation/module-*.md`) hold all of that for Q&A.
- **Anchor every technical point to a business consequence** — "tested on future
  data" → "the number you see is the number production would see."

## Run of show

| Slides | Module | Time | The one thing the room must retain |
| --- | --- | --- | --- |
| 1–2 | Open + agenda | 3 min | This is a working system, not slideware — we'll prove a number live. |
| 3–6 | Why a new approach | 10 min | Rules judge one transaction; the model judges it against the customer's own history. |
| 7–11 | How it works | 15 min | Transaction → behavioral fingerprint → score, in milliseconds; a fair bake-off vs. "your current model" measures the lift honestly. |
| 12–16 | Data to deployment | 15 min | Governed data in, governed API out — retraining is one click on one affordable GPU, and every improvement is on the record. |
| 17–21 | What it takes to run | 7 min | Installs like an app on Cloudera AI; the pattern reuses beyond fraud (AML, credit risk, churn). |
| 22 | Break | — | See you in the cockpit. |

## Per-slide talking points

- **S4 (ceiling):** land the bottom line first: *"a $40 online purchase at 3 a.m. is
  meaningless in isolation — against this cardholder's history it can be a screaming
  anomaly."* Everything else on the slide is elaboration.
- **S5 (what is a TFM):** the ChatGPT analogy carries the whole slide — "same idea,
  but it reads transactions instead of words." The three stat cards say: real public
  dataset, small model, and the output is a 512-number fingerprint.
- **S6 (the claim):** the promise you'll cash in the lab. If asked why "Average
  Precision": it counts *fraud caught per 100 analyst reviews*, which is what a fraud
  team actually staffs for.
- **S8 (inference path):** walk TRANSLATE → UNDERSTAND → SCORE. The Miro diagram
  above the cards is the engineer's view of the same three steps — point at it,
  don't read it.
- **S9 (512 → 64):** one sentence is enough: a standard compression step that makes
  everything downstream cheaper, and the headline number is measured *after* it.
- **S10–11 (bake-off):** stress fairness — same data, same procedure; only what each
  model *sees* changes. "Combined" is the adoption story: nobody discards their
  current model on day one.
- **S13 (governed data):** the model trains where the data already lives, governed —
  no CSV on a laptop. Train on the past, test on the future.
- **S14 (one GPU):** the cost message: engineered for the NVIDIA L4 a team can
  actually get approved, not an 80 GB flagship.
- **S15 (instant updates):** two beats — no maintenance window, and a logged history
  that survives "prove the model got better."
- **S16 (register/deploy):** governance is the differentiator: versioned model,
  secured API, paper trail from prediction back to the data.
- **S18 (installs like an app):** five automated steps, no integration project;
  demo mode means the walkthrough never depends on hardware being free.
- **S20 (beyond fraud):** the method is the reusable asset — swap the data, keep
  the method. This is the slide execs remember.
- **S21 (lab preview):** set expectations: point-and-click, checkpoints, no coding.
  The lab guide is [`../part2-lab/lab-guide.md`](../part2-lab/lab-guide.md).

## Diagrams

The four diagram slides embed live captures of the shared Miro board
(<https://miro.com/app/board/uXjVHsmGr9A=/>); each image and its "Diagram on Miro ↗"
caption link to the exact frame. The diagram labels are the engineer-level names —
in this session, gesture at the flow and use the slide's plain-language cards for
the narration.
