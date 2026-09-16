# Module 4 — Platform Architecture & Lab Preview (10 min)

**Goal:** attendees know exactly what they'll click in the lab, and can generalize the
Cloudera + NVIDIA pattern beyond fraud in customer conversations.

---

## Slide 4.1 — Native Cloudera AI environment (3 min)

**On the slide**

- Ships as a **one-click Applied ML Prototype (AMP)**: `.project-metadata.yaml` declares
  five tasks on an NVIDIA GPU runtime (JupyterLab / Python 3.12 / Nvidia GPU edition):

| # | Task | What it does |
| --- | --- | --- |
| 1 | `install_deps` | Pinned GPU-aware Python stack (RAPIDS cu12, torch cu121, transformers, XGBoost) |
| 2 | `build_frontend` | User-local Node 20 + Vite build of the React SPA |
| 3 | `fetch_model` | TFM checkpoint (~56 MB) + NVIDIA blueprint tokenizer package |
| 4 | `prepare_data` | TabFormer → temporal splits → Impala / S3 Parquet |
| 5 | `TFM Fraud Demo` | The application: FastAPI backend (private :7100) + SPA (public :8100) |

- Runs *inside* Cloudera AI Workbench: workload auth (`CDSW_APIV2_KEY`) drives registry
  calls; the app binds CML's application port; jobs are CML jobs.

**Speaker notes**

- "Native" means no sidecar infrastructure: one CML application container runs both the
  GPU inference engine and the SPA; CML exposes a single authenticated URL. The AMP is
  the repeatable deployment artifact an SE can stand up in a customer workspace.
- Graceful degradation is a feature to sell: without a GPU or checkpoint the app boots
  in a clearly-labelled **DEMO-FALLBACK** mode (`docker compose up` → localhost:8100),
  so the UI always demos — but today's lab runs the REAL GPU path.

---

## Slide 4.2 — NVIDIA-enhanced performance on a single L4 (2 min)

**On the slide**

- The full workload — embedding generation, PCA, 3× XGBoost training, cuML UMAP, and
  live per-transaction inference — runs on **one NVIDIA L4 (24 GB)**.
- Engineering that makes it fit: RMM memory pool shared across libraries, cuDF spill to
  host, streaming one split at a time, batch-512 embedding, PyTorch expandable segments.
- Sizing guidance: demo = 1× L4-class GPU; production = 1–2× A10G/L40S-class for
  concurrent inference + in-app training.

**Speaker notes**

- The message for cost-sensitive buyers: foundation-model economics here are *mundane* —
  a ~56 MB model on a commodity inference GPU, not an H100 cluster. GPU spend goes to
  throughput, not to making the demo possible.
- In the lab they will watch the GPU meters live during their own training run.

---

## Slide 4.3 — Generalizing beyond fraud (2 min)

**On the slide**

The pattern — *pretrained sequence embeddings + your existing tabular features + a
classical head, benchmarked side-by-side, governed on Cloudera AI* — transfers directly:

| Use case | Sequence being modeled |
| --- | --- |
| **AML** | Transaction/counterparty flows per entity |
| **Credit risk** | Repayment & utilization histories |
| **Payments abuse** | Merchant/consumer interaction streams |
| **Churn** | Customer engagement event sequences |

**Speaker notes**

- The demo's methodology is the reusable asset: temporal splits, frozen-hyperparameter
  head comparison, AP-first evaluation, budget ladder, registry lineage. Swap the
  dataset and tokenizer; keep the harness.
- The NEXUS fourth-head scaffolding shows the same harness composing an *external*
  foundation-model endpoint into the benchmark — the pattern for "bring your own FM."

---

## Slide 4.4 — Lab preview: what you will do in 60 minutes (3 min)

**On the slide**

1. **Score live transactions** — real GPU path (tokenizer → decoder → PCA → heads),
   watch three heads disagree, see your transaction land on the UMAP map.
2. **Trigger a GPU training run** — watch the staged pipeline and live GPU meters; see
   the model **hot-reload** and your next score change.
3. **Analyze diagnostics & embedding clusters** — validation curves, class separation,
   raw-vs-embedding feature importance, run-over-run lift.
4. **Register & deploy** — push your run to the Cloudera AI Model Registry and stand up
   a governed REST endpoint; score it with `curl`.

**Speaker notes**

- Set expectations on timing: a tier-1 training run (4,000 rows/split) takes a few
  minutes on the L4 — designed for a lab, not a lunch break. Bigger tiers are for the
  ambitious.
- Logistics to state before the break: workshop URL(s) for each attendee's app
  instance, and where the lab guide lives (`workshop/part2-lab/lab-guide.md`).
- Cliffhanger into the break: "When we come back, everything from the last 50 minutes
  becomes buttons you press."
