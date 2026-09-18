# Lab 0 — Instructor Setup (complete BEFORE the session)

Budget ~90 minutes the day before (dependency install + dataset download dominate).
The lab assumes each attendee (or pair) has a running instance of the app in REAL GPU
mode with data already loaded. **Do not** make attendees run `prepare_data` live — the
~2.4 GB TabFormer download is the one step that can blow the schedule.

## 1. Provision

Per attendee instance (or one shared instance for a guided lab):

- Cloudera AI (CML) workspace with an **NVIDIA GPU ML Runtime** (JupyterLab /
  Python 3.12 / Nvidia GPU edition, CUDA 12 driver). L4-class (24 GB) or better.
- Impala virtual warehouse reachable from the workspace **or** S3-compatible object
  storage (VAST/MinIO) credentials.
- Outbound HTTPS (TFM checkpoint from GitHub; TabFormer from IBM Box).

Deploy the repo as an AMP — the five tasks in `.project-metadata.yaml` run in order:
`install_deps` → `build_frontend` → `fetch_model` → `prepare_data` → application
(`TFM Fraud Demo`). Or run the equivalent manual sequence from the repo README.

Set AMP variables at create time if using Impala: `IMPALA_CONNECTION_NAME`,
`IMPALA_DATABASE`. For S3/VAST, put credentials in an untracked `.vast.env` or enter
them in the app's Data dialog after boot (see `docs/impala-vast-s3-config.md`).

## 2. Load data

In each instance: **Data** dialog → pick backend → **Save & test connection** (must go
green) → **Load TabFormer →**. Watch the streaming log to completion; note the printed
per-split row counts (~1M train capped / ~100K val / ~100K test).

## 3. Prime the demo state

1. Confirm the header badge reads **`live · GPU`** (not `demo fallback`).
2. Run **one** training run (Build artifacts / "Train next run") to completion so the
   Inference tab has real metrics and a UMAP background when attendees arrive.
3. Then click **Reset demo** (Model Lifecycle tab, two-click confirm) if you want
   attendees to experience run 1 → run 2 lift themselves — reset clears run history and
   replays the budget ladder from tier 1 (4,000 rows/split), while the trained artifacts
   from your priming run keep the Inference tab populated.
4. Registry: verify `GET /api/registry` shows available (needs `CDSW_API_URL`,
   `CDSW_APIV2_KEY`, `CDSW_PROJECT_ID` + `cmlapi` — automatic inside CML). If the lab
   includes Exercise 4, confirm the workspace has capacity for one small CML Model
   deployment (2 CPU / 4 GB) per instance that will deploy.

## 4. Dry-run timings on YOUR hardware

Record these the day before and adjust the lab plan:

| Step | Expected on L4 | Yours |
| --- | --- | --- |
| Tier-1 training run (4,000 rows/split) | ~3–6 min | |
| Tier-2 run (8,000 rows/split) | ~2× tier 1 | |
| Register to Model Registry | < 1 min | |
| Deploy CML Model endpoint (build + deploy) | ~5–10 min | |

If endpoint deployment is slow in your workspace, deploy one instance's endpoint
yourself in advance and demo Exercise 4's `curl` against it as a fallback.

## 5. Fallback plan

- **No GPU available on the day**: `docker compose up --build` → http://localhost:8500
  runs the full UI in clearly-labelled DEMO-FALLBACK mode. Exercises 1–3 remain
  walkable (synthetic scores); skip the retrain-lift claims and Exercise 4.
- **One instance dies mid-lab**: pair the attendee with a neighbor; state is per-project
  so nothing is shared.
- Connectivity triage commands: `python deploy/vast_probe.py` (object storage),
  `GET /api/status` (engine mode + reason), export log in the Build-artifacts dialog.

## 6. Handouts / links to circulate

- Lab guide: `workshop/part2-lab/lab-guide.md`
- Glossary: `workshop/glossary.md`
- Each attendee's application URL (CML application page)
