# TFM Live Demo — Fraud Inference Cockpit

A single-screen web demo that runs **live inference against the real
Transaction Foundation Model checkpoint**. You compose (or load) a transaction,
and watch it flow through the actual blueprint pipeline:

```
raw txn → GPU tokenizer → decoder foundation model → 512-d embedding
        → PCA-64 → 3 XGBoost heads (raw / embeddings / combined)
        → fraud probabilities + lift + UMAP position
```

The headline story is built in: the **embeddings head** and **combined head**
score the same transaction next to the **raw-features baseline**, with the
test-set AUC/AP lift shown up top — the exact narrative from notebook 05, made
clickable.

## Layout

```
tfm-demo/
├── app.py                  # FastAPI backend (live inference)
├── export_for_demo.py      # run once after NB04 + NB05 to dump artifacts
├── requirements-demo.txt   # fastapi/uvicorn/joblib (rest is in the NeMo container)
├── static/index.html       # the cockpit UI
└── demo_artifacts/         # created by export_for_demo.py
```

Place this `tfm-demo/` folder **inside the blueprint repo root** (next to
`models/`, `src/`, and the notebooks) so it can import `src/` and find the
checkpoint.

## Setup (inside the NeMo container)

1. Launch the container and run notebooks **04** then **05** as the blueprint
   README describes, so `data/embeddings/` is populated and the checkpoint is
   pulled via `git lfs pull`.

2. Export the trained heads, PCA, encoder, UMAP, metrics, and real examples:
   ```bash
   cd <blueprint repo root>
   python tfm-demo/export_for_demo.py
   ```
   This writes everything into `tfm-demo/demo_artifacts/` and prints the lift
   numbers. It reuses the *exact* XGBoost params and feature engineering from
   notebook 05, so the demo numbers match the notebook.

3. Install the demo-layer deps and start the server:
   ```bash
   pip install -r tfm-demo/requirements-demo.txt
   python tfm-demo/app.py
   ```
   Open `http://localhost:8000` (forward the port with
   `ssh -L 8000:localhost:8000 user@host` if you're on a remote GPU box).

## Modes

The top-left badge always tells you what's running:

- **REAL** — checkpoint + tokenizer + XGBoost heads loaded; every score is a
  live forward pass through the decoder. This is what you demo.
- **DEMO-FALLBACK** — no GPU or `demo_artifacts/` missing. The UI still works
  with clearly-labelled synthetic scores so you can build/preview the front end
  off-GPU (e.g. on a laptop). Nothing is ever silently faked.

## Training data · storage (VAST S3 or Impala)

The temporal splits `train`, `val_eval` and `test_eval` live in a storage
target you pick in the **Data** dialog (header button) — not in local files.
Two backends (`tfm_demo/storage.py` dispatches):

- **VAST S3** (default when configured) — each split is one Parquet object at
  `s3://<bucket>/<prefix>/<split>.parquet`, written and read **directly with
  boto3**. No warehouse in the path — this exists because the CDW/Impala s3a
  client is rejected by the VAST endpoint with a bare `400` while the same S3
  calls succeed via boto3 (see `docs/impala-vast-s3-config.md` and
  `scripts/vast_probe.py`). Enter endpoint / bucket / folder / keys in the
  dialog (`$VAST_ENDPOINT`, `$VAST_BUCKET`, `$VAST_PATH`, `$VAST_ACCESS_KEY`,
  `$VAST_SECRET_KEY` seed the defaults — the same env vars the probe script
  uses). Path-style addressing and sigv4 are hard-wired; TLS verification is
  off for the internal-CA endpoint unless `$VAST_VERIFY_SSL=1` (or a CA bundle
  path). Uploads are parallel twice over — the three splits upload
  concurrently, and each object bigger than one part is a concurrent multipart
  upload — and resilient: TCP keepalive, generous timeouts, and whole-upload
  retries with backoff (the endpoint has been seen dropping TLS mid-part).
  Knobs: `$VAST_UPLOAD_PART_MB` (default 8, clamped to S3's 5 MiB minimum),
  `$VAST_UPLOAD_CONCURRENCY` (default 8), `$VAST_UPLOAD_RETRIES` (default 3).
  Trailing checksums (botocore ≥ 1.36's aws-chunked upload default) are
  disabled because this VAST release can't parse them; set
  `$VAST_TRAILING_CHECKSUMS=1` only against a store that supports them.
  **Broken-endpoint fallback:** the live previewhub VAST hangs on any upload
  body over exactly 60 KiB (server-side defect — probed and reproduced with
  curl over HTTP/1.1 and HTTP/2; GETs are healthy). Uploads probe the ceiling
  once per run and automatically fall back to concurrent ≤60 KiB multipart
  parts with per-part retries — slow (~10 min/MB observed) but reliable.
  `$VAST_MAX_BODY_KB` overrides the probe (0 = force the normal fast path,
  N = force N-KiB parts). Since that VAST also drops user metadata on
  multipart objects, tiny-part uploads leave a `<split>.parquet.rows` sidecar
  that the status check falls back to.
  On a slow/flaky link try `VAST_UPLOAD_PART_MB=5` (smaller parts finish
  before proxy timeouts) and lower concurrency; if multipart itself is the
  problem, a large part size (e.g. 64) forces single PUTs.
- **Impala (CDW)** — splits as Impala tables through a **CML data connection**
  (`$IMPALA_CONNECTION_NAME` / `$IMPALA_DATABASE` seed the defaults; outside
  CML, `$IMPALA_HOST` + `_PORT/_USER/_PASSWORD/_AUTH/_SSL/_HTTP` bypasses the
  connection with impyla directly).

The flow is the same either way:

1. Open the **Data** dialog, pick the backend, fill in the target, *Save &
   test* — the dialog shows per-split row counts. Settings persist in
   `.data_settings.json` (mode 0600; it can hold the VAST secret key).
2. Click **Load TabFormer → …** — downloads the ~2.4 GB TabFormer dump,
   rebuilds NB01's temporal split in chunked pandas, and writes the three
   splits (`scripts/prepare_data.py`, also runnable as the CML job).
3. **Build artifacts** (training) then reads the splits back into cuDF on the
   GPU.

Details that matter: reads are row-position-stable (Impala: a `row_id` column
+ `ORDER BY row_id`; VAST: Parquet's inherent row order) because the embedding
cache is keyed by row position; re-ingesting a target clears its embedding
cache under `data/embeddings/<cache-key>/`. On the Impala backend TabFormer's
column names are mapped to snake_case (`Is Fraud?` → `is_fraud`) and mapped
back on read; Parquet keeps the original names.

## Demo flow (suggested)

1. Point at the metrics strip: baseline AUC/AP vs the foundation-model head, and
   the **+X% lift** — "no hand-crafted features, the model learned this from raw
   sequences."
2. Click **Real fraud (test set)** → Run inference. The three heads fill in;
   the combined head flags it, and the point lands inside the red fraud cluster
   on the embedding map.
3. Edit a field live — flip channel to *Online* or push the amount up — and
   re-run to show the score and map position move.

## Customising

- **More / different examples:** edit `demo_artifacts/examples.json` (or change
  the selection logic in `export_for_demo.py`).
- **Different decision threshold:** the UI flags at P(fraud) ≥ 0.5; change it in
  `static/index.html` (`decided = sc.combined >= 0.5`).
- **Bigger embedding map:** raise `viz_n` in `export_for_demo.py`.
- **Branding / colours:** the CSS variables at the top of `index.html`
  (`--signal`, `--amber`, fonts) are the whole theme.

Built on the NVIDIA AI Blueprint *Transaction Foundation Model* (Apache-2.0).
