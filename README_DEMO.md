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
