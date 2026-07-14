# NEXUS (Fundamental Large Tabular Model) — fourth model card

**Status:** scaffolding merged, live transport pending Fundamental access.
The stub path works today; the real SDK wiring is a contained day-one task
(see the checklist at the bottom).

## Why

The demo's story is a three-way comparison — classic GBM on raw features vs.
XGBoost on TFM embeddings vs. both combined. Fundamental's NEXUS (the "Large
Tabular Model" launched Feb 2026: pre-trained tabular foundation model,
sklearn-style `NEXUSClassifier` SDK, SageMaker-hosted via AWS Marketplace)
extends that to a four-way story: **classic GBM vs. TFM-embeddings vs. Large
Tabular Model**. NEXUS deliberately scores the *raw untransformed transaction*
(the 13 `RAW_FEATURE_COLS`) — its pitch is raw tables in, no feature
engineering — so it skips the ordinal-encoder and PCA paths entirely.

## Architecture

Everything NEXUS lives in **`tfm_demo/nexus.py`**; nothing else imports the
SDK or reads `NEXUS_*` env vars. Three modes via `$NEXUS_MODE`:

| mode | behavior |
|---|---|
| `off` (default) | head doesn't exist; API payloads byte-identical to the pre-NEXUS app |
| `stub` | deterministic fake scores + canned metrics, flagged `"stub": true` end-to-end — demos the full path with zero AWS |
| `live` | real SDK against a pre-deployed SageMaker endpoint |

The mode is switchable from the UI (Build-artifacts dialog → off/stub/live
selector → `GET/POST /api/nexus`), persisted in `.nexus_settings.json`
(gitignored). A UI choice wins over `$NEXUS_MODE`; both export and scoring
read the mode per use, so changes apply without a restart. The rest of the
config (endpoint, bucket, timeouts) stays env-only — live mode's selector
button is locked until `NEXUS_ENDPOINT_NAME` + `NEXUS_S3_BUCKET` are set.

Integration points (all additive, all skip cleanly when unconfigured):

- **Export** (`tfm_demo/export.py`): after the three XGBoost heads, `nexus.fit_head()`
  gets the untransformed `X_*_raw` pandas frames and is evaluated on the same
  test rows (same AUC/AP, same raw-head lift baseline). Failure or absence →
  one log line, export unaffected. Writes `demo_artifacts/nexus.json` (model
  ref + metrics) when it ran, unlinks it when it didn't; `summary.json` gains a
  4th `models` entry + `nexus_auc_pct`/`nexus_ap_pct` lift keys only then.
- **Engine** (`tfm_demo/engine.py`): `score()` fires `nexus.score_one()` on a
  2-thread pool *before* the local pipeline runs and harvests it after, with a
  hard budget (`$NEXUS_SCORE_TIMEOUT_S`, default 2.5 s) — the demo never blocks
  on the remote call. Response gains `scores.nexus: number|null` plus a
  `nexus: {status: ok|timeout|unavailable, latency_ms}` side-channel, present
  only when configured. Real mode never gates on NEXUS (`_load_real_stack`
  untouched); the stub head also rides on demo-fallback mode.
- **UI**: `ModelHeads` renders a 4th purple bar when `scores` carries the key
  (null-safe: `—` + "timed out"/"unavailable" caption; shows remote latency).
  `MetricsStrip` is now summary-driven and renders whatever models the export
  produced, with a `· stub` tag when applicable. `ExportDialog` shows NEXUS
  lift stats when present.
- **`/api/status`** reports `"nexus": off|stub|live`.

## Configuration (env-only; AWS creds via boto3's standard chain)

```
NEXUS_MODE             off | stub | live            (default off)
NEXUS_ENDPOINT_NAME    pre-deployed SageMaker endpoint (required for live)
NEXUS_REGION           falls back to $AWS_DEFAULT_REGION
NEXUS_S3_BUCKET        SDK staging bucket             (required for live)
NEXUS_S3_PREFIX        staging prefix                 (default nexus-staging)
NEXUS_SCORE_TIMEOUT_S  live-scoring budget            (default 2.5)
NEXUS_FIT_TIMEOUT_S    fit/adapt polling ceiling      (default 3600)
NEXUS_TRAIN_MAX        training rows shipped to fit   (default 20000)
```

Deliberately NOT in `.data_settings.json` / the Data dialog: NEXUS is a model
backend, not a data backend, and only the server process reads it.

## Cost runbook (important)

The launch deployment is a **single-tenant `ml.p5en.48xlarge` (8× H200), on
the order of $60+/hr before the Marketplace software fee**. No code path in
this app ever creates or deletes an endpoint. Operating procedure:

1. Deploy the endpoint from SageMaker JumpStart **before the demo window**.
2. Run the export (`NEXUS_MODE=live`) and the demo.
3. **Delete the endpoint immediately after.** An InService endpoint bills
   whether or not it serves traffic.

## Day-one-of-access checklist

Run `scripts/nexus_probe.py` first — each step isolates one unknown below.

1. **Exact pip package name/version** → fill `requirements-nexus.txt`, fix the
   import inside `nexus._sdk()` (one place); decide whether
   `scripts/install_deps.py` should install it.
2. **Single-row predict latency** (probe step 6, p50/p95) → set the default
   `NEXUS_SCORE_TIMEOUT_S` honestly. If p95 > ~3 s, keep null-on-timeout and
   consider pre-scoring `examples.json` at export time as a follow-up.
3. **`fit()` semantics** — fine-tune (persistent model ref) or in-context
   (training data must ride along at predict time)? → implement `nexus._fit`
   accordingly; whatever scoring needs later must land in the returned ref
   dict (persisted as `nexus.json`, handed back via `score_one(meta=)`).
4. **Transport** — real-time InvokeEndpoint or S3-batch only? → implement
   `nexus._predict_frame`; those two private helpers are the *only* intended
   change surface.
5. **Input contract** for the 13 columns (Amount float vs `"$…"` string, NaN
   Zip, int64 hashed Merchant Name, column names with spaces — probe step 7)
   → align `engine._raw_frame` / export frames if needed.
6. **`predict_proba` orientation** — which column is the positive class.
7. **IAM from CML**: `sagemaker:InvokeEndpoint` + staging-bucket RW; confirm
   the credential chain resolves inside the container (probe step 2).
8. **Cost controls**: confirm idle billing, write the start/stop runbook into
   the demo checklist; validate `NEXUS_TRAIN_MAX` fit cost/time.
9. **Real export** with `NEXUS_MODE=live`; sanity-check AUC/AP against the raw
   XGBoost baseline before demoing the lift numbers.

## Verifying the scaffolding (no access needed)

- `NEXUS_MODE` unset → export log shows one skip hint; `summary.json`, score
  payloads, and UI identical to before.
- `NEXUS_MODE=stub` → export trains a fake head (~2 s), writes `nexus.json`,
  4-model summary with `"stub": true`; the UI shows a 4th purple score bar and
  a `large tabular model · stub` metrics card; `/api/score` carries
  `scores.nexus` + `nexus.status`.
- Timeout path: set `NEXUS_SCORE_TIMEOUT_S=0` → bar shows `—` / "timed out".
