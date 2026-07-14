# SPDX-License-Identifier: Apache-2.0
"""Offline artifact export — runs notebook 04 (embeddings) AND notebook 05
(XGBoost) end-to-end, then serialises everything the live demo needs into
`demo_artifacts/`:

    preprocessor.joblib   sklearn OrdinalEncoder column transformer
    pca.joblib            PCA(512 -> 64)
    xgb_raw/embed/combined.joblib   the three XGBoost heads
    umap2d.joblib         cuML UMAP fitted on a test subsample (live projection)
    umap_background.json  scatter points {x, y, fraud} for the embedding map
    examples.json         real test transactions to click
    summary.json          headline test-set AUC / AP / lift

Embeddings (NB04) are generated in-app by running the decoder foundation model
over the temporal splits — no separate notebook run is required — and cached to
`data/embeddings/` so subsequent exports are fast. `run_export()` is importable
so the API can run it on the GPU backend and stream progress to the UI (see
tfm_demo/jobs.py); the root `export_for_demo.py` shim runs it from the CLI.

Prerequisites at runtime: the decoder-foundation-model checkpoint (config.MODEL_DIR
/ $MODEL_DIR), the blueprint's src/, and the temporal splits loaded into the
UI-configured storage target — an Impala database or VAST S3 objects
(scripts/prepare_data.py writes them; see tfm_demo/storage.py).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import numpy as np
import joblib

from .config import ARTIFACTS as OUT
from .config import (
    DATA_DIR, FRAUD_COL, MAX_LENGTH, MERCHANT_HASH_SIZE, MODEL_DIR, PCA_DIM,
    RAW_FEATURE_COLS, TOKENIZER_COLS,
)

EMBED_ROOT = DATA_DIR / "embeddings"


def _embed_dir():
    """Embedding cache dir, keyed by the configured data target (Impala
    database or VAST bucket/prefix). The cached sel/labels/embeddings are
    row-POSITION-keyed, so a cache built against one target's tables must never
    be reused against another's (prepare_data clears the current target's cache
    on re-ingest for the same reason)."""
    from . import storage
    return EMBED_ROOT / storage.cache_key()

# Per-split row cap for in-app embedding generation, so the UI "Build artifacts"
# button stays tractable (embedding the full multi-million-row dataset would take
# hours). Override with $EMBED_MAX_PER_SPLIT; the cached path always uses the
# rows that were embedded.
EMBED_MAX = int(os.environ.get("EMBED_MAX_PER_SPLIT", "20000"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "512"))

# XGBoost params copied verbatim from notebook 05.
XGB_PARAMS_RAW = dict(n_estimators=400, max_depth=8, learning_rate=0.0023,
                      colsample_bytree=0.95, min_child_weight=12, subsample=0.673,
                      reg_alpha=0.01, reg_lambda=0.001, random_state=42)
XGB_PARAMS_EMBED = dict(n_estimators=435, max_depth=12, learning_rate=0.03774,
                        colsample_bytree=0.587, min_child_weight=2.61, subsample=0.569,
                        reg_alpha=0.01364, reg_lambda=9.7e-05, gamma=1.7, random_state=42)
XGB_PARAMS_COMBINED = dict(n_estimators=512, max_depth=12, learning_rate=0.00305,
                           colsample_bytree=0.768, min_child_weight=25.85, subsample=0.65,
                           reg_alpha=0.01, reg_lambda=0.0001, gamma=4.8, random_state=42)

Progress = Optional[Callable[[str], None]]


# --------------------------------------------------------------------------- #
# data loading + feature engineering
# --------------------------------------------------------------------------- #
# Source columns read from each split. "Hour" is excluded — it is derived from
# "Time" below, not stored. Projecting columns keeps the GPU read lean.
_SOURCE_COLS = list(dict.fromkeys(
    [c for c in (*TOKENIZER_COLS, *RAW_FEATURE_COLS) if c != "Hour"]
    + [FRAUD_COL, "Merchant City"]
))


def _load_split(name: str):
    """Read a temporal split from the configured storage backend into a cuDF
    frame on the GPU — raw columns only, no feature engineering yet.

    Both backends pull the split in bounded chunks (host holds one chunk at a
    time) and push each straight to the GPU, in a stable write order (Impala:
    ORDER BY row_id; VAST: Parquet's inherent order) — the embedding cache and
    the sel arrays are keyed by row position. The whole split (up to
    PREP_TRAIN_CAP rows) then stays on the GPU — only the ~EMBED_MAX-row
    selections are later copied back to host (see run_export); materialising
    full splits on the host OOM-killed the 16 GB container in an earlier
    revision.
    """
    from . import storage
    return storage.read_split_cudf(name, _SOURCE_COLS)


def _engineer(gdf):
    """NB05 numeric coercions (Hour from Time, Amount "$…" -> float), on a cuDF
    frame. Run this on the ~EMBED_MAX-row selection, never the full split — the
    string ops materialise full-size temporaries, which blows the GPU budget on
    small cards (the deployed L4 has 24 GB) when done before selection."""
    gdf = gdf.copy()
    gdf["Hour"] = gdf["Time"].str.split(":", n=1, expand=True)[0].astype("int32")
    gdf["Amount"] = (gdf["Amount"].str.replace("$", "", regex=False)
                     .str.replace(",", "").astype("float64"))
    return gdf


def _labels(df) -> np.ndarray:
    """Binary fraud labels as a host numpy array (works on cuDF or pandas)."""
    m = ((df[FRAUD_COL] == "Yes") | (df[FRAUD_COL].astype(str) == "1")).astype("int32")
    # On non-null numeric columns BOTH .to_numpy() and .to_pandas() copy
    # device->host through numba.cuda (cuDF's values_host fast path) — the
    # path that segfaults when the numba driver shim is broken (live crashes
    # proved to_pandas() does NOT avoid it; see tfm_demo/gpu.py). to_arrow()
    # copies through libcudf/Arrow with no numba involvement.
    if hasattr(m, "to_arrow"):
        return m.to_arrow().to_numpy(zero_copy_only=False).astype("int32", copy=False)
    return np.asarray(m.to_numpy())


def _balanced_train_sel(train_df) -> np.ndarray:
    """NB05's balanced training subsample (~10% fraud), capped at EMBED_MAX.

    Returns positional indices (0..N-1) — equal to the cuDF RangeIndex labels, so
    `.loc[sel]` selects the same rows. Labels are pulled to host once (a single
    int column) rather than boolean-masking the GPU index, which keeps this
    independent of cuDF Index indexing quirks.
    """
    y = _labels(train_df)
    fraud_idx = np.nonzero(y == 1)[0]
    normal_idx = np.nonzero(y == 0)[0]
    target = min(EMBED_MAX, len(train_df))
    np.random.seed(42)
    n_fraud = min(len(fraud_idx), int(target * 0.1))
    n_normal = min(len(normal_idx), target - n_fraud)
    sel = np.concatenate([np.random.choice(fraud_idx, n_fraud, replace=False),
                          np.random.choice(normal_idx, n_normal, replace=False)])
    np.random.shuffle(sel)
    return sel


def _natural_sel(df, seed: int) -> np.ndarray:
    """Random subsample preserving the natural fraud rate (for val/test eval).

    Returns positional indices into the cuDF RangeIndex (label == position)."""
    n = len(df)
    if n <= EMBED_MAX:
        return np.arange(n)
    np.random.seed(seed)
    return np.sort(np.random.choice(n, EMBED_MAX, replace=False))


# --------------------------------------------------------------------------- #
# embeddings (notebook 04, in-app)
# --------------------------------------------------------------------------- #
def _build_inference(emit):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"decoder-foundation-model checkpoint missing at {MODEL_DIR}. "
            "In-app embedding generation needs it — run `python scripts/fetch_model.py` "
            "(or set $MODEL_DIR to an existing checkpoint)."
        )
    from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer
    from src.decoder_inference import HuggingFaceDecoderInference
    tokenizer = FinancialTabularTokenizer(
        merchant_hash_size=MERCHANT_HASH_SIZE,
        category_hierarchy=True, temporal_encoding=True,
    )
    inference = HuggingFaceDecoderInference(
        model_path=MODEL_DIR, tokenizer=tokenizer, pooling="last_token",
    )
    emit(f"Loaded foundation model from {MODEL_DIR}")
    return FinancialTokenizerPipeline, inference


def _embed_rows(tok_df, pipeline_cls, inference, emit) -> np.ndarray:
    """Tokenize a (cuDF) frame of raw transactions and extract last-token embeddings."""
    import torch

    # tok_df is already a cuDF frame (the selected rows) — feed the GPU tokenizer
    # directly, no host roundtrip.
    gdf = tok_df[TOKENIZER_COLS].reset_index(drop=True)
    pip = pipeline_cls(merchant_hash_size=MERCHANT_HASH_SIZE)
    gdf = pip.preprocess(gdf)
    pip.fit(gdf)
    padded = np.asarray(pip.encode(pip.transform(gdf), max_length=MAX_LENGTH))  # (N,128)

    out = []
    for i in range(0, len(padded), EMBED_BATCH):
        chunk = torch.from_numpy(padded[i:i + EMBED_BATCH])
        out.append(inference.extract_embeddings(chunk, return_numpy=True))
        emit(f"  embeddings {min(i + EMBED_BATCH, len(padded))}/{len(padded)}")
    return np.vstack(out)


# split keys (table/object names live in storage.SPLIT_TABLES) / RNG seed for
# the natural-rate subsample.
_SPLITS = ("train", "val", "test")
_SPLIT_SEED = {"val": 7, "test": 11}


def _meminfo() -> str:
    """Host RSS + GPU used/total, for per-stage memory logging."""
    parts = []
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts.append("host RSS " + line.split(":", 1)[1].strip())
                    break
    except Exception:                                              # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"GPU {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB")
    except Exception:                                              # noqa: BLE001
        pass
    return " · ".join(parts) or "mem n/a"


def _all_cached() -> bool:
    d = _embed_dir()
    return all((d / f"{s}_embeddings.npy").exists()
               and (d / f"{s}_labels.npy").exists()
               and (d / f"{s}_sel.npy").exists()
               for s in _SPLITS)


def _process_split(name, pipeline_cls, inference, emit) -> Dict:
    """Load ONE split, pick its rows, embed them, and return only small host
    arrays — freeing the full GPU frame before the next split. Peak memory is one
    split, not three (the old code held train+val+test resident at once, which
    OOM-killed the build). Embeddings cache to disk; raw features are re-derived
    from the cheaply re-read selection each run.

    Returns {emb (N,512), y (N,), sel (N,), raw (pandas N×RAW_FEATURE_COLS),
    rows (pandas full selected rows for the test split's examples, else None)}.
    """
    d = _embed_dir()
    emb_p = d / f"{name}_embeddings.npy"
    lab_p = d / f"{name}_labels.npy"
    sel_p = d / f"{name}_sel.npy"
    cached = emb_p.exists() and lab_p.exists() and sel_p.exists()

    emit(f"Loading {name} split from storage ...")
    df = _load_split(name)
    emit(f"  {name}: {len(df):,} rows loaded  [{_meminfo()}]")

    if cached:
        sel = np.load(sel_p)
    elif name == "train":
        sel = _balanced_train_sel(df)
    else:
        sel = _natural_sel(df, _SPLIT_SEED[name])
    emit(f"  {name}: selected {len(sel):,} rows; slicing ...  [{_meminfo()}]")

    # Slice to the selected rows and drop the full GPU frame immediately, THEN
    # engineer features — only the ~EMBED_MAX-row subset ever gets the string-op
    # temporaries, and the forward pass runs with just the subset + model resident.
    sub = df.loc[sel].reset_index(drop=True)
    del df
    emit(f"  {name}: engineering features ...  [{_meminfo()}]")
    # Tokenizer wants the raw columns (Amount still a "$..." string), captured
    # before we coerce numerics for the raw-feature head.
    tok_sel = sub[TOKENIZER_COLS].copy()
    df_sel = _engineer(sub)
    del sub
    emit(f"  {name}: features ready  [{_meminfo()}]")

    if cached:
        emit(f"  using cached embeddings for {name}")
        emb, lab = np.load(emb_p), np.load(lab_p)
    else:
        emit(f"Embedding {name} ({len(sel):,} rows) ...  [{_meminfo()}]")
        emb = _embed_rows(tok_sel, pipeline_cls, inference, emit)
        lab = _labels(df_sel)
        np.save(emb_p, emb)
        np.save(lab_p, lab)
        np.save(sel_p, np.asarray(sel))

    # Host copies via Arrow, not .to_pandas() — the latter routes non-null
    # numeric columns through numba.cuda (see _labels).
    raw = df_sel[RAW_FEATURE_COLS].to_arrow().to_pandas()
    rows = df_sel.to_arrow().to_pandas() if name == "test" else None
    return {"emb": emb, "y": lab, "sel": np.asarray(sel), "raw": raw, "rows": rows}


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def run_export(progress: Progress = None) -> Dict:
    """Generate embeddings, train the heads, fit PCA/UMAP, write artifacts."""
    emit = progress or (lambda _m: None)

    # Fail fast (before the GPU stack loads) if the splits aren't there.
    from . import storage
    emit(f"Checking training data ({storage.target()}) ...")
    ready, detail = storage.splits_ready()
    if not ready:
        raise RuntimeError(
            f"Training data not ready: {detail}. Open the Data dialog in the "
            "UI to configure the storage target and run the data load (or run "
            "scripts/prepare_data.py)."
        )
    emit(f"Splits ready: {detail}")

    from .gpu import configure_gpu_memory, gpu_stack_versions, host_copy_canary
    configure_gpu_memory()                 # RMM pool + cuDF spill, before any cuDF use
    emit(f"GPU stack: {gpu_stack_versions()}")

    # The cuDF device->host copy path (values_host -> numba.cuda) has
    # segfaulted on this container, and a SIGSEGV in this worker thread kills
    # the whole server. Probe it in a throwaway subprocess first so a broken
    # stack fails the export with a readable error instead.
    emit("Preflight: probing the cuDF device->host copy path ...")
    canary_failure = host_copy_canary()
    if canary_failure:
        raise RuntimeError(
            "cuDF device->host copy preflight crashed — the export would "
            f"segfault the server at its first .to_pandas(). Probe said: "
            f"{canary_failure}\n"
            "Knobs: DEMO_NUMBA_NV_BINDING=0 reverts numba to its ctypes driver "
            "shim (=1/default routes it through cuda-python); re-run the "
            "'Install dependencies' job to converge the pinned matrix in "
            "requirements-gpu.txt."
        )
    emit("Preflight OK — host copies are healthy.")

    import xgboost as xgb
    import torch
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Best-effort: have XGBoost allocate from the shared RMM pool instead of
    # carving its own arena (only takes effect on RMM-enabled builds).
    try:
        xgb.set_config(use_rmm=True)
    except Exception:                                              # noqa: BLE001
        pass

    OUT.mkdir(exist_ok=True)
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    emit(f"Compute device: {xgb_device}")

    # ---- per split: load -> select -> embed -> free (peak = ONE split) ----
    # The model is loaded once, only if some split still needs embedding.
    _embed_dir().mkdir(parents=True, exist_ok=True)
    pipeline_cls = inference = None
    if not _all_cached():
        emit("Generating embeddings in-app (notebook 04 step) ...")
        pipeline_cls, inference = _build_inference(emit)

    parts = {name: _process_split(name, pipeline_cls, inference, emit)
             for name in ("train", "val", "test")}
    X_train_e, y_train, sel_tr = parts["train"]["emb"], parts["train"]["y"], parts["train"]["sel"]
    X_val_e, y_val, sel_va = parts["val"]["emb"], parts["val"]["y"], parts["val"]["sel"]
    X_test_e, y_test, sel_te = parts["test"]["emb"], parts["test"]["y"], parts["test"]["sel"]

    # ---- PCA 512 -> 64 ----------------------------------------------------
    emit(f"PCA {X_train_e.shape[1]}d -> {PCA_DIM}d ...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    Xtr_pca = pca.fit_transform(X_train_e)
    Xva_pca = pca.transform(X_val_e)
    Xte_pca = pca.transform(X_test_e)

    # ---- raw tabular features (NB05), aligned to the embedded rows --------
    # Already host pandas, restricted to the ~EMBED_MAX selected rows per split.
    X_train_raw = parts["train"]["raw"]
    X_val_raw = parts["val"]["raw"]
    X_test_raw = parts["test"]["raw"]

    preproc = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough",
    )
    Xtr_raw = preproc.fit_transform(X_train_raw)
    Xva_raw = preproc.transform(X_val_raw)
    Xte_raw = preproc.transform(X_test_raw)
    n_raw = Xtr_raw.shape[1]

    # ---- train the three heads -------------------------------------------
    def fit(params, Xt, Xv, Xte, name):
        emit(f"Training {name} head ...")
        clf = xgb.XGBClassifier(**params, tree_method="hist", device=xgb_device,
                                early_stopping_rounds=20, eval_metric="auc")
        clf.fit(Xt, y_train, eval_set=[(Xv, y_val)], verbose=False)
        pt = clf.predict_proba(Xte)[:, 1]
        auc, ap = roc_auc_score(y_test, pt), average_precision_score(y_test, pt)
        emit(f"  {name}: AUC {auc:.4f} · AP {ap:.4f}")
        return clf, auc, ap

    clf_raw, auc_raw, ap_raw = fit(XGB_PARAMS_RAW, Xtr_raw, Xva_raw, Xte_raw, "raw")
    clf_emb, auc_emb, ap_emb = fit(XGB_PARAMS_EMBED, Xtr_pca, Xva_pca, Xte_pca, "embed")
    Xtr_c = np.hstack([Xtr_raw, Xtr_pca]); Xva_c = np.hstack([Xva_raw, Xva_pca])
    Xte_c = np.hstack([Xte_raw, Xte_pca])
    clf_comb, auc_comb, ap_comb = fit(XGB_PARAMS_COMBINED, Xtr_c, Xva_c, Xte_c, "combined")

    # ---- optional fourth head: NEXUS Large Tabular Model -------------------
    # Gets the UNTRANSFORMED raw frames (X_*_raw pandas, not the ordinal-
    # encoded Xtr_raw matrices) — the LTM's premise is raw tables in, no
    # feature engineering. Skipped without failing the export, like UMAP.
    from . import nexus
    nexus_meta = None
    if nexus.configured():
        try:
            emit(f"Training NEXUS head ({nexus.mode()} mode) ...")
            nexus_meta = nexus.fit_head(X_train_raw, y_train, X_val_raw, y_val,
                                        X_test_raw, y_test, progress=emit)
            emit(f"  nexus: AUC {nexus_meta['test_auc']:.4f} · "
                 f"AP {nexus_meta['test_ap']:.4f}"
                 + (" [stub]" if nexus_meta.get("stub") else ""))
        except Exception as exc:                                   # noqa: BLE001
            emit(f"(skipping NEXUS head: {exc})")
            nexus_meta = None
    else:
        emit("NEXUS not configured — skipping the Large Tabular Model head "
             "(set NEXUS_MODE=stub|live to enable).")

    # ---- UMAP (live projection + background scatter) ---------------------
    umap_bg = []
    try:
        from cuml.manifold import UMAP as cumlUMAP
        import cupy as cp
        viz_n = min(8000, len(X_test_e))
        np.random.seed(42)
        idx = np.random.choice(len(X_test_e), viz_n, replace=False)
        umap = cumlUMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        pts = cp.asarray(X_test_e[idx])
        umap.fit(pts)
        # The background MUST be projected with transform(), not taken from
        # fit_transform(): the live "This transaction" dot is placed with
        # umap.transform() at scoring time (engine.score), and cuML does not
        # guarantee transform() shares fit_transform()'s orientation — the
        # fitted embedding can come out mirrored, which stranded the live dot
        # on the opposite side of the map. One operator, one coordinate frame.
        coords = cp.asnumpy(umap.transform(pts))
        joblib.dump(umap, OUT / "umap2d.joblib")
        umap_bg = [{"x": float(c[0]), "y": float(c[1]), "fraud": int(y_test[idx][i])}
                   for i, c in enumerate(coords)]
        emit(f"UMAP fitted + {len(umap_bg)} background points")
    except Exception as exc:                                       # noqa: BLE001
        emit(f"(skipping UMAP: {exc})")

    # ---- real example transactions to click ------------------------------
    examples = []
    example_rows = []            # test-split row index of each example
    # The test split's selected rows (host pandas) for per-row .iloc access.
    test_raw_reset = parts["test"]["rows"]
    for want_fraud, label in [(1, "Real fraud (test set)"),
                              (0, "Real legitimate (test set)"),
                              (1, "Real fraud #2 (test set)")]:
        pool = np.where(y_test == want_fraud)[0]
        if len(pool):
            row = int(pool[len(pool) // 2])
            example_rows.append(row)
            r = test_raw_reset.iloc[row]
            examples.append({"label": label, "is_fraud": bool(want_fraud), "txn": {
                "Amount": f"${float(r['Amount']):.2f}", "Merchant Name": str(r["Merchant Name"]),
                "Merchant City": str(r["Merchant City"]), "Merchant State": str(r["Merchant State"]),
                "Use Chip": str(r["Use Chip"]), "MCC": int(r["MCC"]), "Zip": str(r["Zip"]),
                "Time": str(r["Time"]), "Year": int(r["Year"]), "Month": int(r["Month"]),
                "Day": int(r["Day"]), "Card": int(r["Card"]), "User": int(r["User"])}})

    # Diagnostic: stamp each example with where the BATCH-path embedding of
    # that exact row projects on the map. The UI overlays it as a hollow ring
    # so a live score of an untouched example shows the live-vs-batch drift
    # directly (a big gap = the single-row scoring pipeline embeds the row
    # differently than the export did — worth chasing in engine._embed_one).
    if umap_bg and example_rows:
        try:
            exp = cp.asnumpy(umap.transform(cp.asarray(X_test_e[example_rows])))
            for ex, c in zip(examples, exp):
                ex["expected_position"] = {"x": float(c[0]), "y": float(c[1])}
            emit("Examples stamped with their batch-path map positions")
        except Exception as exc:                                   # noqa: BLE001
            emit(f"(skipping example map positions: {exc})")

    # ---- write everything -------------------------------------------------
    emit(f"Writing artifacts to {OUT} ...")
    joblib.dump(preproc, OUT / "preprocessor.joblib")
    joblib.dump(pca, OUT / "pca.joblib")
    joblib.dump(clf_raw, OUT / "xgb_raw.joblib")
    joblib.dump(clf_emb, OUT / "xgb_embed.joblib")
    joblib.dump(clf_comb, OUT / "xgb_combined.joblib")
    if nexus_meta:
        (OUT / "nexus.json").write_text(json.dumps(nexus_meta, indent=2))
    else:
        # A stale nexus.json from an earlier stub/live export must not leak
        # into an export where the head didn't run.
        (OUT / "nexus.json").unlink(missing_ok=True)
    (OUT / "umap_background.json").write_text(json.dumps(umap_bg))
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2))

    def lift(x, base):
        return round((x - base) / base * 100, 2)

    summary = {
        "placeholder": False, "n_raw_features": int(n_raw), "pca_dim": PCA_DIM,
        "models": [
            {"key": "raw", "label": "Raw tabular features",
             "test_auc": round(auc_raw, 4), "test_ap": round(ap_raw, 4)},
            {"key": "embed", "label": "Foundation-model embeddings",
             "test_auc": round(auc_emb, 4), "test_ap": round(ap_emb, 4)},
            {"key": "combined", "label": "Combined",
             "test_auc": round(auc_comb, 4), "test_ap": round(ap_comb, 4)},
        ],
        "lift": {"embed_auc_pct": lift(auc_emb, auc_raw), "embed_ap_pct": lift(ap_emb, ap_raw),
                 "combined_auc_pct": lift(auc_comb, auc_raw),
                 "combined_ap_pct": lift(ap_comb, ap_raw)},
    }
    if nexus_meta:
        summary["models"].append({
            "key": nexus.NEXUS_KEY, "label": nexus.NEXUS_LABEL,
            "test_auc": nexus_meta["test_auc"], "test_ap": nexus_meta["test_ap"],
            "stub": bool(nexus_meta.get("stub")),
        })
        summary["lift"]["nexus_auc_pct"] = lift(nexus_meta["test_auc"], auc_raw)
        summary["lift"]["nexus_ap_pct"] = lift(nexus_meta["test_ap"], ap_raw)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(f"Done — artifacts written to {OUT}")
    emit(f"Lift (AP): embed {summary['lift']['embed_ap_pct']}% · "
         f"combined {summary['lift']['combined_ap_pct']}%")
    return summary
