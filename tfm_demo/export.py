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
/ $MODEL_DIR), the blueprint's src/, and the temporal parquet splits under
config.DATA_DIR.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import joblib

from .config import ARTIFACTS as OUT
from .config import (
    DATA_DIR, FRAUD_COL, MAX_LENGTH, MERCHANT_HASH_SIZE, MODEL_DIR, PCA_DIM,
    RAW_FEATURE_COLS, TOKENIZER_COLS,
)

EMBED_DIR = DATA_DIR / "embeddings"
TEMPORAL_DIR = DATA_DIR / "TabFormer" / "temporal_split"

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
    """Read a temporal parquet into a cuDF frame on the GPU and engineer features
    there; return (tokenizer_input, engineered) as cuDF frames.

    The whole split (up to PREP_TRAIN_CAP rows) stays on the GPU — only the
    ~EMBED_MAX-row selections are later copied to host (see run_export). The old
    code did `.to_pandas()` on the *full* frame for all three splits at once,
    which materialised millions of Python strings and OOM-killed the process on
    the 16 GB host container before the model even loaded. Keeping the heavy
    frames in cuDF fixes that and keeps the pipeline GPU-native (RAPIDS).
    """
    import cudf
    gdf = cudf.read_parquet(str(TEMPORAL_DIR / name), columns=_SOURCE_COLS)
    # Tokenizer wants the raw columns (Amount still a "$..." string), captured
    # before we coerce numerics for the raw-feature head.
    tok = gdf[TOKENIZER_COLS].copy()
    gdf = gdf.copy()
    gdf["Hour"] = gdf["Time"].str.split(":", n=1, expand=True)[0].astype("int32")
    gdf["Amount"] = (gdf["Amount"].str.replace("$", "", regex=False)
                     .str.replace(",", "").astype("float64"))
    return tok, gdf


def _labels(df) -> np.ndarray:
    """Binary fraud labels as a host numpy array (works on cuDF or pandas)."""
    m = (df[FRAUD_COL] == "Yes") | (df[FRAUD_COL].astype(str) == "1")
    return np.asarray(m.astype("int32").to_numpy())


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


def _embeddings(splits, emit) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each split return (embeddings, labels, selection), cached on disk.

    `splits` maps name -> (tok_df, engineered_pdf, selection_indices).
    """
    cached = all((EMBED_DIR / f"{s}_embeddings.npy").exists()
                 and (EMBED_DIR / f"{s}_sel.npy").exists() for s in splits)
    if cached:
        emit(f"Using cached embeddings in {EMBED_DIR}")
        return {s: (np.load(EMBED_DIR / f"{s}_embeddings.npy"),
                    np.load(EMBED_DIR / f"{s}_labels.npy"),
                    np.load(EMBED_DIR / f"{s}_sel.npy")) for s in splits}

    emit("Generating embeddings in-app (notebook 04 step) ...")
    pipeline_cls, inference = _build_inference(emit)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for s, (tok, pdf, sel) in splits.items():
        emit(f"Embedding {s} ({len(sel)} rows) ...")
        emb = _embed_rows(tok.loc[sel], pipeline_cls, inference, emit)
        lab = _labels(pdf.loc[sel])
        np.save(EMBED_DIR / f"{s}_embeddings.npy", emb)
        np.save(EMBED_DIR / f"{s}_labels.npy", lab)
        np.save(EMBED_DIR / f"{s}_sel.npy", np.asarray(sel))
        result[s] = (emb, lab, np.asarray(sel))
    return result


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def run_export(progress: Progress = None) -> Dict:
    """Generate embeddings, train the heads, fit PCA/UMAP, write artifacts."""
    emit = progress or (lambda _m: None)

    import xgboost as xgb
    import torch
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score

    OUT.mkdir(exist_ok=True)
    if not TEMPORAL_DIR.exists():
        raise FileNotFoundError(
            f"temporal splits missing at {TEMPORAL_DIR} — set $DATA_DIR or place "
            "the TabFormer temporal_split parquets there."
        )
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    emit(f"Compute device: {xgb_device}")

    # ---- load splits + choose the rows each split will use ----------------
    emit("Loading temporal parquets + feature engineering ...")
    tok_tr, train_pdf = _load_split("train.parquet")
    tok_va, val_pdf = _load_split("val_eval.parquet")
    tok_te, test_pdf = _load_split("test_eval.parquet")
    splits = {
        "train": (tok_tr, train_pdf, _balanced_train_sel(train_pdf)),
        "val":   (tok_va, val_pdf, _natural_sel(val_pdf, 7)),
        "test":  (tok_te, test_pdf, _natural_sel(test_pdf, 11)),
    }

    # ---- embeddings (NB04, in-app + cached) -------------------------------
    emb = _embeddings(splits, emit)
    X_train_e, y_train, sel_tr = emb["train"]
    X_val_e, y_val, sel_va = emb["val"]
    X_test_e, y_test, sel_te = emb["test"]

    # ---- PCA 512 -> 64 ----------------------------------------------------
    emit(f"PCA {X_train_e.shape[1]}d -> {PCA_DIM}d ...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    Xtr_pca = pca.fit_transform(X_train_e)
    Xva_pca = pca.transform(X_val_e)
    Xte_pca = pca.transform(X_test_e)

    # ---- raw tabular features (NB05), aligned to the embedded rows --------
    # Copy only the selected rows (~EMBED_MAX) to host pandas for sklearn.
    X_train_raw = train_pdf.loc[sel_tr, RAW_FEATURE_COLS].reset_index(drop=True).to_pandas()
    X_val_raw = val_pdf.loc[sel_va, RAW_FEATURE_COLS].reset_index(drop=True).to_pandas()
    X_test_raw = test_pdf.loc[sel_te, RAW_FEATURE_COLS].reset_index(drop=True).to_pandas()

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

    # ---- UMAP (live projection + background scatter) ---------------------
    umap_bg = []
    try:
        from cuml.manifold import UMAP as cumlUMAP
        import cupy as cp
        viz_n = min(8000, len(X_test_e))
        np.random.seed(42)
        idx = np.random.choice(len(X_test_e), viz_n, replace=False)
        umap = cumlUMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        coords = cp.asnumpy(umap.fit_transform(cp.asarray(X_test_e[idx])))
        joblib.dump(umap, OUT / "umap2d.joblib")
        umap_bg = [{"x": float(c[0]), "y": float(c[1]), "fraud": int(y_test[idx][i])}
                   for i, c in enumerate(coords)]
        emit(f"UMAP fitted + {len(umap_bg)} background points")
    except Exception as exc:                                       # noqa: BLE001
        emit(f"(skipping UMAP: {exc})")

    # ---- real example transactions to click ------------------------------
    examples = []
    # Selected test rows to host pandas for per-row .iloc element access.
    test_raw_reset = test_pdf.loc[sel_te].reset_index(drop=True).to_pandas()
    for want_fraud, label in [(1, "Real fraud (test set)"),
                              (0, "Real legitimate (test set)"),
                              (1, "Real fraud #2 (test set)")]:
        pool = np.where(y_test == want_fraud)[0]
        if len(pool):
            r = test_raw_reset.iloc[pool[len(pool) // 2]]
            examples.append({"label": label, "is_fraud": bool(want_fraud), "txn": {
                "Amount": f"${float(r['Amount']):.2f}", "Merchant Name": str(r["Merchant Name"]),
                "Merchant City": str(r["Merchant City"]), "Merchant State": str(r["Merchant State"]),
                "Use Chip": str(r["Use Chip"]), "MCC": int(r["MCC"]), "Zip": str(r["Zip"]),
                "Time": str(r["Time"]), "Year": int(r["Year"]), "Month": int(r["Month"]),
                "Day": int(r["Day"]), "Card": int(r["Card"]), "User": int(r["User"])}})

    # ---- write everything -------------------------------------------------
    emit(f"Writing artifacts to {OUT} ...")
    joblib.dump(preproc, OUT / "preprocessor.joblib")
    joblib.dump(pca, OUT / "pca.joblib")
    joblib.dump(clf_raw, OUT / "xgb_raw.joblib")
    joblib.dump(clf_emb, OUT / "xgb_embed.joblib")
    joblib.dump(clf_comb, OUT / "xgb_combined.joblib")
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
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(f"Done — artifacts written to {OUT}")
    emit(f"Lift (AP): embed {summary['lift']['embed_ap_pct']}% · "
         f"combined {summary['lift']['combined_ap_pct']}%")
    return summary
