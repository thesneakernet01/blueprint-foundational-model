# SPDX-License-Identifier: Apache-2.0
"""Offline artifact export — mirrors notebook 05 exactly, then serialises
everything the live demo needs into `demo_artifacts/`:

    preprocessor.joblib   sklearn OrdinalEncoder column transformer
    pca.joblib            PCA(512 -> 64)
    xgb_raw/embed/combined.joblib   the three XGBoost heads
    umap2d.joblib         cuML UMAP fitted on a test subsample (live projection)
    umap_background.json  scatter points {x, y, fraud} for the embedding map
    examples.json         real test transactions to click
    summary.json          headline test-set AUC / AP / lift

`run_export()` is importable so the API can run it on the GPU backend and stream
progress to the UI (see tfm_demo/jobs.py); the root `export_for_demo.py` shim
runs the same function from the CLI. Requires notebook 04 (embeddings) and the
temporal parquet splits to exist under the blueprint repo first.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

import numpy as np
import joblib

from .config import ARTIFACTS as OUT
from .config import FRAUD_COL, PCA_DIM, RAW_FEATURE_COLS, REPO_ROOT

EMBED_DIR = REPO_ROOT / "data" / "embeddings"
TEMPORAL_DIR = REPO_ROOT / "data" / "TabFormer" / "temporal_split"

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


def run_export(progress: Progress = None) -> Dict:
    """Train the heads, fit PCA/UMAP, write artifacts, and return the summary.

    `progress(msg)` is called with human-readable step messages so the caller
    (CLI or API) can stream them. Raises on missing prerequisites (e.g. NB04
    embeddings) — the caller surfaces the error.
    """
    emit = progress or (lambda _m: None)

    import cudf
    import xgboost as xgb
    import torch
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score

    OUT.mkdir(exist_ok=True)
    if not EMBED_DIR.exists():
        raise FileNotFoundError(f"missing {EMBED_DIR} — run notebook 04 first")
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    emit(f"Compute device: {xgb_device}")

    # ---- embeddings (NB04) ------------------------------------------------
    emit("Loading embeddings (notebook 04 output) ...")
    X_train_e = np.load(EMBED_DIR / "train_embeddings.npy")
    y_train = np.load(EMBED_DIR / "train_labels.npy")
    train_ids = np.load(EMBED_DIR / "train_row_ids.npy")
    X_val_e = np.load(EMBED_DIR / "val_embeddings.npy")
    y_val = np.load(EMBED_DIR / "val_labels.npy")
    val_ids = np.load(EMBED_DIR / "val_row_ids.npy")
    X_test_e = np.load(EMBED_DIR / "test_embeddings.npy")
    y_test = np.load(EMBED_DIR / "test_labels.npy")
    test_ids = np.load(EMBED_DIR / "test_row_ids.npy")
    n_train = len(X_train_e)

    # ---- PCA 512 -> 64 ----------------------------------------------------
    emit(f"PCA {X_train_e.shape[1]}d -> {PCA_DIM}d ...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    Xtr_pca = pca.fit_transform(X_train_e)
    Xva_pca = pca.transform(X_val_e)
    Xte_pca = pca.transform(X_test_e)

    # ---- raw tabular features (NB05) -------------------------------------
    emit("Loading temporal parquets + feature engineering ...")
    train_pdf = cudf.read_parquet(str(TEMPORAL_DIR / "train.parquet")).to_pandas()
    val_pdf = cudf.read_parquet(str(TEMPORAL_DIR / "val_eval.parquet")).to_pandas()
    test_pdf = cudf.read_parquet(str(TEMPORAL_DIR / "test_eval.parquet")).to_pandas()
    for pdf in (train_pdf, val_pdf, test_pdf):
        pdf["Hour"] = pdf["Time"].str.split(":", n=1, expand=True)[0].astype(int)
        pdf["Amount"] = pdf["Amount"].str.replace("$", "", regex=False)\
            .str.replace(",", "").astype(float)

    fraud_mask = (train_pdf[FRAUD_COL] == "Yes") | (train_pdf[FRAUD_COL] == "1")
    fraud_idx = train_pdf.index[fraud_mask].tolist()
    normal_idx = train_pdf.index[~fraud_mask].tolist()
    np.random.seed(42)
    n_fraud = min(len(fraud_idx), int(n_train * 0.1))
    n_normal = min(len(normal_idx), n_train - n_fraud)
    bal = np.concatenate([np.random.choice(fraud_idx, n_fraud, replace=False),
                          np.random.choice(normal_idx, n_normal, replace=False)])
    np.random.shuffle(bal)

    X_train_raw = train_pdf.loc[bal, RAW_FEATURE_COLS].reset_index(drop=True)\
        .iloc[train_ids].reset_index(drop=True)
    X_val_raw = val_pdf.iloc[val_ids][RAW_FEATURE_COLS].reset_index(drop=True)
    X_test_raw = test_pdf.iloc[test_ids][RAW_FEATURE_COLS].reset_index(drop=True)

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
    for want_fraud, label in [(1, "Real fraud (test set)"),
                              (0, "Real legitimate (test set)"),
                              (1, "Real fraud #2 (test set)")]:
        pool = np.where(y_test == want_fraud)[0]
        if len(pool):
            r = test_pdf.iloc[test_ids[pool[len(pool) // 2]]]
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
