# SPDX-License-Identifier: Apache-2.0
"""File-based scoring entry point for a CML Model build.

Used only when the workspace can't build directly from a Model Registry
version (older CML releases): tfm_demo/registry.py falls back to a build with
file_path=tfm_demo/registry_predict.py, function_name=predict. Serves the same
bundle the registry path would: the combined XGBoost head + preprocessor + PCA
from demo_artifacts/. Input rows must already carry the pca_0..pca_63 embedding
components — the foundation-model embedding stage runs upstream.

Request:  {"rows": [{<raw feature cols...>, "pca_0": ..., ...}, ...]}
Response: {"probability": [<fraud probability per row>]}
"""

import joblib
import numpy as np
import pandas as pd

from tfm_demo.config import ARTIFACTS

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = {
            "preproc": joblib.load(ARTIFACTS / "preprocessor.joblib"),
            "pca": joblib.load(ARTIFACTS / "pca.joblib"),
            "clf": joblib.load(ARTIFACTS / "xgb_combined.joblib"),
        }
    return _bundle


def predict(args):
    b = _load()
    df = pd.DataFrame(args["rows"])
    pca_cols = sorted((c for c in df.columns if c.startswith("pca_")),
                      key=lambda c: int(c.split("_")[1]))
    raw = df[[c for c in df.columns if not c.startswith("pca_")]]
    X = np.hstack([b["preproc"].transform(raw), df[pca_cols].to_numpy()])
    return {"probability": b["clf"].predict_proba(X)[:, 1].tolist()}
