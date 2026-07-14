# SPDX-License-Identifier: Apache-2.0
"""The inference engine: holds the heavy objects and runs a single transaction
end-to-end. Boots in REAL mode when the checkpoint, artifacts, and a GPU are all
present; otherwise DEMO-FALLBACK with clearly-labelled synthetic scores.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Dict, List, Optional, Tuple

from . import nexus
from .config import (
    ARTIFACTS,
    MAX_LENGTH,
    MERCHANT_HASH_SIZE,
    MODEL_DIR,
    RAW_FEATURE_COLS,
    TOKENIZER_COLS,
    log,
)
from .defaults import builtin_examples, builtin_summary

# Remote NEXUS calls run off-thread so the local pipeline never waits on the
# network; created lazily so an unconfigured app spawns no threads.
_NEXUS_POOL: Optional[ThreadPoolExecutor] = None


def _nexus_pool() -> ThreadPoolExecutor:
    global _NEXUS_POOL
    if _NEXUS_POOL is None:
        _NEXUS_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nexus")
    return _NEXUS_POOL


class Engine:
    def __init__(self) -> None:
        self.mode = "demo-fallback"      # flips to "real" if everything loads
        self.gpu = False
        self.detail = "not initialised"
        self.summary: Dict = {}
        self.examples: List[Dict] = []
        self.umap_background: List[Dict] = []
        self.nexus_meta: Dict = {}
        self._ready = False

        # heavy objects (real mode only)
        self._inference = None
        self._pipeline_cls = None
        self._tokenizer = None
        self._pca = None
        self._preproc = None
        self._xgb = {}
        self._umap = None
        self._np = None

    # -- startup -------------------------------------------------------------
    def warmup(self) -> None:
        self._load_static_assets()
        try:
            self._load_real_stack()
            self.mode = "real"
            self.detail = "checkpoint + tokenizer + XGBoost heads loaded"
            log.info("REAL mode active: %s", self.detail)
        except Exception as exc:                       # noqa: BLE001
            self.mode = "demo-fallback"
            self.detail = f"falling back to synthetic scoring: {exc}"
            log.warning("DEMO-FALLBACK mode: %s", exc)
        self._ready = True

    def _load_static_assets(self) -> None:
        """summary / examples / umap background are cheap JSON — load if present."""
        import numpy as np
        self._np = np
        # A re-warmup after an export where the NEXUS head didn't run must
        # drop the previous run's reference, not keep serving it.
        self.nexus_meta = {}
        for name, attr, _default in [
            ("summary.json", "summary", {}),
            ("examples.json", "examples", []),
            ("umap_background.json", "umap_background", []),
            ("nexus.json", "nexus_meta", {}),
        ]:
            p = ARTIFACTS / name
            if p.exists():
                setattr(self, attr, json.loads(p.read_text()))
        if not self.examples:
            self.examples = builtin_examples()
        if not self.summary:
            self.summary = builtin_summary()

    def _load_real_stack(self) -> None:
        from .gpu import configure_gpu_memory
        configure_gpu_memory()      # RMM pool + cuDF spill — must precede cuDF/cuML use

        import torch
        import joblib

        self.gpu = torch.cuda.is_available()
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"checkpoint missing at {MODEL_DIR} (run git lfs pull)")
        for f in ("preprocessor.joblib", "pca.joblib",
                  "xgb_raw.joblib", "xgb_embed.joblib", "xgb_combined.joblib"):
            if not (ARTIFACTS / f).exists():
                raise FileNotFoundError(f"{f} missing — run export_for_demo.py")

        from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer
        from src.decoder_inference import HuggingFaceDecoderInference

        self._pipeline_cls = FinancialTokenizerPipeline
        self._tokenizer = FinancialTabularTokenizer(
            merchant_hash_size=MERCHANT_HASH_SIZE,
            category_hierarchy=True,
            temporal_encoding=True,
        )
        self._inference = HuggingFaceDecoderInference(
            model_path=MODEL_DIR, tokenizer=self._tokenizer, pooling="last_token",
        )
        self._preproc = joblib.load(ARTIFACTS / "preprocessor.joblib")
        self._pca = joblib.load(ARTIFACTS / "pca.joblib")
        self._xgb = {
            "raw": joblib.load(ARTIFACTS / "xgb_raw.joblib"),
            "embed": joblib.load(ARTIFACTS / "xgb_embed.joblib"),
            "combined": joblib.load(ARTIFACTS / "xgb_combined.joblib"),
        }
        umap_path = ARTIFACTS / "umap2d.joblib"
        self._umap = joblib.load(umap_path) if umap_path.exists() else None

    # -- the real per-transaction pipeline -----------------------------------
    def _embed_one(self, txn: Dict):
        """raw transaction dict -> (512-d embedding np.array, token strings)."""
        import cudf

        txn = dict(txn)
        # In the dataset a missing Zip is a real null, which the blueprint's
        # preprocess fillna()s before its zip3 astype(int). But examples.json
        # serialises it as the string "nan" (str() of a float NaN), and users
        # can type anything — non-digit strings crash the int cast in cuDF.
        # Normalise to the fillna value. (The raw-feature path is left alone:
        # there "nan" coerces to float NaN, matching training.)
        z = str(txn.get("Zip", "")).strip()
        z = z[:-2] if z.endswith(".0") else z
        txn["Zip"] = z if z.isdigit() else "00000"

        gdf = cudf.DataFrame({c: [txn.get(c)] for c in TOKENIZER_COLS})
        pip = self._pipeline_cls(merchant_hash_size=MERCHANT_HASH_SIZE)
        gdf = pip.preprocess(gdf)
        pip.fit(gdf)
        token_df = pip.transform(gdf)
        padded = pip.encode(token_df, max_length=MAX_LENGTH)          # (1, 128)
        tokens = self._tokenizer.decode(padded[0].tolist()).split()
        emb = self._inference.extract_embeddings(
            __import__("torch").from_numpy(padded), return_numpy=True
        )                                                             # (1, 512)
        return emb, tokens

    def _raw_frame(self, txn: Dict):
        """The single-row UNTRANSFORMED raw-feature frame (numeric coercions
        applied, sklearn preprocessor not) — what the NEXUS head scores and
        what `_raw_vector` feeds to the fitted preprocessor."""
        import pandas as pd
        row = {c: txn.get(c) for c in RAW_FEATURE_COLS}
        # numeric coercions matching NB05 feature engineering
        amt = str(row.get("Amount", "0")).replace("$", "").replace(",", "")
        row["Amount"] = float(amt or 0)
        if row.get("Hour") in (None, ""):
            t = str(txn.get("Time", "0:00"))
            row["Hour"] = int(t.split(":")[0]) if ":" in t else 0
        # The preprocessor was fitted on TabFormer dtypes, where "Merchant Name"
        # is a hashed int64 and "Zip" a float64 — passthrough numerics, not
        # OrdinalEncoder columns. A string here (the UI posts e.g. "AMAZON")
        # makes the feature matrix object-dtype and XGBoost's float coercion
        # 500s. Map non-numeric merchants to -1.0 (an unseen hash, mirroring
        # the encoder's unknown_value=-1 convention).
        try:
            row["Merchant Name"] = float(str(row.get("Merchant Name", "")).strip())
        except ValueError:
            row["Merchant Name"] = -1.0
        try:
            row["Zip"] = float(str(row.get("Zip", "")).strip() or 0)
        except ValueError:
            row["Zip"] = 0.0
        return pd.DataFrame([row], columns=RAW_FEATURE_COLS)

    def _raw_vector(self, txn: Dict):
        return self._preproc.transform(self._raw_frame(txn))

    # -- NEXUS fourth head (remote; additive, null-safe, never blocks long) ---
    def _nexus_submit(self, txn: Dict) -> Optional[Tuple[Future, float]]:
        """Fire the remote NEXUS score before the local pipeline runs, so the
        two overlap. None when the head is off — the response then carries no
        nexus keys at all."""
        if not nexus.configured():
            return None
        try:
            frame = self._raw_frame(txn)
        except Exception as exc:                                   # noqa: BLE001
            log.warning("[nexus] could not build raw frame: %s", exc)
            return None
        t0 = time.perf_counter()
        return _nexus_pool().submit(nexus.score_one, frame, self.nexus_meta), t0

    def _nexus_harvest(self, submitted: Tuple[Future, float]):
        """(score|None, status side-channel) within the scoring budget. A
        timed-out call finishes harmlessly on the pool thread."""
        future, t0 = submitted
        try:
            p = future.result(timeout=nexus.score_timeout_s())
        except FutureTimeout:
            return None, {"status": "timeout", "latency_ms": None}
        except Exception:                                          # noqa: BLE001
            p = None
        latency = round((time.perf_counter() - t0) * 1000, 1)
        if p is None:
            return None, {"status": "unavailable", "latency_ms": None}
        return p, {"status": "ok", "latency_ms": latency}

    def score(self, txn: Dict) -> Dict:
        submitted = self._nexus_submit(txn)
        out = self._score_real(txn) if self.mode == "real" else self._score_fallback(txn)
        if submitted is not None:
            out["scores"]["nexus"], out["nexus"] = self._nexus_harvest(submitted)
        return out

    def _score_real(self, txn: Dict) -> Dict:
        np = self._np
        emb, tokens = self._embed_one(txn)
        emb_pca = self._pca.transform(emb)                            # (1, 64)
        raw_vec = self._raw_vector(txn)                               # (1, n_raw)
        combined = np.hstack([raw_vec, emb_pca])

        p_raw = float(self._xgb["raw"].predict_proba(raw_vec)[0, 1])
        p_emb = float(self._xgb["embed"].predict_proba(emb_pca)[0, 1])
        p_comb = float(self._xgb["combined"].predict_proba(combined)[0, 1])

        pos = None
        if self._umap is not None:
            try:
                import cupy as cp
                xy = self._umap.transform(cp.asarray(emb))
                xy = cp.asnumpy(xy)[0]
                pos = {"x": float(xy[0]), "y": float(xy[1])}
            except Exception:                                          # noqa: BLE001
                pos = None

        return {
            "mode": self.mode,
            "tokens": tokens,
            "embedding_dim": int(emb.shape[1]),
            "scores": {"raw": p_raw, "embed": p_emb, "combined": p_comb},
            "position": pos,
        }

    # -- fallback (no GPU / no artifacts): plausible, clearly-labelled --------
    def _score_fallback(self, txn: Dict) -> Dict:
        np = self._np
        amt = float(str(txn.get("Amount", "0")).replace("$", "").replace(",", "") or 0)
        online = "ONLINE" in str(txn.get("Use Chip", "")).upper()
        odd_hour = int(str(txn.get("Time", "12:00")).split(":")[0]) in range(1, 5)
        signal = 0.18 + 0.30 * online + 0.22 * odd_hour + min(amt / 6000.0, 0.35)
        rng = np.random.default_rng(int(abs(hash(json.dumps(txn, sort_keys=True))) % 2**31))
        base = float(np.clip(signal + rng.normal(0, 0.05), 0.01, 0.98))
        return {
            "mode": self.mode,
            "tokens": ["<bos>", "AMT_?", "MERCH_?", "...", "<eos>"],
            "embedding_dim": 512,
            "scores": {
                "raw": float(np.clip(base - 0.07, 0.01, 0.99)),
                "embed": float(np.clip(base + 0.05, 0.01, 0.99)),
                "combined": float(np.clip(base + 0.08, 0.01, 0.99)),
            },
            "position": {"x": float(rng.normal(0, 4)), "y": float(rng.normal(0, 4))},
        }
