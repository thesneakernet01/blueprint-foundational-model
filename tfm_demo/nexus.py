# SPDX-License-Identifier: Apache-2.0
"""NEXUS (Fundamental's Large Tabular Model) head — the fourth model card.

This module owns ALL knowledge of NEXUS: SDK import, configuration, the
SageMaker/S3 transport, and a stub. Nothing else in the app may import the
NEXUS SDK or read NEXUS_* env vars.

Unlike the XGBoost heads, NEXUS scores the RAW transaction — the untransformed
RAW_FEATURE_COLS frame, not the ordinal-encoded matrix and not the PCA
embeddings — because the model's premise is that it ingests raw tables with no
feature engineering.

Modes ($NEXUS_MODE):
    off   (default) the head does not exist; every caller skips it and API
          payloads are byte-identical to the pre-NEXUS app.
    stub  deterministic fake scores + canned metrics flagged "stub": true —
          exercises the full export -> summary -> UI path with zero AWS access.
    live  the real SDK against a pre-deployed SageMaker endpoint. The two
          transport helpers (_fit / _predict_frame) are the ONLY code that will
          change once Fundamental access lands (see docs/nexus-ltm-design.md
          for the day-one checklist); today they raise NexusUnavailable.

Live config is env-only (AWS credentials via boto3's standard chain):
    NEXUS_ENDPOINT_NAME   pre-deployed SageMaker endpoint (never created here —
                          the instance is ~$60+/hr; lifecycle is a runbook item)
    NEXUS_REGION          falls back to $AWS_DEFAULT_REGION
    NEXUS_S3_BUCKET       staging bucket for the SDK's data round-trips
    NEXUS_S3_PREFIX       staging prefix (default "nexus-staging")
    NEXUS_SCORE_TIMEOUT_S live-scoring budget, default 2.5 (engine returns a
                          null score past it — the demo never blocks on NEXUS)
    NEXUS_FIT_TIMEOUT_S   fit/adapt polling ceiling, default 3600
    NEXUS_TRAIN_MAX       cap on training rows shipped to fit, default 20000
"""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from typing import Callable, Dict, Optional

from .config import PROJECT_ROOT, RAW_FEATURE_COLS, log

NEXUS_KEY = "nexus"
NEXUS_LABEL = "Large Tabular Model (NEXUS)"

Progress = Optional[Callable[[str], None]]


class NexusUnavailable(RuntimeError):
    """NEXUS cannot serve the request (not installed / not implemented /
    endpoint unreachable). score_one() converts this to a null score."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
# The mode is settable from the UI (Build-artifacts dialog -> POST /api/nexus)
# for demo boxes with no shell access. It persists in its own file — NOT in
# .data_settings.json, which the Data dialog rewrites wholesale on save — and
# a UI choice wins over $NEXUS_MODE. Both export and scoring read mode() per
# use, so changes apply without a restart.
MODES = ("off", "stub", "live")
_SETTINGS_PATH = PROJECT_ROOT / ".nexus_settings.json"
_SETTINGS_LOCK = threading.Lock()


def _persisted_mode() -> str:
    try:
        m = json.loads(_SETTINGS_PATH.read_text()).get("mode", "")
        return m if m in MODES else ""
    except FileNotFoundError:
        return ""
    except Exception as exc:                                       # noqa: BLE001
        _warn_ratelimited(f"unreadable {_SETTINGS_PATH.name}: {exc}")
        return ""


def save_mode(m: str) -> None:
    m = (m or "").strip().lower()
    if m not in MODES:
        raise ValueError(f"nexus mode must be one of {', '.join(MODES)}")
    with _SETTINGS_LOCK:
        tmp = _SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"mode": m}))
        tmp.replace(_SETTINGS_PATH)


def settings() -> Dict:
    """What the UI needs to render the mode control."""
    return {
        "mode": mode(),
        "live_ready": bool(_endpoint() and _bucket()),
        "target": target(),
    }


def mode() -> str:
    m = _persisted_mode()
    if m:
        return m
    m = os.environ.get("NEXUS_MODE", "off").strip().lower() or "off"
    if m not in MODES:
        _warn_ratelimited(f"unknown NEXUS_MODE={m!r} — treating as 'off'")
        return "off"
    return m


def _endpoint() -> str:
    return os.environ.get("NEXUS_ENDPOINT_NAME", "").strip()


def _region() -> str:
    return (os.environ.get("NEXUS_REGION", "").strip()
            or os.environ.get("AWS_DEFAULT_REGION", "").strip())


def _bucket() -> str:
    return os.environ.get("NEXUS_S3_BUCKET", "").strip()


def _prefix() -> str:
    return os.environ.get("NEXUS_S3_PREFIX", "nexus-staging").strip().strip("/")


def score_timeout_s() -> float:
    return float(os.environ.get("NEXUS_SCORE_TIMEOUT_S", "2.5"))


def fit_timeout_s() -> float:
    return float(os.environ.get("NEXUS_FIT_TIMEOUT_S", "3600"))


def train_max() -> int:
    return int(os.environ.get("NEXUS_TRAIN_MAX", "20000"))


def configured() -> bool:
    """Whether the NEXUS head should exist at all. off -> False; stub -> True;
    live -> only with an endpoint + staging bucket named."""
    m = mode()
    if m == "stub":
        return True
    if m == "live":
        return bool(_endpoint() and _bucket())
    return False


def target() -> str:
    if mode() == "stub":
        return "stub (deterministic fake scores)"
    return (f"SageMaker endpoint '{_endpoint() or '<unset>'}' "
            f"· staging s3://{_bucket() or '<unset>'}/{_prefix()}"
            + (f" · {_region()}" if _region() else ""))


def _sdk():
    """The one place the (name-TBD) Fundamental SDK is imported. Day-one task:
    fill in the real package name here and in requirements-nexus.txt."""
    try:
        raise ImportError("package name unknown until Fundamental access")
        # import nexus_sdk  # noqa: ERA001 — day-one: real import goes here
        # return nexus_sdk
    except ImportError as exc:
        raise NexusUnavailable(
            "NEXUS SDK not installed — see requirements-nexus.txt and the "
            "day-one checklist in docs/nexus-ltm-design.md"
        ) from exc


# --------------------------------------------------------------------------- #
# live transport (the ONLY code that changes on day one of access)
# --------------------------------------------------------------------------- #
def _fit(train_df, y_train, val_df, y_val, progress: Callable[[str], None]) -> Dict:
    """Fit/adapt NEXUS on the training frame. Returns a model reference dict
    (whatever the SDK needs to score later: model id, S3 URI of the adapted
    state or of the in-context training data — unknown until access)."""
    _sdk()
    raise NexusUnavailable("NEXUS live transport not implemented — day-one task")


def _predict_frame(df, meta: Optional[Dict], timeout_s: float):
    """P(fraud) for each row of an untransformed RAW_FEATURE_COLS frame,
    as a 1-d float array (positive class)."""
    _sdk()
    raise NexusUnavailable("NEXUS live transport not implemented — day-one task")


# --------------------------------------------------------------------------- #
# stub — deterministic, clearly labelled, zero AWS
# --------------------------------------------------------------------------- #
# Canned card metrics: plausible next to the XGBoost heads, and always paired
# with "stub": true so the UI can tag them.
_STUB_AUC = 0.9712
_STUB_AP = 0.6205


def _stub_score(row: Dict) -> float:
    """Deterministic across processes (crc32, not salted hash), shaped like
    engine._score_fallback's signal but offset so the bar visibly differs."""
    import numpy as np

    try:
        amt = float(row.get("Amount") or 0)
    except (TypeError, ValueError):
        amt = 0.0
    online = "ONLINE" in str(row.get("Use Chip", "")).upper()
    try:
        odd_hour = int(row.get("Hour") or 12) in range(1, 5)
    except (TypeError, ValueError):
        odd_hour = False
    signal = 0.24 + 0.30 * online + 0.22 * odd_hour + min(amt / 6000.0, 0.35)
    seed = zlib.crc32(repr(sorted((k, str(v)) for k, v in row.items())).encode())
    rng = np.random.default_rng(seed & 0x7FFFFFFF)
    return float(np.clip(signal + rng.normal(0, 0.04), 0.01, 0.99))


# --------------------------------------------------------------------------- #
# public surface
# --------------------------------------------------------------------------- #
def fit_head(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
             progress: Progress = None) -> Dict:
    """Train/adapt the NEXUS head on the untransformed RAW_FEATURE_COLS frames
    and evaluate it on the same test rows as the XGBoost heads. Returns the
    meta dict persisted as demo_artifacts/nexus.json. May raise — the export
    catches and skips the head."""
    emit = progress or (lambda m: log.info("[nexus] %s", m))
    for name, df in (("train", X_train_raw), ("val", X_val_raw), ("test", X_test_raw)):
        missing = [c for c in RAW_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} frame is missing raw columns: {missing}")

    n_train = min(len(X_train_raw), train_max())
    meta: Dict = {
        "mode": mode(),
        "endpoint": _endpoint() or None,
        "region": _region() or None,
        "n_train": int(n_train),
        "n_test": int(len(X_test_raw)),
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stub": mode() == "stub",
    }

    if mode() == "stub":
        emit(f"  nexus: [stub] adapting on {n_train:,} raw rows "
             f"({len(RAW_FEATURE_COLS)} columns, no feature engineering) ...")
        time.sleep(2)
        emit(f"  nexus: [stub] scoring {len(X_test_raw):,} test rows ...")
        meta.update(model_ref="stub", sdk_version="stub",
                    test_auc=_STUB_AUC, test_ap=_STUB_AP)
        return meta

    # live
    from sklearn.metrics import average_precision_score, roc_auc_score

    sdk = _sdk()
    train = X_train_raw.iloc[:n_train]
    ref = _fit(train, y_train[:n_train], X_val_raw, y_val, emit)
    meta.update(ref, sdk_version=getattr(sdk, "__version__", "?"))
    probs = _predict_frame(X_test_raw, meta, timeout_s=fit_timeout_s())
    meta["test_auc"] = round(float(roc_auc_score(y_test, probs)), 4)
    meta["test_ap"] = round(float(average_precision_score(y_test, probs)), 4)
    return meta


def score_one(txn_raw_df, meta: Optional[Dict] = None,
              timeout_s: Optional[float] = None) -> Optional[float]:
    """P(fraud) for a single-row untransformed RAW_FEATURE_COLS frame.
    NEVER raises: None on off/misconfiguration/timeout/error (the engine turns
    that into a null score + status for the UI)."""
    try:
        if not configured():
            return None
        if mode() == "stub":
            return _stub_score(txn_raw_df.iloc[0].to_dict())
        probs = _predict_frame(txn_raw_df, meta, timeout_s or score_timeout_s())
        return float(probs[0])
    except Exception as exc:                                       # noqa: BLE001
        _warn_ratelimited(f"score failed: {exc}")
        return None


def check() -> Dict:
    """Cheap connectivity probe for scripts/nexus_probe.py and debugging.
    Never raises."""
    out: Dict = {"ok": False, "mode": mode(), "target": target(),
                 "error": None, "latency_ms": None}
    if mode() == "off":
        out["error"] = "NEXUS_MODE is off"
        return out
    if mode() == "stub":
        out["ok"] = True
        return out
    if not configured():
        out["error"] = "NEXUS_MODE=live needs NEXUS_ENDPOINT_NAME and NEXUS_S3_BUCKET"
        return out
    try:
        import boto3
        from botocore.config import Config

        cfg = Config(connect_timeout=5, read_timeout=15,
                     retries={"mode": "standard", "max_attempts": 2})
        t0 = time.perf_counter()
        sm = boto3.client("sagemaker", region_name=_region() or None, config=cfg)
        desc = sm.describe_endpoint(EndpointName=_endpoint())
        status = desc.get("EndpointStatus", "?")
        boto3.client("s3", region_name=_region() or None, config=cfg) \
             .head_bucket(Bucket=_bucket())
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if status != "InService":
            out["error"] = f"endpoint status is {status}, not InService"
        else:
            out["ok"] = True
    except Exception as exc:                                       # noqa: BLE001
        out["error"] = str(exc).split("\n", 1)[0][:300]
    return out


# --------------------------------------------------------------------------- #
# rate-limited warnings (score_one runs per transaction — don't spam the log)
# --------------------------------------------------------------------------- #
_WARN_LOCK = threading.Lock()
_WARN_LAST = [0.0]
_WARN_EVERY_S = 30.0


def _warn_ratelimited(msg: str) -> None:
    with _WARN_LOCK:
        now = time.monotonic()
        if now - _WARN_LAST[0] < _WARN_EVERY_S:
            return
        _WARN_LAST[0] = now
    log.warning("[nexus] %s", msg)
