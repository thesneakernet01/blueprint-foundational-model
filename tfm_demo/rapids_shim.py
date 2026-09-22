# SPDX-License-Identifier: Apache-2.0
"""A pandas/scikit-learn-backed stand-in for cudf/cupy/cuml, installed into
sys.modules only when the real RAPIDS stack can't be imported (ROCm boxes, or
plain CPU hosts with `torch` but no NVIDIA wheels).

Why a module shim instead of editing call sites: `src/tokenizer/*.py` and
`src/decoder_inference.py` are the fetched blueprint tree — gpu.py's
_patch_cudf_compat already establishes the pattern of monkeypatching around a
RAPIDS API gap from the outside rather than hand-editing those files, and this
extends the same approach to "RAPIDS isn't installed at all". Every consumer
(the vendored tokenizer files, plus tfm_demo/engine.py, impala.py, vast.py,
export.py) keeps its `import cudf` / `import cupy as cp` / `from cuml...`
lines completely unchanged and transparently gets the CPU-backed
implementation — that's what makes the backend swap "hidden".

Inert on any host where real cudf imports successfully: install() is a no-op
there, so this module never touches a real RAPIDS deployment.

Scope is deliberately narrow — only the exact cudf/cupy/cuml surface actually
called anywhere in this repo (confirmed by an audit of every call site), not
a general RAPIDS reimplementation:
  * cudf.Series / cudf.DataFrame -> pandas.Series / pandas.DataFrame, with
    .to_pandas() (identity) and Series.hash_values() (via
    pandas.util.hash_pandas_object) added by monkeypatching pandas' own
    classes -- so the shim survives arbitrary pandas op chains (concat,
    groupby, arithmetic, .str/.dt accessors) without needing a subclass to
    propagate through each one.
  * cudf.concat/to_datetime/from_pandas/set_option -> pandas equivalents.
  * cupy -> numpy, attribute-proxied wholesale, plus a no-op cuda.Stream
    (the tokenizer pipeline uses cp.cuda.Stream purely as an optional
    context-manager knob; single-threaded CPU execution needs no stream).
  * cuml.preprocessing.KBinsDiscretizer -> sklearn's, wrapped so
    .transform() returns a DataFrame (matching cuML's frame-in/frame-out
    behavior) instead of sklearn's bare ndarray -- numerical.py immediately
    calls .map() on the result, which ndarrays don't have. This is the one
    non-trivial adapter; everything else is a direct alias.
  * cuml.manifold.UMAP -> umap-learn's UMAP when installed (near-identical
    constructor/fit/transform API); omitted entirely otherwise, so
    export.py's existing try/except around the UMAP block keeps skipping it
    exactly as it does today when cuml is absent -- no new failure mode.
"""

from __future__ import annotations

import sys
import types

from .config import log

_installed = False


def install() -> bool:
    """Idempotent. Returns True if the shim is active (installed now or
    already installed earlier in this process); False if real RAPIDS is
    present and nothing was touched."""
    global _installed
    if _installed:
        return True
    try:
        import cudf as _real_cudf                                  # noqa: F401
        return False
    except Exception:                                               # noqa: BLE001
        pass

    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:                                        # noqa: BLE001
        # No RAPIDS AND no pandas (e.g. the web-only demo-fallback image,
        # which never needs this stack at all) -- nothing to shim with.
        # Leave cudf/cupy/cuml unimportable, same as before this shim existed.
        log.info("RAPIDS shim unavailable (no pandas either): %s", exc)
        return False

    _patch_pandas(pd)
    sys.modules["cudf"] = _build_cudf_module(pd)
    sys.modules["cupy"] = _build_cupy_module(np)
    cuml_mod, preprocessing_mod, manifold_mod = _build_cuml_module(pd)
    sys.modules["cuml"] = cuml_mod
    sys.modules["cuml.preprocessing"] = preprocessing_mod
    if manifold_mod is not None:
        sys.modules["cuml.manifold"] = manifold_mod

    _installed = True
    log.info(
        "RAPIDS shim installed (no cudf/cuml on this host) — tokenizer and "
        "UMAP stages run on pandas/scikit-learn%s",
        "" if manifold_mod is not None
        else " (umap-learn not installed: UMAP export step will skip, as it "
             "already does today when cuml is missing)",
    )
    return True


def _patch_pandas(pd) -> None:
    """Add the handful of cuDF-only methods the vendored code calls, directly
    onto pandas' own classes -- so the shim keeps working through arbitrary
    pandas op chains (concat, groupby, arithmetic) that a custom subclass
    would need to propagate through explicitly. Only ever runs when RAPIDS is
    absent (see install()), so this never touches a real-CUDA deployment."""
    if not hasattr(pd.Series, "to_pandas"):
        pd.Series.to_pandas = lambda self: self
    if not hasattr(pd.Series, "hash_values"):
        pd.Series.hash_values = lambda self: pd.util.hash_pandas_object(self, index=False)
    if not hasattr(pd.DataFrame, "to_pandas"):
        pd.DataFrame.to_pandas = lambda self: self
    if not hasattr(pd.DataFrame, "from_arrow"):
        # cudf.DataFrame.from_arrow(tbl) -- tbl is a pyarrow.Table (duck-typed,
        # no pyarrow import needed here).
        pd.DataFrame.from_arrow = staticmethod(lambda tbl: tbl.to_pandas())


def _build_cudf_module(pd) -> types.ModuleType:
    mod = types.ModuleType("cudf")
    mod.Series = pd.Series
    mod.DataFrame = pd.DataFrame
    mod.concat = pd.concat
    mod.to_datetime = pd.to_datetime
    mod.from_pandas = lambda df: df                    # already host-resident
    mod.set_option = lambda *a, **k: None               # e.g. "spill" -- no device memory to spill
    mod.__tfm_demo_shim__ = True
    return mod


def _build_cupy_module(np) -> types.ModuleType:
    mod = types.ModuleType("cupy")
    for name in dir(np):
        if not name.startswith("_"):
            setattr(mod, name, getattr(np, name))
    mod.asnumpy = lambda x: np.asarray(x)

    class _NullStream:
        """cp.cuda.Stream, cp.cuda.Stream(non_blocking=True) and
        cp.cuda.Stream.null.synchronize() -- all no-ops on single-threaded CPU."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def synchronize(self) -> None:
            pass

    _NullStream.null = _NullStream()

    cuda_mod = types.ModuleType("cupy.cuda")
    cuda_mod.Stream = _NullStream
    mod.cuda = cuda_mod
    mod.__tfm_demo_shim__ = True
    return mod


def _build_cuml_module(pd):
    from sklearn.preprocessing import KBinsDiscretizer as _SKKBinsDiscretizer

    class _KBinsDiscretizer(_SKKBinsDiscretizer):
        """cuML's KBinsDiscretizer.transform() returns a frame, matching its
        frame-in/frame-out convention; sklearn's returns a bare ndarray, and
        numerical.py calls .map() on the result unconditionally."""

        def transform(self, X):
            columns = list(X.columns) if hasattr(X, "columns") else None
            index = X.index if hasattr(X, "index") else None
            arr = super().transform(X)
            if hasattr(arr, "toarray"):          # only if encode="onehot"
                arr = arr.toarray()
            return pd.DataFrame(arr, columns=columns, index=index)

    preprocessing_mod = types.ModuleType("cuml.preprocessing")
    preprocessing_mod.KBinsDiscretizer = _KBinsDiscretizer

    manifold_mod = None
    try:
        import umap
        manifold_mod = types.ModuleType("cuml.manifold")
        manifold_mod.UMAP = umap.UMAP
    except Exception:                                               # noqa: BLE001
        pass

    cuml_mod = types.ModuleType("cuml")
    cuml_mod.preprocessing = preprocessing_mod
    if manifold_mod is not None:
        cuml_mod.manifold = manifold_mod
    cuml_mod.__tfm_demo_shim__ = True
    return cuml_mod, preprocessing_mod, manifold_mod
