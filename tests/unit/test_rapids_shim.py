# SPDX-License-Identifier: Apache-2.0
"""tfm_demo.rapids_shim — the pandas/scikit-learn stand-in for cudf/cupy/cuml
that lets the vendored src/tokenizer/*.py pipeline run unmodified when RAPIDS
isn't installed (ROCm boxes, or this CPU-only test image). Exercises the
exact call patterns the tokenizer pipeline actually uses, not a general cudf
API surface.
"""

import numpy as np
import pandas as pd
import pytest

from tfm_demo import rapids_shim


@pytest.fixture(scope="module", autouse=True)
def _shim_installed():
    # Real cudf isn't importable in this test image, so this exercises the
    # actual install path (not a mock of it).
    assert rapids_shim.install() is True
    import cudf
    assert getattr(cudf, "__tfm_demo_shim__", False) is True
    yield


def test_series_and_dataframe_are_pandas_backed():
    import cudf
    s = cudf.Series([1, 2, 3])
    assert isinstance(s, pd.Series)
    assert s.to_pandas() is s
    assert s.sum() == 6

    df = cudf.DataFrame({"a": [1, 2]})
    assert isinstance(df, pd.DataFrame)
    assert df.to_pandas() is df


def test_hash_values_deterministic_and_bucketable():
    import cudf
    s = cudf.Series(["AMAZON", "WALMART", "AMAZON"])
    hashed = s.hash_values()
    # categorical_hash.py does `column_data % vocab_limit` then `.map(dict)`.
    bucket = hashed % 2048
    idx_to_token = {i: f"MERCH_{i}" for i in range(2048)}
    tokens = bucket.map(idx_to_token)
    assert tokens.tolist()[0] == tokens.tolist()[2]        # same input -> same hash
    assert all(isinstance(t, str) and t.startswith("MERCH_") for t in tokens)


def test_cudf_concat_and_from_arrow():
    import cudf
    import pyarrow as pa

    a = cudf.DataFrame({"x": [1, 2]})
    b = cudf.DataFrame({"y": [3, 4]})
    combined = cudf.concat([a, b], axis=1)
    assert list(combined.columns) == ["x", "y"]

    tbl = pa.table({"z": [10, 20, 30]})
    df = cudf.DataFrame.from_arrow(tbl)
    assert df["z"].tolist() == [10, 20, 30]


def test_cupy_shim_numeric_ops_and_stream():
    import cupy as cp

    arr = cp.asarray([1.0, 2.0, 3.0])
    assert np.allclose(cp.asnumpy(cp.log(arr + 1.0)), np.log(arr + 1.0))
    assert cp.digitize(cp.array([0.5]), cp.array([0.0, 1.0])).tolist() == [1]

    # timedelta.py's optional stream context manager + pipeline.py's
    # cp.cuda.Stream.null.synchronize() -- both must be harmless no-ops.
    with cp.cuda.Stream(non_blocking=True):
        pass
    cp.cuda.Stream.null.synchronize()


def test_kbins_discretizer_transform_returns_dataframe_not_ndarray():
    """numerical.py calls .map() unconditionally on builder.transform()'s
    result -- sklearn's own KBinsDiscretizer.transform() returns a bare
    ndarray, which has no .map(). This is the one non-trivial adapter."""
    from cuml.preprocessing import KBinsDiscretizer

    col = pd.DataFrame({"amount": np.linspace(0, 1000, 50)})
    builder = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    builder.fit(col)
    bins = builder.transform(col)

    assert isinstance(bins, pd.DataFrame)
    assert list(bins.index) == list(col.index)

    idx_to_token = {i: f"AMT_{i}" for i in range(10)}
    tokens = bins.iloc[:, 0].astype("int32").map(idx_to_token)
    assert tokens.notna().all()
    assert set(tokens.unique()) <= set(idx_to_token.values())


def test_gpu_configure_gpu_memory_installs_shim_end_to_end():
    """The real entry point (gpu.py, called from engine.warmup() and
    run_export()) must reach the shim before any cuDF/cuML use -- not just
    rapids_shim.install() called directly, as the fixture above does."""
    from tfm_demo import gpu

    gpu.configure_gpu_memory()          # must not raise even with no RAPIDS/RMM
    import cudf
    assert cudf.Series([1, 2, 3]).to_pandas().sum() == 6
    assert gpu.host_copy_canary() is None      # skipped off-CUDA, not run-and-passed
    assert gpu.gpu_stack_versions().startswith("backend=")


def test_cuml_manifold_present_or_cleanly_absent():
    """UMAP is optional (umap-learn may not be installed) -- either it's
    there with the expected constructor kwargs, or cuml.manifold is simply
    missing so export.py's existing try/except keeps skipping it, exactly as
    it does today when cuml itself is absent."""
    import cuml

    if hasattr(cuml, "manifold"):
        umap_cls = cuml.manifold.UMAP
        umap_cls(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
    else:
        with pytest.raises(ModuleNotFoundError):
            import cuml.manifold                                    # noqa: F401
