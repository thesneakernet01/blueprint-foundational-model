# SPDX-License-Identifier: Apache-2.0
"""tfm_demo.accel.backend()/xgb_device() — the cuda/rocm/cpu detection that
drives the whole hidden-backend switch. No real GPU is available in this test
image (CUDA or ROCm), so torch itself is monkeypatched rather than requiring
one."""

import sys
import types

import pytest

from tfm_demo import accel


def _fake_torch(cuda_available: bool, hip: str | None, cuda: str | None):
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda i=0: "Fake GPU",
    )
    mod.version = types.SimpleNamespace(hip=hip, cuda=cuda)
    return mod


@pytest.fixture(autouse=True)
def _reset_accel_cache(monkeypatch):
    monkeypatch.setattr(accel, "_backend", None)
    monkeypatch.setattr(accel, "_rapids_available", None)
    yield
    monkeypatch.setattr(accel, "_backend", None)
    monkeypatch.setattr(accel, "_rapids_available", None)


@pytest.mark.parametrize(
    "cuda_available,hip,cuda,expected",
    [
        (True, None, "12.1", "cuda"),
        (True, "6.2.41134", None, "rocm"),
        (False, None, None, "cpu"),
    ],
)
def test_backend_detection(monkeypatch, cuda_available, hip, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available, hip, cuda))
    assert accel.backend() == expected


@pytest.mark.parametrize(
    "cuda_available,hip,cuda,expected_device",
    [(True, None, "12.1", "cuda"), (True, "6.2.41134", None, "cpu"), (False, None, None, "cpu")],
)
def test_xgb_device_only_real_cuda_gets_gpu(monkeypatch, cuda_available, hip, cuda, expected_device):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available, hip, cuda))
    assert accel.xgb_device() == expected_device


def test_backend_caches_first_result(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "6.2.41134", None))
    assert accel.backend() == "rocm"
    # Swapping the fake torch after the first call must not change the answer.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.1"))
    assert accel.backend() == "rocm"
