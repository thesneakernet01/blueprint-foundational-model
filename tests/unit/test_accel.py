# SPDX-License-Identifier: Apache-2.0
"""tfm_demo.accel.backend()/xgb_status() — the cuda/rocm/cpu detection that
drives the whole hidden-backend switch, plus the separate question of whether
the installed XGBoost can use that GPU (mainline CUDA wheel vs. AMD's ROCm/HIP
build). No real GPU is available in this test image (CUDA or ROCm), so torch
and xgboost are both monkeypatched rather than requiring one."""

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


def _fake_xgboost(version: str = "3.2.0", **build_info):
    mod = types.ModuleType("xgboost")
    mod.__version__ = version
    mod.build_info = lambda: dict(build_info)
    return mod


@pytest.fixture(autouse=True)
def _reset_accel_cache(monkeypatch):
    for attr in ("_backend", "_rapids_available", "_xgb_status"):
        monkeypatch.setattr(accel, attr, None)
    # Neither device force nor the skip-knob leaks in from the environment.
    monkeypatch.delenv("DEMO_XGB_DEVICE", raising=False)
    monkeypatch.delenv("DEMO_XGB_GPU_PROBE", raising=False)
    yield
    for attr in ("_backend", "_rapids_available", "_xgb_status"):
        monkeypatch.setattr(accel, attr, None)


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


def _use(monkeypatch, *, torch_mod, xgb_mod=None, probe=None):
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    if xgb_mod is None:
        monkeypatch.setitem(sys.modules, "xgboost", None)   # import -> ImportError
    else:
        monkeypatch.setitem(sys.modules, "xgboost", xgb_mod)
    monkeypatch.setattr(accel, "_probe_xgb_gpu", lambda *a, **k: probe)


def test_xgb_device_nvidia_wheel_on_cuda(monkeypatch):
    """The NVIDIA path is unchanged: mainline's CUDA wheel trains on the GPU."""
    _use(monkeypatch, torch_mod=_fake_torch(True, None, "12.1"),
         xgb_mod=_fake_xgboost(USE_CUDA=True, USE_HIP=False))
    assert accel.xgb_device() == "cuda"


def test_xgb_device_cpu_wheel_on_rocm_falls_back(monkeypatch):
    """Mainline XGBoost on an AMD box has no HIP kernels — heads run on CPU,
    and the detail says how to fix it."""
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=False))
    status = accel.xgb_status()
    assert (status["device"], status["gpu"], status["build"]) == ("cpu", False, "cpu")
    assert "ROCm/HIP" in status["detail"]


def test_xgb_device_amd_hip_build_on_rocm_uses_gpu(monkeypatch):
    """AMD's build reports USE_HIP and still takes device="cuda"."""
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=True), probe=None)
    status = accel.xgb_status()
    assert (status["device"], status["gpu"], status["build"]) == ("cuda", True, "hip")


def test_xgb_device_amd_hip_build_failing_probe_falls_back(monkeypatch):
    """A HIP build compiled for the wrong gfx arch degrades to CPU, not a crash."""
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=True),
         probe="exit 1: hipErrorNoBinaryForGpu")
    status = accel.xgb_status()
    assert status["device"] == "cpu" and status["gpu"] is False
    assert "hipErrorNoBinaryForGpu" in status["detail"]


def test_xgb_device_no_gpu(monkeypatch):
    _use(monkeypatch, torch_mod=_fake_torch(False, None, None),
         xgb_mod=_fake_xgboost(USE_CUDA=True, USE_HIP=False))
    assert accel.xgb_device() == "cpu"


def test_xgb_device_without_xgboost_installed(monkeypatch):
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None), xgb_mod=None)
    assert accel.xgb_status()["detail"] == "xgboost is not installed"


@pytest.mark.parametrize("forced,expected", [("cpu", "cpu"), ("cuda", "cuda")])
def test_xgb_device_env_override(monkeypatch, forced, expected):
    monkeypatch.setenv("DEMO_XGB_DEVICE", forced)
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=False))
    assert accel.xgb_device() == expected


def test_xgb_probe_runs_at_most_once(monkeypatch):
    """The subprocess smoke-fit is cached with the rest of the status."""
    calls = []
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=True))
    monkeypatch.setattr(accel, "_probe_xgb_gpu",
                        lambda *a, **k: calls.append(1) and None)
    assert accel.xgb_device() == "cuda"
    assert accel.xgb_device() == "cuda"
    assert len(calls) == 1


def test_xgb_probe_skipped_by_env(monkeypatch):
    monkeypatch.setenv("DEMO_XGB_GPU_PROBE", "0")
    _use(monkeypatch, torch_mod=_fake_torch(True, "6.2.41134", None),
         xgb_mod=_fake_xgboost(USE_CUDA=False, USE_HIP=True),
         probe="would have failed")
    assert accel.xgb_device() == "cuda"


def test_backend_caches_first_result(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "6.2.41134", None))
    assert accel.backend() == "rocm"
    # Swapping the fake torch after the first call must not change the answer.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.1"))
    assert accel.backend() == "rocm"
