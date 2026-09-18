#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build and install the AMD (ROCm/HIP) build of XGBoost, so the three fraud
# heads in tfm_demo/export.py train on the AMD GPU instead of the CPU.
#
# Why a build and not a wheel: mainline XGBoost's PyPI wheel is CUDA-only (it
# is CPU-only on AMD hardware), and AMD does not publish an XGBoost wheel on
# repo.radeon.com the way it does torch/tensorflow. The HIP port lives in
# AMD-Ecosystem/xgboost (a fork of dmlc/xgboost carrying src/common/cuda_to_hip.h
# and the rocgputreeshap submodule) and is built with -DUSE_HIP=ON.
#
# The resulting package reports USE_HIP=true in xgboost.build_info(), which is
# exactly what tfm_demo/accel.py probes, and it still takes device="cuda" —
# HIP is a source-level translation of the CUDA API, so the device string
# never changes. Nothing in export.py needs to know which vendor it is on.
#
# Usage (on the ROCm host, inside the same Python env the app runs in):
#     deploy/build_xgboost_rocm.sh
#
# Environment:
#   XGBOOST_ROCM_REF    git ref of the fork to build     (default release/3.2.0)
#   XGBOOST_ROCM_ARCH   GPU arch(s), e.g. gfx942         (default: auto-detected)
#   XGBOOST_ROCM_DIR    build/checkout directory         (default /tmp/xgboost-rocm)
#   ROCM_PATH           ROCm install prefix              (default /opt/rocm)
#   XGBOOST_BUILD_JOBS  parallel compile jobs            (default: nproc)
#
# Expect a long build (tens of minutes — it compiles every HIP kernel for the
# selected arch). Re-runs reuse the checkout and the CMake cache.

set -euo pipefail

REPO="https://github.com/AMD-Ecosystem/xgboost.git"
REF="${XGBOOST_ROCM_REF:-release/3.2.0}"
SRC="${XGBOOST_ROCM_DIR:-/tmp/xgboost-rocm}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="${XGBOOST_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
PYTHON="${PYTHON:-python3}"

command -v git   >/dev/null || { echo "git is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required (pip install cmake)" >&2; exit 1; }
[ -x "${ROCM_PATH}/bin/hipcc" ] || {
  echo "hipcc not found at ${ROCM_PATH}/bin — set ROCM_PATH to the ROCm prefix" >&2
  exit 1
}

# Compiling for every arch ROCm knows about multiplies the build time; default
# to the arch of the card actually in this box.
ARCH="${XGBOOST_ROCM_ARCH:-}"
if [ -z "${ARCH}" ]; then
  ARCH="$("${ROCM_PATH}/bin/rocm_agent_enumerator" 2>/dev/null | grep -v '^gfx000$' | head -1 || true)"
fi
if [ -z "${ARCH}" ]; then
  ARCH="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]\+' | head -1 || true)"
fi
[ -n "${ARCH}" ] || { echo "could not detect the GPU arch — set XGBOOST_ROCM_ARCH (e.g. gfx942)" >&2; exit 1; }

echo "Building AMD XGBoost: ref=${REF} arch=${ARCH} rocm=${ROCM_PATH} jobs=${JOBS}"

if [ -d "${SRC}/.git" ]; then
  git -C "${SRC}" fetch --depth 1 origin "${REF}"
  git -C "${SRC}" checkout -q FETCH_HEAD
  git -C "${SRC}" submodule update --init --recursive --depth 1
else
  git clone --recursive --depth 1 --branch "${REF}" "${REPO}" "${SRC}"
fi

cmake -S "${SRC}" -B "${SRC}/build" \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="${ARCH}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${SRC}/build" --parallel "${JOBS}"

# XGBoost >= 2.0 builds libxgboost from source during `pip install .`; point it
# at the shared object we just built instead of compiling the whole thing twice.
"${PYTHON}" -m pip install --no-input --force-reinstall \
  --config-settings use_system_libxgboost=True "${SRC}/python-package"

"${PYTHON}" - <<'PY'
import xgboost
info = xgboost.build_info()
print(f"xgboost {xgboost.__version__}  USE_HIP={info.get('USE_HIP')}  USE_CUDA={info.get('USE_CUDA')}")
if not info.get("USE_HIP"):
    raise SystemExit("build_info() does not report USE_HIP — the ROCm build did not take")
print("AMD XGBoost installed — the fraud heads will train with device=\"cuda\" on the AMD GPU.")
PY
