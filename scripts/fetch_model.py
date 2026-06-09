# SPDX-License-Identifier: Apache-2.0
"""Fetch the decoder-foundation-model checkpoint into config.MODEL_DIR.

The checkpoint lives in the NVIDIA TFM blueprint repo, tracked with Git LFS
(~56 MB). Rather than require git/git-lfs in the CML runtime, we download the
files over plain HTTPS:
  * regular files from raw.githubusercontent.com,
  * LFS-tracked files (e.g. the .safetensors weights) from GitHub's `media`
    endpoint, which resolves the LFS pointer to the real blob.

Idempotent: skips the download when a non-pointer checkpoint is already present.
Honors $MODEL_DIR (via config). Override the source with $TFM_MODEL_REF (branch
/tag/sha).

Run:  python scripts/fetch_model.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Make `tfm_demo` importable when run as `python scripts/fetch_model.py`.
try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
from tfm_demo.config import MODEL_DIR  # noqa: E402

OWNER = "NVIDIA-AI-Blueprints"
REPO = "transaction-foundation-model"
REF = os.environ.get("TFM_MODEL_REF", "main")
SUBDIR = "models/decoder-foundation-model"

API = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{SUBDIR}?ref={REF}"
RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REF}/{SUBDIR}"
MEDIA = f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{REF}/{SUBDIR}"

# Fallback if the GitHub contents API is unavailable (e.g. unauthenticated rate
# limit on a shared egress IP). The dir is a standard HF checkpoint layout.
DEFAULT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
]


def _list_files() -> list[str]:
    try:
        with urllib.request.urlopen(API, timeout=30) as r:
            return [it["name"] for it in json.load(r) if it["type"] == "file"]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        print(f"fetch_model: contents API unavailable ({exc}); using default list")
        return DEFAULT_FILES


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _is_lfs_pointer(data: bytes) -> bool:
    return data[:40].startswith(b"version https://git-lfs")


def _already_present(names: list[str]) -> bool:
    """True if every file exists and the weights aren't still LFS pointers."""
    for n in names:
        p = MODEL_DIR / n
        if not p.exists():
            return False
        if n.endswith(".safetensors") and _is_lfs_pointer(p.read_bytes()[:64]):
            return False
    return bool(names)


def fetch() -> None:
    names = _list_files()
    if _already_present(names):
        print(f"fetch_model: checkpoint already present at {MODEL_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"fetch_model: downloading {len(names)} files -> {MODEL_DIR}")
    for n in names:
        data = _get(f"{RAW}/{n}")
        if _is_lfs_pointer(data):                 # LFS-tracked: pull the real blob
            data = _get(f"{MEDIA}/{n}")
        (MODEL_DIR / n).write_bytes(data)
        print(f"  {n}: {len(data):,} bytes")
    print(f"fetch_model: checkpoint staged at {MODEL_DIR}")


if __name__ == "__main__":
    fetch()
