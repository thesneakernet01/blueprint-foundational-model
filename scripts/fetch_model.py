# SPDX-License-Identifier: Apache-2.0
"""Fetch the blueprint pieces the demo imports at runtime:

  * the decoder-foundation-model checkpoint -> config.MODEL_DIR, and
  * the blueprint's `src/` package (tokenizer, decoder inference) -> ./src/,
    which config.py already puts on sys.path. A standalone CML project has
    neither; without src/ the export dies with "No module named src".

Both live in the NVIDIA TFM blueprint repo, the weights tracked with Git LFS
(~56 MB). Rather than require git/git-lfs in the CML runtime, we download the
files over plain HTTPS:
  * regular files from raw.githubusercontent.com,
  * LFS-tracked files (e.g. the .safetensors weights) from GitHub's `media`
    endpoint, which resolves the LFS pointer to the real blob.

Idempotent: skips the checkpoint when a non-pointer copy is present, and src/
when it is already importable (in-project or the nested-in-blueprint layout).
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

RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REF}"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{REF}"
RAW = f"{RAW_BASE}/{SUBDIR}"
MEDIA = f"{MEDIA_BASE}/{SUBDIR}"

# Fallback if the GitHub contents API is unavailable (e.g. unauthenticated rate
# limit on a shared egress IP). The dir is a standard HF checkpoint layout.
DEFAULT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
]

# The blueprint package the engine/export import (src.tokenizer,
# src.decoder_inference). Fetched recursively into PROJECT_ROOT/src; the
# fallback list is only the modules we import directly, so warn when used.
SRC_DEST = _ROOT / "src"
SRC_DEFAULT_FILES = ["src/__init__.py", "src/tokenizer.py", "src/decoder_inference.py"]


def _list_dir(subdir: str) -> list[dict]:
    """Entries ({name, path, type}) of one blueprint dir via the contents API."""
    api = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{subdir}?ref={REF}"
    with urllib.request.urlopen(api, timeout=30) as r:
        return json.load(r)


def _list_files() -> list[str]:
    try:
        return [it["name"] for it in _list_dir(SUBDIR) if it["type"] == "file"]
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


def _src_paths() -> list[str]:
    """All file paths under the blueprint's src/, recursively. Falls back to the
    two modules we import (plus __init__) if the contents API is unavailable —
    enough to run, but warn in case src/ grows internal imports upstream."""
    try:
        paths, stack = [], ["src"]
        while stack:
            for it in _list_dir(stack.pop()):
                if it["type"] == "file":
                    paths.append(it["path"])
                elif it["type"] == "dir":
                    stack.append(it["path"])
        return paths
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        print(f"fetch_model: contents API unavailable ({exc}); fetching only "
              f"{SRC_DEFAULT_FILES} — rerun later if src/ has more modules")
        return SRC_DEFAULT_FILES


def fetch_src() -> None:
    # Already importable? Either fetched previously (in-project) or the demo is
    # nested inside the blueprint repo (src/ in the parent, see config.py).
    for root in (_ROOT, _ROOT.parent):
        if all((root / "src" / m).exists()
               for m in ("tokenizer/__init__.py", "decoder_inference.py")):
            print(f"fetch_model: blueprint src/ already present at {root / 'src'}")
            return

    paths = _src_paths()
    print(f"fetch_model: downloading {len(paths)} src files -> {SRC_DEST}")
    for p in paths:
        data = _get(f"{RAW_BASE}/{p}")
        if _is_lfs_pointer(data):
            data = _get(f"{MEDIA_BASE}/{p}")
        dest = SRC_DEST.parent / p                 # p is repo-relative ("src/...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"  {p}: {len(data):,} bytes")
    print(f"fetch_model: blueprint src/ staged at {SRC_DEST}")


if __name__ == "__main__":
    fetch()
    fetch_src()
