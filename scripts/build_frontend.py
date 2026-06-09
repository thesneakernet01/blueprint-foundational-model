# SPDX-License-Identifier: Apache-2.0
"""Build the React SPA inside a CML runtime, which ships no Node toolchain.

Node + npm are installed into a **user-local** prefix (default
`~/.local/node`), never a global/system location, so no root or `sudo` is
needed and nothing outside the user's home is touched. The downloaded Node's
`bin/` is put on PATH only for the child npm processes, then we run
`npm ci && npm run build` in `frontend/`, producing `frontend/dist`.

`ensure_node()` / `npm_env()` are reused by `scripts/serve_app.py` to run the
Vite preview server from the same user-local toolchain.

Run:
    python scripts/build_frontend.py
Override the toolchain location / version with $NODE_PREFIX / $NODE_VERSION.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

NODE_VERSION = os.environ.get("NODE_VERSION", "v20.18.0")  # current LTS


def project_root() -> Path:
    # `__file__` is undefined when exec'd as a notebook cell; fall back to cwd.
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


def _node_dist() -> tuple[str, str]:
    """(dirname, url) of the official prebuilt Node tarball for this machine."""
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64",
            "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        sys.exit(f"build_frontend: unsupported architecture {machine!r}")
    name = f"node-{NODE_VERSION}-linux-{arch}"
    return name, f"https://nodejs.org/dist/{NODE_VERSION}/{name}.tar.xz"


def ensure_node() -> Path:
    """Install Node user-locally if absent; return its bin/ directory."""
    # Default to ~/.local/node — a per-user prefix. Never /usr or a global npm
    # prefix, so this needs no elevated privileges.
    prefix = Path(os.environ.get("NODE_PREFIX") or (Path.home() / ".local" / "node"))
    name, url = _node_dist()
    node_bin = prefix / name / "bin"
    if (node_bin / "node").exists():
        print(f"build_frontend: reusing Node at {node_bin}")
        return node_bin

    prefix.mkdir(parents=True, exist_ok=True)
    tarball = prefix / f"{name}.tar.xz"
    print(f"build_frontend: downloading {url}")
    urllib.request.urlretrieve(url, tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(prefix)
    tarball.unlink(missing_ok=True)
    print(f"build_frontend: installed Node {NODE_VERSION} -> {node_bin}")
    return node_bin


def npm_env(node_bin: Path) -> dict[str, str]:
    """A copy of os.environ with the user-local Node bin/ prepended to PATH."""
    env = dict(os.environ)
    env["PATH"] = f"{node_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def build() -> None:
    """Install deps and produce frontend/dist from the user-local toolchain."""
    env = npm_env(ensure_node())
    frontend = project_root() / "frontend"
    install = ["npm", "ci"] if (frontend / "package-lock.json").exists() else ["npm", "install"]
    subprocess.run(install, cwd=frontend, env=env, check=True)
    subprocess.run(["npm", "run", "build"], cwd=frontend, env=env, check=True)
    print("build_frontend: SPA built -> frontend/dist")


if __name__ == "__main__":
    build()
