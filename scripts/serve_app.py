# SPDX-License-Identifier: Apache-2.0
"""Single CML application entrypoint: runs the API and the UI in one container.

CML exposes exactly one public port (CDSW_APP_PORT) per application, and
separate applications can't reach each other over localhost — so the two halves
of the demo run side by side here:

  * the FastAPI backend (uvicorn `app:app`) on 127.0.0.1:$BACKEND_PORT
    (default 7100), private to the container;
  * the Vite preview server on 127.0.0.1:$CDSW_APP_PORT (default 8100), the
    public port, serving the built SPA and proxying /api/* to the backend
    (see frontend/vite.config.ts).

The browser only ever talks to the Vite port; Vite tunnels API calls inward, so
the SPA uses same-origin relative /api/* and needs no VITE_API_BASE. Vite and
its toolchain come from the user-local Node install (build_frontend.ensure_node).

This is a supervisor: it starts both children, and if either exits it tears the
other down and exits with the same code, so CML restarts the whole app cleanly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Import the user-local Node helpers from the sibling build script.
try:
    _SCRIPTS = Path(__file__).resolve().parent
except NameError:
    _SCRIPTS = Path.cwd() / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from build_frontend import build, ensure_node, npm_env, project_root  # noqa: E402


def _ports() -> tuple[int, int]:
    backend = int(os.environ.get("BACKEND_PORT") or 7100)
    public = int(os.environ.get("CDSW_APP_PORT") or os.environ.get("PORT") or 8100)
    return backend, public


def main() -> None:
    root = project_root()
    backend_port, public_port = _ports()

    # The proxy target must agree with what the backend binds.
    os.environ["BACKEND_PORT"] = str(backend_port)

    # The SPA bundle must exist for `vite preview`; build it if a prior Job
    # didn't (e.g. running this entrypoint standalone).
    if not (root / "frontend" / "dist" / "index.html").exists():
        print("serve_app: frontend/dist missing — building it now")
        build()

    node_bin = ensure_node()
    env = npm_env(node_bin)
    env["BACKEND_PORT"] = str(backend_port)

    # Backend: bind the private port directly via uvicorn, bypassing the
    # CDSW_APP_PORT logic in config.server_host_port (that port is the UI's).
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=str(root),
    )
    # Frontend: Vite preview on the public port, proxying /api -> backend.
    frontend = subprocess.Popen(
        ["npm", "run", "preview", "--",
         "--host", "127.0.0.1", "--port", str(public_port), "--strictPort"],
        cwd=str(root / "frontend"),
        env=env,
    )
    print(f"serve_app: backend pid={backend.pid} :{backend_port}  "
          f"frontend pid={frontend.pid} :{public_port}")

    children = {"backend": backend, "frontend": frontend}

    def _shutdown(*_a) -> None:
        for p in children.values():
            if p.poll() is None:
                p.terminate()

    # signal.signal only works on the main thread; under an ipykernel-hosted
    # runtime it may raise — fall back to plain teardown on child exit.
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (_shutdown(), sys.exit(0)))
    except ValueError:
        pass

    try:
        while True:
            for name, p in children.items():
                rc = p.poll()
                if rc is not None:
                    print(f"serve_app: {name} exited ({rc}); shutting down")
                    _shutdown()
                    for q in children.values():
                        try:
                            q.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            q.kill()
                    sys.exit(rc if rc else 1)
            time.sleep(1)
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
