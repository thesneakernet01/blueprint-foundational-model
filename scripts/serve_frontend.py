# SPDX-License-Identifier: Apache-2.0
"""AMP application entrypoint for the React SPA.

Serves the prebuilt static bundle in `frontend/dist/` with SPA-style fallback
(unknown non-file routes -> index.html), bound to the port Cloudera AI injects
(CDSW_APP_PORT) on 127.0.0.1 as the platform expects.

NOTE: `frontend/dist/` must already be built (`cd frontend && npm run build`) with
VITE_API_BASE pointing at the deployed TFM API application's URL. Cloudera ML
Runtimes don't ship Node, so build it beforehand (locally or in CI) and have the
bundle present in the project.
"""

import http.server
import os
import socketserver
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
PORT = int(os.environ.get("CDSW_APP_PORT", "8090"))


class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self):
        requested = (DIST / self.path.lstrip("/")).resolve()
        if not str(requested).startswith(str(DIST)) or not requested.is_file():
            self.path = "/index.html"  # SPA fallback
        return super().do_GET()


if __name__ == "__main__":
    if not (DIST / "index.html").exists():
        raise SystemExit(
            f"{DIST}/index.html not found — build the SPA first "
            "(cd frontend && npm run build)."
        )
    with socketserver.TCPServer(("127.0.0.1", PORT), SPAHandler) as httpd:
        print(f"Serving SPA from {DIST} on 127.0.0.1:{PORT}")
        httpd.serve_forever()
