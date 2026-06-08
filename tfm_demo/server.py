# SPDX-License-Identifier: Apache-2.0
"""Process entrypoint — resolves host/port (Cloudera-aware) and runs uvicorn."""

from __future__ import annotations

from .app import app
from .config import server_host_port


def main() -> None:
    import asyncio

    import uvicorn

    host, port = server_host_port()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # Plain process (python app.py / a CML Job): own the loop.
        server.run()
    else:
        # Hosted inside an already-running loop (a CML Application backed by an
        # ipykernel/JupyterLab runtime). uvicorn.run()'s asyncio.run() refuses
        # to nest, so schedule serve() on the existing loop, which the kernel
        # keeps alive.
        loop.create_task(server.serve())


if __name__ == "__main__":
    main()
