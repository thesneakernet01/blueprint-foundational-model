# SPDX-License-Identifier: Apache-2.0
"""Process entrypoint — resolves host/port (Cloudera-aware) and runs uvicorn."""

from __future__ import annotations

from .app import app
from .config import server_host_port


def main() -> None:
    import uvicorn
    host, port = server_host_port()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
