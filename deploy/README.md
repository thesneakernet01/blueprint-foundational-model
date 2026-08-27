# deploy/ — provisioning · toolchain · connectivity probes

Stands up the environment this accelerator runs in: pinned Python installs, a user-local
Node toolchain, connectivity diagnostics for the external platforms (VAST, NEXUS), and the
Docker path.

## What goes here

| Path | Contents | Automated by |
|------|----------|--------------|
| `install_deps.py` | Pinned Python dependencies (GPU-aware; `requirements-*.txt` at root). | AMP task 1 |
| `build_frontend.py` | User-local Node LTS + `npm ci && npm run build` → `app/frontend/dist`; exports `ensure_node`/`npm_env`/`project_root` reused by `app/serve_app.py`. | AMP task 2 |
| `vast_probe.py` · `vast_upload_probe.py` · `vast_s3a_debug.py` | VAST S3 connectivity/upload/S3a diagnostics. | manual |
| `nexus_probe.py` | NEXUS LTM endpoint probe. | manual |
| `docker/` | `backend.Dockerfile` · `frontend.Dockerfile` · `nginx.conf` (used by the root `docker-compose.yml`). | `docker compose up -d --build` |

## Environment variables

- `.vast.env` (untracked) — `VAST_ACCESS_KEY` / `VAST_SECRET_KEY` + endpoint settings.
- `CDSW_APP_PORT` / `BACKEND_PORT` — Application ports; GPU selection via the CML runtime.
- Full reference: [`../docs/impala-vast-s3-config.md`](../docs/impala-vast-s3-config.md).

## Conventions

- Version pins are load-bearing on the GPU runtime — change `requirements-*.txt`
  deliberately.
- Probes are **diagnostics**, never imported by the app.
