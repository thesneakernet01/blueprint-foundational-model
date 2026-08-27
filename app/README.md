# app/ — the inference cockpit (Serve layer)

One CML Application: `serve_app.py` supervises uvicorn (the API, bound to
`127.0.0.1:$BACKEND_PORT`) behind a Vite preview server on `$CDSW_APP_PORT` serving the
React SPA.

| Path | Contents |
|------|----------|
| `serve_app.py` | The Application entry point (AMP task 5) — builds `frontend/dist` if missing (Node helpers imported from `deploy/build_frontend.py`), starts both processes, supervises them. |
| `frontend/` | The React SPA cockpit: live scoring, AUC/AP lift, UMAP, data dialog, settings. |

## Run it

```bash
python app/serve_app.py            # both processes (build on demand)
cd app/frontend && npm run dev     # frontend dev server
docker compose up -d --build       # containerized (repo root; deploy/docker/)
```

## Conventions

- The backend implementation is the root `tfm_demo/` package
  ([ADR-001](../docs/architecture/ADR-001-root-packages.md)) — `app/` holds the Serve
  wiring, not the engine.
