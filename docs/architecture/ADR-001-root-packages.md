# ADR-001 — Blueprint packages and model artifacts stay at the repo root

**Status:** accepted · **Date:** 2026-08

## Context

The inference stack is built from two Python packages imported as top-level names:

- `src/` — the NVIDIA blueprint package (tokenizer, decoder inference), **staged by
  `pipelines/fetch_model.py`** which writes files to `<root>/src/...` and warns that
  without it the export "dies with 'No module named src'".
- `tfm_demo/` — the backend package, imported by the root `app.py` entrypoint
  (`uvicorn app:app` semantics) and by `export_for_demo.py`.

Both packages are resolved from the project root as the working directory (CML sessions,
jobs, and the Application all start there). `models/` is written by `fetch_model.py` and
read CWD-relative. Moving any of these under a template layer would break the import
contract and the blueprint-staging logic for zero functional gain.

## Decision

1. Keep `src/`, `tfm_demo/`, `app.py`, `export_for_demo.py`, `models/`,
   `requirements-*.txt`, `docker-compose.yml`, and the untracked `.vast.env` at the repo
   root, annotated in the root README tree.
2. Split the former `scripts/` by layer: install/build/probes → `deploy/`, data/model jobs
   → `pipelines/`, the Application entry → `app/serve_app.py` (which imports the Node
   helpers from `deploy/build_frontend.py`).
3. `frontend/` → `app/frontend/`; `docker/` → `deploy/docker/`.

## Consequences

- `.project-metadata.yaml`'s five task scripts point at the new `deploy/`, `pipelines/`,
  and `app/` paths.
- `tfm_demo/jobs.py` launches `pipelines/prepare_data.py` (was `scripts/…`).
- All scripts keep their `parent.parent` root derivation — every new home is exactly one
  directory below the root, so no depth changes were needed.
