# pipelines/ — the Process layer (AMP jobs)

The two data/model jobs of the AMP chain. Both are standalone scripts run from the project
root, with `parent.parent` root discovery and notebook-safe `__file__` guards.

| Path | Contents | AMP task |
|------|----------|----------|
| `fetch_model.py` | Stages the TFM checkpoint into `models/` and the blueprint `src/` package into the repo root (idempotent; LFS-pointer aware). | 3 — Fetch model |
| `prepare_data.py` | Builds transaction splits and writes VAST/S3 Parquet + Impala DDL; also launched from the UI via `tfm_demo/jobs.py` as a subprocess. | 4 — Prepare data |

## Conventions

- Scripts stay **one directory below root** (their `parent.parent` root derivation depends
  on it).
- Heavy pandas work stays in the subprocess boundary (`jobs.py` design) — don't inline it
  into the backend process.
