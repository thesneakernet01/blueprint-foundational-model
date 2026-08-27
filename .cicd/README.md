# .cicd/ — build → test → deploy

The **one Git-driven pipeline** shape, mapped onto this accelerator's real commands.
`pipeline.yml` is an illustrative stub, not wired to any CI system.

| Stage | Does | Maps to |
|-------|------|---------|
| **build** | Frontend bundle | `python deploy/build_frontend.py` |
| **test** | Probe + fallback boot | `python deploy/vast_probe.py` (on-cluster) · DEMO-FALLBACK boot of `app/serve_app.py` |
| **deploy** | The AMP | Deploy the repo as a CML AMP (`.project-metadata.yaml`, 5 tasks, GPU runtime) · local: `docker compose up -d --build` |
