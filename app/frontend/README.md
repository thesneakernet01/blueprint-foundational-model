# TFM Demo — React Frontend

A standalone React + TypeScript SPA for the Transaction Foundation Model live
fraud-inference demo. It is a pure client: it calls the FastAPI backend
(`../app.py`) over HTTP and renders the result. The backend is what runs the
NVIDIA stack (GPU tokenizer → decoder checkpoint → embeddings → XGBoost heads),
so this app does no model work itself.

```
┌────────────────────────┐        HTTP /api/*         ┌──────────────────────────────┐
│  React SPA (this dir)  │ ─────────────────────────▶ │  FastAPI backend (../app.py) │
│  static, runs anywhere │ ◀───────────────────────── │  GPU node / Cloudera project │
└────────────────────────┘     JSON scores + UMAP      └──────────────────────────────┘
```

Built on the shared Cloudera/NVIDIA dark design system
(`demo-design-template` — surface scale, indigo accent, status trio, Inter +
JetBrains Mono).

## Stack

React 19 · TypeScript · Vite 6 · Tailwind 3 · Recharts · lucide-react.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
```

Point it at the backend with `VITE_API_BASE` (defaults to
`http://localhost:8000`):

```bash
echo 'VITE_API_BASE=http://localhost:8000' > .env.local
```

For a remote GPU box, set `VITE_API_BASE` to that backend's URL (and make sure
the backend's `CORS_ORIGINS` allows this app's origin — see below).

## Build

```bash
npm run build        # → dist/  (static files; host on any static server / CDN)
npm run preview      # serve the production build locally
```

`VITE_API_BASE` is baked in at **build** time, so build with the value that
matches where the backend will live, e.g.:

```bash
VITE_API_BASE=https://<backend-url> npm run build
```

## Backend (the GPU half) in a Cloudera project

The model work happens in `../app.py`, which is meant to run as a **Cloudera ML
Application on a GPU-enabled runtime** (inside the NeMo container, next to the
blueprint repo so it can import `src/` and load the checkpoint). Key wiring:

- **Port** — when `CDSW_APP_PORT` is set, `app.py` binds it on `127.0.0.1` (the
  CML convention); the platform exposes the public Application URL. Otherwise it
  falls back to `PORT`, then `8000`, on `0.0.0.0`.
- **CORS** — `app.py` reads `CORS_ORIGINS` (comma-separated; default `*`). For a
  locked-down deploy set it to this SPA's origin.
- **Mode** — `/api/status` reports `real` (checkpoint + GPU + artifacts loaded)
  or `demo-fallback` (synthetic, clearly labelled). The header badge shows it
  live, so the SPA works against a backend in either mode.

Two common deployment shapes:

1. **Backend in Cloudera, frontend hosted separately** (this split). Build the
   SPA with `VITE_API_BASE=<CML Application URL>`; set the backend's
   `CORS_ORIGINS` to where the SPA is served.
2. **Both as Cloudera Applications** — one GPU Application for `app.py`, one
   static Application serving `dist/`.

## Source map

| File | Role |
|------|------|
| `src/api.ts` | Typed client + response models for every `/api/*` endpoint |
| `src/App.tsx` | Loads status/summary/examples/umap; owns form + score state |
| `src/components/Header.tsx` | Brand + live mode/GPU status dots |
| `src/components/MetricsStrip.tsx` | Per-head AUC/AP cards + AP lift |
| `src/components/TransactionComposer.tsx` | Example picker + editable txn form; maps form ⇄ API payload |
| `src/components/ModelHeads.tsx` | Token strip, 3 probability bars, fraud verdict |
| `src/components/EmbeddingMap.tsx` | UMAP scatter (Recharts, lazy-loaded) + live point |
| `src/components/ExportDialog.tsx` | "Build artifacts" modal: triggers the backend export, streams the log, shows lift; refreshes the dashboard on success |

## Building artifacts from the UI

The header's **Build artifacts** button runs the artifact export **on the
backend** (`POST /api/export`) — it trains the XGBoost heads and fits PCA/UMAP on
the GPU, streaming progress that the dialog polls from `GET /api/export/status`.
When it finishes the engine reloads (→ REAL mode) and the metrics strip,
examples, and embedding map refresh automatically. No CLI step is required;
`export_for_demo.py` remains as an optional command-line equivalent.
