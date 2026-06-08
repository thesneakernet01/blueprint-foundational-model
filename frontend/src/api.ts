// Typed client for the TFM live-demo FastAPI backend (../app.py).
// The SPA is served standalone, so it talks to the backend cross-origin; the
// base URL must be baked in at build time via VITE_API_BASE.
//
// In CML the UI and API are separate applications on different subdomains, so
// VITE_API_BASE has to be the absolute tfm-api URL — there is no working
// default. `||` (not `??`) is deliberate: an empty string (the AMP env-var
// default) collapses to the localhost dev default rather than silently
// producing a same-origin relative base that would hit the UI app, not the API.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export type Mode = 'real' | 'demo-fallback' | string;

export interface StatusResp {
  mode: Mode;
  gpu: boolean;
  detail: string;
  model_dir: string;
}

export interface ModelSummary {
  key: 'raw' | 'embed' | 'combined' | string;
  label: string;
  test_auc: number | null;
  test_ap: number | null;
}

export interface Lift {
  embed_auc_pct: number | null;
  embed_ap_pct: number | null;
  combined_auc_pct: number | null;
  combined_ap_pct: number | null;
}

export interface Summary {
  placeholder: boolean;
  n_raw_features: number;
  pca_dim: number;
  models: ModelSummary[];
  lift: Lift;
  note?: string;
}

// The backend transaction payload uses space-separated alias keys.
export type TxnPayload = {
  Amount: string;
  'Merchant Name': string;
  'Merchant City': string;
  'Merchant State': string;
  'Use Chip': string;
  MCC: number;
  Zip: string;
  Time: string;
  Year: number;
  Month: number;
  Day: number;
  Card: number;
  User: number;
};

export interface Example {
  label: string;
  is_fraud: boolean | null;
  txn: Record<string, string | number>;
}

export interface UmapPoint {
  x: number;
  y: number;
  fraud: number;
}

export interface Scores {
  raw: number;
  embed: number;
  combined: number;
}

export interface ScoreResp {
  mode: Mode;
  tokens: string[];
  embedding_dim: number;
  scores: Scores;
  position: { x: number; y: number } | null;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const getStatus = () => getJSON<StatusResp>('/api/status');
export const getSummary = () => getJSON<Summary>('/api/summary');
export const getExamples = () => getJSON<Example[]>('/api/examples');
export const getUmap = () => getJSON<UmapPoint[]>('/api/umap');

export async function postScore(txn: TxnPayload): Promise<ScoreResp> {
  const res = await fetch(`${API_BASE}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(txn),
  });
  if (!res.ok) throw new Error(`/api/score → ${res.status} ${res.statusText}`);
  return (await res.json()) as ScoreResp;
}

// ---- artifact export (runs on the GPU backend) -----------------------------

export type ExportState = 'idle' | 'running' | 'done' | 'error';

export interface ExportStatus {
  state: ExportState;
  log: string[];
  summary: Summary | null;
  error: string | null;
  elapsed_sec: number | null;
  engine_mode: Mode;
}

/** Kick off an export. `started` is false (HTTP 409) if one is already running. */
export async function startExport(): Promise<{ started: boolean } & ExportStatus> {
  const res = await fetch(`${API_BASE}/api/export`, { method: 'POST' });
  if (res.status !== 202 && res.status !== 409) {
    throw new Error(`/api/export → ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

export const getExportStatus = () => getJSON<ExportStatus>('/api/export/status');
