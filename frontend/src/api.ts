// Typed client for the TFM live-demo FastAPI backend (../app.py).
// The UI and API ship as one CML application: Vite owns the public port and
// proxies /api/* to the backend on 127.0.0.1 (see frontend/vite.config.ts).
// So the SPA talks to its own origin with relative URLs — no cross-origin, no
// build-time API URL. VITE_API_BASE remains an optional override for the rare
// case of pointing at a separately-hosted backend.
const API_BASE = import.meta.env.VITE_API_BASE || '';

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

/** Live backend resource snapshot piggybacked on the export-status poll.
 *  Any field is null when that probe is unavailable (no GPU, no cgroups). */
export interface ResourceSample {
  cpu_pct: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  gpu_name: string | null;
  gpu_util_pct: number | null;
  gpu_mem_used_gb: number | null;
  gpu_mem_total_gb: number | null;
}

export interface ExportStatus {
  state: ExportState;
  log: string[];
  summary: Summary | null;
  error: string | null;
  elapsed_sec: number | null;
  engine_mode: Mode;
  resources?: ResourceSample | null;
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

// ---- data target (splits stored in Impala tables or on VAST S3) -------------

export type DataBackend = 'impala' | 'vast';

export interface ImpalaSettings {
  connection: string;
  database: string;
}

export interface VastSettings {
  endpoint: string;
  bucket: string;
  prefix: string;
  access_key: string;
  /** Always '' from the server; send '' to keep the stored secret. */
  secret_key: string;
  /** GET-only: whether a secret is already stored server-side. */
  secret_set?: boolean;
}

export interface DataSettings {
  backend: DataBackend;
  impala: ImpalaSettings;
  vast: VastSettings;
}

/** Connectivity + per-split row counts (null = table/object missing). */
export interface DataCheck {
  ok: boolean;
  error: string | null;
  backend: DataBackend;
  /** Human-readable description of the target, e.g. "VAST s3://bucket/prefix @ endpoint". */
  target: string;
  tables: Record<string, number | null>;
}

export const getDataSettings = () => getJSON<DataSettings>('/api/data');

/** Save the storage target, then test it. Slow on a cold warehouse. */
export async function postDataSettings(cfg: DataSettings): Promise<DataSettings & { check: DataCheck }> {
  const res = await fetch(`${API_BASE}/api/data`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(body?.error ?? `/api/data → ${res.status} ${res.statusText}`);
  }
  return body;
}

/** Data-load (TabFormer download → split → storage) job status; same shape as
 *  ExportStatus except summary is the post-load storage check. */
export interface PrepareStatus {
  state: ExportState;
  log: string[];
  summary: DataCheck | null;
  error: string | null;
  elapsed_sec: number | null;
  resources?: ResourceSample | null;
}

export async function startPrepare(): Promise<{ started: boolean } & PrepareStatus> {
  const res = await fetch(`${API_BASE}/api/data/prepare`, { method: 'POST' });
  if (res.status !== 202 && res.status !== 409) {
    throw new Error(`/api/data/prepare → ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

export const getPrepareStatus = () => getJSON<PrepareStatus>('/api/data/prepare/status');
