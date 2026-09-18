// Typed client for the TFM live-demo FastAPI backend (../app.py).
// The UI and API ship as one CML application: Vite owns the public port and
// proxies /api/* to the backend on 127.0.0.1 (see frontend/vite.config.ts).
// So the SPA talks to its own origin with relative URLs — no cross-origin, no
// build-time API URL. VITE_API_BASE remains an optional override for the rare
// case of pointing at a separately-hosted backend.
const API_BASE = import.meta.env.VITE_API_BASE || '';

export type Mode = 'real' | 'demo-fallback' | string;
export type GpuBackend = 'cuda' | 'rocm' | 'cpu' | string;

export interface StatusResp {
  mode: Mode;
  gpu: boolean;
  gpu_backend: GpuBackend;
  /** Device the XGBoost fraud heads train/score on: 'cuda' (NVIDIA CUDA build,
   *  or AMD's HIP build — HIP keeps the 'cuda' device string) or 'cpu'. */
  xgb_device: 'cuda' | 'cpu' | string;
  /** True when those heads really run on the GPU. On ROCm this is false unless
   *  AMD's HIP build of XGBoost is installed, even though gpu is true. */
  xgb_gpu: boolean;
  /** Human-readable why — e.g. "AMD ROCm/HIP build (xgboost 3.2.0)". */
  xgb_detail: string;
  detail: string;
  model_dir: string;
}

export interface ModelSummary {
  key: 'raw' | 'embed' | 'combined' | 'nexus' | string;
  label: string;
  test_auc: number | null;
  test_ap: number | null;
  /** True when the metrics came from the NEXUS stub, not a real model. */
  stub?: boolean;
}

export interface Lift {
  embed_auc_pct: number | null;
  embed_ap_pct: number | null;
  combined_auc_pct: number | null;
  combined_ap_pct: number | null;
  /** Present only when the NEXUS head ran during the export. */
  nexus_auc_pct?: number | null;
  nexus_ap_pct?: number | null;
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
  /** Where this row's batch-path embedding projects on the UMAP map — the UI
   *  overlays it as a ring so live-vs-batch drift is visible (diagnostic). */
  expected_position?: { x: number; y: number } | null;
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
  /** Key present only when the NEXUS head is configured; null = the remote
   *  call timed out or failed for this transaction. */
  nexus?: number | null;
}

/** NEXUS side-channel riding on the score response (present iff configured). */
export interface NexusInfo {
  status: 'ok' | 'timeout' | 'unavailable';
  latency_ms: number | null;
}

export interface ScoreResp {
  mode: Mode;
  tokens: string[];
  embedding_dim: number;
  scores: Scores;
  position: { x: number; y: number } | null;
  nexus?: NexusInfo;
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
  /** Current pipeline stage id (check|embed|pca|train|nexus|artifacts|reload|done)
   *  and the stages already completed — drives the animated pipeline flow. */
  stage?: string | null;
  stages_done?: string[];
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

// ---- run history + Model Registry (Model Lifecycle dashboard) ---------------

/** The progressive training budget a run was (or will be) granted. */
export interface RunBudget {
  tier: number;
  embed_max: number;
  xgb_scale: number;
  /** Per-head boosting rounds actually used (recorded runs only). */
  n_estimators?: Record<string, number>;
}

export interface RunRegistry {
  registered: boolean;
  model_version: number | null;
  version_id: string | null;
  registered_at: string | null;
  deployed: boolean;
  deployed_at: string | null;
}

/** Per-round validation AUC — the gradient-boosting "loss curve". */
export interface EvalCurve {
  rounds: number[];
  auc: number[];
  /** 1-based round where early stopping kept the model. */
  best_round: number;
}

/** Small per-run training diagnostics recorded by the export. */
export interface RunDiagnostics {
  eval_curves?: Record<string, EvalCurve>;
  /** Test-set score histograms for the combined head. */
  separation?: { edges: number[]; legit: number[]; fraud: number[] };
  importance?: { name: string; importance: number; kind: 'raw' | 'embedding' }[];
}

export interface RunRecord {
  run: number;
  run_id: string;
  started_at: string;
  finished_at: string;
  duration_sec: number;
  budget: RunBudget;
  models: ModelSummary[];
  lift: Lift;
  nexus: boolean;
  registry: RunRegistry;
  /** Absent on runs recorded before diagnostics existed. */
  diagnostics?: RunDiagnostics;
}

export interface RunsResp {
  runs: RunRecord[];
  next_budget: RunBudget;
  schedule: RunBudget[];
}

export const getRuns = () => getJSON<RunsResp>('/api/runs');

/** Clear the run history so the demo replays from tier 0. */
export async function postRunsReset(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/runs/reset`, { method: 'POST' });
  if (!res.ok) throw new Error(`/api/runs/reset → ${res.status} ${res.statusText}`);
}

export interface RegistryVersion {
  version: number | null;
  version_id: string | null;
  created_at: string;
}

/** The register/deploy background job (same shape as the export job). */
export interface RegistryJob {
  state: ExportState;
  action: 'register' | 'deploy' | null;
  log: string[];
  error: string | null;
  elapsed_sec: number | null;
  stage?: string | null;
  stages_done?: string[];
}

export interface RegistryStatus {
  available: boolean;
  /** Why the registry is unavailable (off-CML, no APIv2 key, …). */
  reason: string | null;
  model_name: string;
  artifacts_ready: boolean;
  artifacts_reason: string | null;
  versions: RegistryVersion[];
  model: { id: string; name: string } | null;
  deployment: {
    build_status: string | null;
    status: string | null;
    url: string | null;
  } | null;
  job: RegistryJob;
}

export const getRegistryStatus = () => getJSON<RegistryStatus>('/api/registry');

async function postRegistryAction(path: string): Promise<{ started: boolean }> {
  const res = await fetch(`${API_BASE}${path}`, { method: 'POST' });
  const body = await res.json().catch(() => null);
  if (res.status === 503 || res.status === 409) {
    throw new Error(body?.error ?? `${path} → ${res.status}`);
  }
  if (res.status !== 202) {
    throw new Error(`${path} → ${res.status} ${res.statusText}`);
  }
  return body;
}

/** Register the latest exported bundle as a new Model Registry version. */
export const postRegister = () => postRegistryAction('/api/registry/register');
/** Build + deploy the newest registered version as a CML Model endpoint. */
export const postDeploy = () => postRegistryAction('/api/registry/deploy');

// ---- NEXUS head mode (fourth model card) ------------------------------------

export type NexusMode = 'off' | 'stub' | 'live';

export interface NexusSettings {
  mode: NexusMode;
  /** Whether the live-mode env config (endpoint + staging bucket) is present. */
  live_ready: boolean;
  target: string;
}

export const getNexusSettings = () => getJSON<NexusSettings>('/api/nexus');

/** Persist the NEXUS head mode. Applies immediately; the metrics card still
 *  needs a re-export to (dis)appear, while the score bar reacts on next run. */
export async function postNexusSettings(mode: NexusMode): Promise<NexusSettings> {
  const res = await fetch(`${API_BASE}/api/nexus`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(body?.error ?? `/api/nexus → ${res.status} ${res.statusText}`);
  }
  return body;
}

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

/** Drop the stored settings for one backend (or everything, backend choice
 *  included) so the env-var defaults apply again; returns the result. */
export async function deleteDataSettings(scope: DataBackend | 'all'): Promise<DataSettings> {
  const res = await fetch(`${API_BASE}/api/data?scope=${scope}`, { method: 'DELETE' });
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
