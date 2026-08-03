import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, AlertCircle, CheckCircle, Loader2, Play, X } from 'lucide-react';
import {
  getExportStatus,
  getNexusSettings,
  postNexusSettings,
  startExport,
  type ExportState,
  type ExportStatus,
  type NexusMode,
  type NexusSettings,
  type ResourceSample,
} from '../api';

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called once an export finishes successfully so the dashboard can refresh. */
  onExported: () => void;
}

const POLL_MS = 1500;

const STATE_META: Record<ExportState, { label: string; cls: string }> = {
  idle: { label: 'Idle', cls: 'bg-surface-4 text-gray-400' },
  running: { label: 'Running', cls: 'bg-accent/20 text-accent' },
  done: { label: 'Complete', cls: 'bg-status-green/20 text-status-green' },
  error: { label: 'Failed', cls: 'bg-status-red/20 text-status-red' },
};

const NEXUS_HINT: Record<NexusMode, string> = {
  off: 'Fourth model card disabled — the classic three-model story.',
  stub: 'Deterministic placeholder scores, clearly tagged "stub" — no AWS needed.',
  live: 'Real scores via the pre-deployed SageMaker endpoint.',
};

export default function ExportDialog({ open, onClose, onExported }: Props) {
  const [status, setStatus] = useState<ExportStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [reqError, setReqError] = useState<string | null>(null);
  const [nexus, setNexus] = useState<NexusSettings | null>(null);
  const [nexusBusy, setNexusBusy] = useState(false);
  const pollRef = useRef<number | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const notifiedRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    try {
      const s = await getExportStatus();
      setStatus(s);
      if (s.state === 'done' || s.state === 'error') {
        stopPolling();
        if (s.state === 'done' && !notifiedRef.current) {
          notifiedRef.current = true;
          onExported();
        }
      }
    } catch (e) {
      setReqError(e instanceof Error ? e.message : 'Failed to read export status');
      stopPolling();
    }
  }, [onExported, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    poll();
    pollRef.current = window.setInterval(poll, POLL_MS);
  }, [poll, stopPolling]);

  // On open: read current status (an export may already be running elsewhere).
  useEffect(() => {
    if (!open) return;
    notifiedRef.current = false;
    setReqError(null);
    getNexusSettings()
      .then(setNexus)
      .catch(() => {});
    getExportStatus()
      .then((s) => {
        setStatus(s);
        if (s.state === 'running') startPolling();
      })
      .catch(() => {});
    return stopPolling;
  }, [open, startPolling, stopPolling]);

  // Auto-scroll the log to the bottom as lines stream in.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status?.log]);

  const run = useCallback(async () => {
    setBusy(true);
    setReqError(null);
    notifiedRef.current = false;
    try {
      const r = await startExport();
      setStatus(r);
      if (!r.started && r.state !== 'running') {
        // Nothing running and didn't start — surface whatever came back.
        setReqError('Could not start export.');
      } else {
        startPolling();
      }
    } catch (e) {
      setReqError(e instanceof Error ? e.message : 'Failed to start export');
    } finally {
      setBusy(false);
    }
  }, [startPolling]);

  const setNexusMode = useCallback(
    async (mode: NexusMode) => {
      setNexusBusy(true);
      setReqError(null);
      try {
        setNexus(await postNexusSettings(mode));
      } catch (e) {
        setReqError(e instanceof Error ? e.message : 'Failed to set NEXUS mode');
      } finally {
        setNexusBusy(false);
      }
    },
    [],
  );

  if (!open) return null;

  const state = status?.state ?? 'idle';
  const running = state === 'running';
  const meta = STATE_META[state];
  const lift = status?.summary && !status.summary.placeholder ? status.summary.lift : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-surface-1 rounded-xl border border-surface-3 shadow-2xl w-[560px] max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-surface-3">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-white">Build artifacts</h2>
            <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${meta.cls}`}>{meta.label}</span>
            {status?.elapsed_sec != null && (
              <span className="font-mono text-[11px] text-gray-500">{status.elapsed_sec}s</span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-500 hover:text-gray-300 rounded-lg hover:bg-surface-3"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* body */}
        <div className="p-5 space-y-4 overflow-y-auto">
          <p className="text-xs text-gray-400 leading-relaxed">
            Trains the XGBoost heads (plus the NEXUS Large Tabular Model head when
            configured), fits PCA + UMAP on the foundation-model
            embeddings, and writes <span className="font-mono text-gray-300">demo_artifacts/</span> on
            the backend. Reads the training splits from the configured Impala database
            (see the Data dialog) and generates the foundation-model embeddings in-app —
            requires the model checkpoint, the Impala split tables, and a GPU. When it
            finishes, the metrics, examples, and embedding map below refresh automatically.
          </p>

          {/* NEXUS head mode — settable here because demo boxes rarely offer
              shell access; persists server-side, applies without a restart. */}
          {nexus && (
            <div className="bg-surface-2 border border-surface-3 rounded-lg p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="text-xs font-medium text-gray-300">
                  NEXUS head{' '}
                  <span className="text-gray-500 font-normal">· Large Tabular Model</span>
                </div>
                <div className="flex rounded-md overflow-hidden border border-surface-4">
                  {(['off', 'stub', 'live'] as NexusMode[]).map((m) => {
                    const active = nexus.mode === m;
                    const liveLocked = m === 'live' && !nexus.live_ready;
                    return (
                      <button
                        key={m}
                        onClick={() => setNexusMode(m)}
                        disabled={nexusBusy || active || liveLocked}
                        title={
                          liveLocked
                            ? 'Live needs NEXUS_ENDPOINT_NAME + NEXUS_S3_BUCKET set on the backend'
                            : undefined
                        }
                        className={`px-2.5 py-1 text-[11px] font-medium transition-colors ${
                          active
                            ? 'bg-status-purple/20 text-status-purple'
                            : liveLocked
                              ? 'text-gray-600 cursor-not-allowed'
                              : 'text-gray-400 hover:text-gray-200 hover:bg-surface-3'
                        }`}
                      >
                        {m}
                      </button>
                    );
                  })}
                </div>
              </div>
              <p className="text-[11px] text-gray-500 mt-1.5">
                {NEXUS_HINT[nexus.mode]}{' '}
                {nexus.mode !== 'off' &&
                  'The score bar reacts on the next inference; the metrics card needs a re-run of this export.'}
              </p>
            </div>
          )}

          {reqError && (
            <div className="px-4 py-3 bg-status-red-dim/30 border border-status-red/40 rounded-lg text-sm text-status-red">
              {reqError}
            </div>
          )}

          {/* live resource monitor */}
          {status?.resources && <ResourceMonitor res={status.resources} live={running} />}

          {/* live log */}
          {status && status.log.length > 0 && (
            <div
              ref={logRef}
              className="bg-surface-0 border border-surface-3 rounded-lg p-3 max-h-56 overflow-y-auto font-mono text-[11px] leading-relaxed text-gray-400 space-y-0.5"
            >
              {status.log.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith('ERROR')
                      ? 'text-status-red'
                      : line.startsWith('Done') || line.startsWith('Lift')
                        ? 'text-status-green'
                        : ''
                  }
                >
                  {line}
                </div>
              ))}
              {running && (
                <div className="flex items-center gap-2 text-accent pt-1">
                  <Loader2 className="w-3 h-3 animate-spin" /> working…
                </div>
              )}
            </div>
          )}

          {/* result summary */}
          {state === 'done' && lift && (
            <div className="bg-surface-2 border border-status-green/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3 text-status-green text-sm">
                <CheckCircle className="w-4 h-4" />
                Artifacts built · engine mode{' '}
                <span className="font-mono">{status?.engine_mode}</span>
              </div>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                <Stat label="Embeddings AP lift" value={lift.embed_ap_pct} />
                <Stat label="Combined AP lift" value={lift.combined_ap_pct} />
                <Stat label="Embeddings AUC lift" value={lift.embed_auc_pct} />
                <Stat label="Combined AUC lift" value={lift.combined_auc_pct} />
                {lift.nexus_ap_pct !== undefined && (
                  <Stat label="NEXUS AP lift" value={lift.nexus_ap_pct} />
                )}
                {lift.nexus_auc_pct !== undefined && (
                  <Stat label="NEXUS AUC lift" value={lift.nexus_auc_pct} />
                )}
              </div>
            </div>
          )}

          {state === 'error' && status?.error && (
            <div className="flex items-start gap-2 px-4 py-3 bg-status-red-dim/30 border border-status-red/40 rounded-lg text-sm text-status-red">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span>{status.error}</span>
            </div>
          )}
        </div>

        {/* footer */}
        <div className="px-5 py-4 border-t border-surface-3 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="bg-surface-3 text-gray-300 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
          >
            Close
          </button>
          <button
            onClick={run}
            disabled={busy || running}
            className="flex items-center gap-2 bg-accent text-white hover:bg-accent/90 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {busy || running ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Running…
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" /> {state === 'done' || state === 'error' ? 'Re-run export' : 'Run export'}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Live CPU / RAM / GPU meters, fed by the job-status poll (~1.5 s).
 *  Shared with DataDialog (the Impala data-load job reports the same shape). */
export function ResourceMonitor({ res, live }: { res: ResourceSample; live: boolean }) {
  const gpuMemPct =
    res.gpu_mem_used_gb != null && res.gpu_mem_total_gb
      ? (res.gpu_mem_used_gb / res.gpu_mem_total_gb) * 100
      : null;
  const ramPct =
    res.ram_used_gb != null && res.ram_total_gb ? (res.ram_used_gb / res.ram_total_gb) * 100 : null;
  const gb = (used: number | null, total: number | null) =>
    used != null && total != null ? `${used.toFixed(1)} / ${total.toFixed(1)} GB` : undefined;

  const meters = [
    { label: 'GPU', pct: res.gpu_util_pct },
    { label: 'GPU mem', pct: gpuMemPct, detail: gb(res.gpu_mem_used_gb, res.gpu_mem_total_gb) },
    { label: 'CPU', pct: res.cpu_pct },
    { label: 'RAM', pct: ramPct, detail: gb(res.ram_used_gb, res.ram_total_gb) },
  ].filter((m) => m.pct != null || m.detail != null);
  if (meters.length === 0) return null;

  return (
    <div className="bg-surface-2 border border-surface-3 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2.5 text-[10px] uppercase tracking-wide text-gray-500">
        <Activity className={`w-3 h-3 ${live ? 'text-accent' : ''}`} />
        Backend resources
        {res.gpu_name && <span className="font-mono normal-case text-gray-400">· {res.gpu_name}</span>}
      </div>
      <div className="grid grid-cols-2 gap-x-5 gap-y-2.5">
        {meters.map((m) => (
          <Meter key={m.label} label={m.label} pct={m.pct} detail={m.detail} />
        ))}
      </div>
    </div>
  );
}

function Meter({ label, pct, detail }: { label: string; pct: number | null; detail?: string }) {
  const clamped = pct == null ? 0 : Math.min(100, Math.max(0, pct));
  const tone =
    pct == null
      ? 'bg-surface-4'
      : clamped > 90
        ? 'bg-status-red'
        : clamped > 75
          ? 'bg-status-amber'
          : 'bg-accent';
  return (
    <div>
      <div className="flex items-center justify-between mb-1 text-[10px]">
        <span className="text-gray-500 uppercase tracking-wide">{label}</span>
        <span className="font-mono text-gray-400">
          {detail ?? (pct != null ? `${pct.toFixed(0)}%` : '—')}
        </span>
      </div>
      <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${tone}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500">{label}</span>
      <span
        className={`font-mono ${value == null ? 'text-gray-500' : value < 0 ? 'text-status-red' : 'text-status-green'}`}
      >
        {value == null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`}
      </span>
    </div>
  );
}
