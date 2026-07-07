import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle,
  Database,
  DownloadCloud,
  Loader2,
  Plug,
  X,
} from 'lucide-react';
import {
  getImpala,
  getPrepareStatus,
  postImpala,
  startPrepare,
  type ExportState,
  type ImpalaCheck,
  type PrepareStatus,
} from '../api';
import { ResourceMonitor } from './ExportDialog';

interface Props {
  open: boolean;
  onClose: () => void;
}

const POLL_MS = 1500;

const STATE_META: Record<ExportState, { label: string; cls: string }> = {
  idle: { label: 'Idle', cls: 'bg-surface-4 text-gray-400' },
  running: { label: 'Loading data', cls: 'bg-accent/20 text-accent' },
  done: { label: 'Loaded', cls: 'bg-status-green/20 text-status-green' },
  error: { label: 'Failed', cls: 'bg-status-red/20 text-status-red' },
};

/**
 * Impala data target: the CML data connection + database that hold the
 * temporal split tables. prepare_data writes them; training/export reads them.
 */
export default function DataDialog({ open, onClose }: Props) {
  const [connection, setConnection] = useState('');
  const [database, setDatabase] = useState('');
  const [check, setCheck] = useState<ImpalaCheck | null>(null);
  const [testing, setTesting] = useState(false);
  const [reqError, setReqError] = useState<string | null>(null);

  const [prep, setPrep] = useState<PrepareStatus | null>(null);
  const [prepBusy, setPrepBusy] = useState(false);
  const pollRef = useRef<number | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    try {
      const s = await getPrepareStatus();
      setPrep(s);
      if (s.state === 'done' || s.state === 'error') {
        stopPolling();
        if (s.state === 'done' && s.summary) setCheck(s.summary);
      }
    } catch (e) {
      setReqError(e instanceof Error ? e.message : 'Failed to read data-load status');
      stopPolling();
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    poll();
    pollRef.current = window.setInterval(poll, POLL_MS);
  }, [poll, stopPolling]);

  // On open: load the saved settings and pick up an already-running load.
  useEffect(() => {
    if (!open) return;
    setReqError(null);
    getImpala()
      .then((s) => {
        setConnection(s.connection);
        setDatabase(s.database);
      })
      .catch(() => {});
    getPrepareStatus()
      .then((s) => {
        setPrep(s);
        if (s.state === 'running') startPolling();
        if (s.state === 'done' && s.summary) setCheck(s.summary);
      })
      .catch(() => {});
    return stopPolling;
  }, [open, startPolling, stopPolling]);

  // Auto-scroll the log as lines stream in.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [prep?.log]);

  const saveAndTest = useCallback(async () => {
    setTesting(true);
    setReqError(null);
    setCheck(null);
    try {
      const r = await postImpala({ connection, database });
      setCheck(r.check);
    } catch (e) {
      setReqError(e instanceof Error ? e.message : 'Failed to save Impala settings');
    } finally {
      setTesting(false);
    }
  }, [connection, database]);

  const runPrepare = useCallback(async () => {
    setPrepBusy(true);
    setReqError(null);
    try {
      const r = await startPrepare();
      setPrep(r);
      if (r.started || r.state === 'running') startPolling();
    } catch (e) {
      setReqError(e instanceof Error ? e.message : 'Failed to start the data load');
    } finally {
      setPrepBusy(false);
    }
  }, [startPolling]);

  if (!open) return null;

  const state = prep?.state ?? 'idle';
  const running = state === 'running';
  const meta = STATE_META[state];
  const configured = connection.trim() !== '' && database.trim() !== '';

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
            <h2 className="text-base font-semibold text-white">Training data · Impala</h2>
            {state !== 'idle' && (
              <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${meta.cls}`}>
                {meta.label}
              </span>
            )}
            {prep?.elapsed_sec != null && running && (
              <span className="font-mono text-[11px] text-gray-500">{prep.elapsed_sec}s</span>
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
            The temporal training splits live in Impala tables{' '}
            <span className="font-mono text-gray-300">train</span>,{' '}
            <span className="font-mono text-gray-300">val_eval</span> and{' '}
            <span className="font-mono text-gray-300">test_eval</span>. Point the app at a CML
            data connection and database, then load TabFormer into it — artifact builds
            (training) read from these tables.
          </p>

          {/* connection form */}
          <div className="space-y-3">
            <label className="block">
              <span className="text-[10px] uppercase tracking-wide text-gray-500">
                CML data connection
              </span>
              <input
                value={connection}
                onChange={(e) => setConnection(e.target.value)}
                placeholder="e.g. default-impala"
                spellCheck={false}
                className="mt-1 w-full bg-surface-0 border border-surface-3 rounded-md px-3 py-1.5 text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent"
              />
            </label>
            <label className="block">
              <span className="text-[10px] uppercase tracking-wide text-gray-500">
                Impala database
              </span>
              <input
                value={database}
                onChange={(e) => setDatabase(e.target.value)}
                placeholder="e.g. tfm_demo"
                spellCheck={false}
                className="mt-1 w-full bg-surface-0 border border-surface-3 rounded-md px-3 py-1.5 text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent"
              />
            </label>
            <button
              onClick={saveAndTest}
              disabled={testing || !configured}
              className="flex items-center gap-2 bg-surface-3 text-gray-300 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {testing ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Connecting… (a suspended
                  warehouse can take a minute)
                </>
              ) : (
                <>
                  <Plug className="w-3.5 h-3.5" /> Save &amp; test connection
                </>
              )}
            </button>
          </div>

          {reqError && (
            <div className="px-4 py-3 bg-status-red-dim/30 border border-status-red/40 rounded-lg text-sm text-status-red">
              {reqError}
            </div>
          )}

          {/* connection / table status */}
          {check && (
            <div
              className={`rounded-lg border p-4 ${
                check.ok
                  ? 'bg-surface-2 border-status-green/30'
                  : 'bg-status-red-dim/30 border-status-red/40'
              }`}
            >
              {check.ok ? (
                <>
                  <div className="flex items-center gap-2 mb-3 text-status-green text-sm">
                    <CheckCircle className="w-4 h-4" />
                    Connected · database{' '}
                    <span className="font-mono">{check.database}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-x-6 text-xs">
                    {Object.entries(check.tables).map(([table, rows]) => (
                      <div key={table} className="flex flex-col gap-0.5">
                        <span className="text-gray-500 font-mono">{table}</span>
                        <span
                          className={`font-mono ${rows ? 'text-gray-200' : 'text-status-amber'}`}
                        >
                          {rows != null ? `${rows.toLocaleString()} rows` : 'missing'}
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="flex items-start gap-2 text-sm text-status-red">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{check.error}</span>
                </div>
              )}
            </div>
          )}

          {/* live resource monitor while the load runs */}
          {running && prep?.resources && <ResourceMonitor res={prep.resources} live />}

          {/* data-load log */}
          {prep && prep.log.length > 0 && (
            <div
              ref={logRef}
              className="bg-surface-0 border border-surface-3 rounded-lg p-3 max-h-56 overflow-y-auto font-mono text-[11px] leading-relaxed text-gray-400 space-y-0.5"
            >
              {prep.log.map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith('ERROR')
                      ? 'text-status-red'
                      : line.startsWith('prepare_data: wrote')
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

          {state === 'error' && prep?.error && (
            <div className="flex items-start gap-2 px-4 py-3 bg-status-red-dim/30 border border-status-red/40 rounded-lg text-sm text-status-red">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span>{prep.error}</span>
            </div>
          )}
        </div>

        {/* footer */}
        <div className="px-5 py-4 border-t border-surface-3 flex items-center justify-between gap-3">
          <p className="text-[11px] text-gray-500 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5" />
            Load downloads ~2.4 GB of TabFormer and re-ingests the tables.
          </p>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="bg-surface-3 text-gray-300 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
            >
              Close
            </button>
            <button
              onClick={runPrepare}
              disabled={prepBusy || running || !configured}
              title={configured ? undefined : 'Save the connection and database first'}
              className="flex items-center gap-2 bg-accent text-white hover:bg-accent/90 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {prepBusy || running ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading…
                </>
              ) : (
                <>
                  <DownloadCloud className="w-3.5 h-3.5" /> Load TabFormer → Impala
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
