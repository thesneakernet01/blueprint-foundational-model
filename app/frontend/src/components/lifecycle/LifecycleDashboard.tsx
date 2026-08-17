import { useCallback, useEffect, useRef, useState } from 'react';
import { Hammer, RotateCcw } from 'lucide-react';
import {
  getExportStatus,
  getNexusSettings,
  getRegistryStatus,
  getRuns,
  postRunsReset,
  type ExportStatus,
  type RegistryStatus,
  type RunsResp,
} from '../../api';
import PipelineFlow from './PipelineFlow';
import LifecycleStrip from './LifecycleStrip';
import MetricTrend from './MetricTrend';
import BudgetChart from './BudgetChart';
import RunsTable from './RunsTable';
import RegistryPanel from './RegistryPanel';
import { fmtRows } from './theme';

interface Props {
  /** Open the Build-artifacts dialog (exports are started there). */
  onBuild: () => void;
}

const POLL_MS = 1500;

export default function LifecycleDashboard({ onBuild }: Props) {
  const [runsResp, setRunsResp] = useState<RunsResp | null>(null);
  const [registry, setRegistry] = useState<RegistryStatus | null>(null);
  const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null);
  const [nexusOn, setNexusOn] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const prevExportState = useRef<string | null>(null);

  const refreshRuns = useCallback(() => {
    getRuns().then(setRunsResp).catch(() => {});
    getRegistryStatus().then(setRegistry).catch(() => {});
  }, []);

  useEffect(() => {
    refreshRuns();
    getNexusSettings()
      .then((s) => setNexusOn(s.mode !== 'off'))
      .catch(() => {});
  }, [refreshRuns]);

  // Poll the export job so the pipeline animation tracks a run started from
  // the Build dialog (mounted at App level, so it keeps running across tabs).
  useEffect(() => {
    let live = true;
    const tick = () =>
      getExportStatus()
        .then((s) => {
          if (!live) return;
          setExportStatus(s);
          // A run just finished (or failed): pick up the new history/registry.
          if (prevExportState.current === 'running' && s.state !== 'running') {
            refreshRuns();
          }
          prevExportState.current = s.state;
        })
        .catch(() => {});
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [refreshRuns]);

  // Poll the registry while a register/deploy job streams its log.
  useEffect(() => {
    if (registry?.job.state !== 'running') return;
    const id = setInterval(() => {
      getRegistryStatus().then(setRegistry).catch(() => {});
    }, POLL_MS);
    return () => clearInterval(id);
  }, [registry?.job.state]);

  const runs = runsResp?.runs ?? [];
  const lastRun = runs.length ? runs[runs.length - 1] : null;
  const exportRunning = exportStatus?.state === 'running';
  const next = runsResp?.next_budget;

  const doReset = async () => {
    setConfirmReset(false);
    try {
      await postRunsReset();
    } finally {
      refreshRuns();
    }
  };

  return (
    <div className="space-y-6">
      {/* header row: run count, next budget, actions */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="slide-title slide-title-lg">Model Lifecycle</h2>
          <p className="text-xs text-gray-500">
            Train → register → deploy → observe, all inside Cloudera AI Workbench
            {runs.length > 0 && (
              <>
                {' '}
                · <span className="font-mono text-gray-600">{runs.length}</span> run
                {runs.length === 1 ? '' : 's'} recorded
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {next && (
            <span className="text-[11px] text-gray-500">
              Next run:{' '}
              <span className="font-mono text-gray-700">
                {fmtRows(next.embed_max)} rows · {Math.round(next.xgb_scale * 100)}% rounds
              </span>
            </span>
          )}
          <button
            onClick={onBuild}
            disabled={exportRunning}
            className="flex items-center gap-1.5 bg-accent text-white hover:bg-accent/90 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-40"
          >
            <Hammer className="w-3.5 h-3.5" />
            {exportRunning ? 'Training…' : 'Train next run'}
          </button>
          {confirmReset ? (
            <button
              onClick={doReset}
              className="flex items-center gap-1.5 text-status-red bg-status-red/10 border border-status-red/30 hover:bg-status-red/20 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
            >
              Confirm reset
            </button>
          ) : (
            <button
              onClick={() => setConfirmReset(true)}
              disabled={runs.length === 0}
              title="Clear the run history so the demo replays from tier 1"
              className="flex items-center gap-1.5 border border-surface-4 text-gray-600 hover:border-accent/50 hover:text-accent px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset demo
            </button>
          )}
        </div>
      </div>

      {/* row 1: pipeline animation + lifecycle strip */}
      <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 items-start">
        <PipelineFlow status={exportStatus} lastRun={lastRun} nexusOn={nexusOn} />
        <LifecycleStrip runs={runs} registry={registry} exportRunning={exportRunning} />
      </div>

      {/* row 2: metric trend + budget context */}
      <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 items-start">
        <MetricTrend runs={runs} />
        <BudgetChart runs={runs} schedule={runsResp?.schedule ?? []} />
      </div>

      {/* row 3: registry + run table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <RegistryPanel
          registry={registry}
          onRefresh={refreshRuns}
          exportRunning={exportRunning}
        />
        <RunsTable runs={runs} />
      </div>

      {runs.length > 0 && !exportRunning && (
        <p className="text-[10px] text-gray-500">
          Serving artifacts from run #{lastRun?.run} — a reset keeps them live until the next
          training run overwrites them.
        </p>
      )}
    </div>
  );
}
