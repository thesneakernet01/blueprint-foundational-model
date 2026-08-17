import { Check, Database, Boxes, Shrink, TreePine, Layers, Package, RefreshCw, Loader2 } from 'lucide-react';
import type { ExportStatus, RunRecord } from '../../api';
import { fmtDuration } from './theme';

interface Props {
  status: ExportStatus | null;
  lastRun: RunRecord | null;
  /** Whether the NEXUS head is part of the pipeline (mode ≠ off). */
  nexusOn: boolean;
}

// The real export stages, in run_export() order. `code` names each stage's
// home module — this panel doubles as the demo script's pipeline visual.
const STAGES = [
  { id: 'check', label: 'Load splits', code: 'tfm_demo/storage.py', icon: Database },
  { id: 'embed', label: 'FM embeddings', code: 'tfm_demo/export.py', icon: Boxes },
  { id: 'pca', label: 'PCA 512→64', code: 'tfm_demo/export.py', icon: Shrink },
  { id: 'train', label: 'Train XGBoost heads', code: 'tfm_demo/export.py', icon: TreePine },
  { id: 'nexus', label: 'NEXUS LTM head', code: 'tfm_demo/nexus.py', icon: Layers, optional: true },
  { id: 'artifacts', label: 'UMAP + artifacts', code: 'demo_artifacts/', icon: Package },
  { id: 'reload', label: 'Hot reload', code: 'tfm_demo/jobs.py', icon: RefreshCw },
] as const;

type StageState = 'done' | 'active' | 'pending';

export default function PipelineFlow({ status, lastRun, nexusOn }: Props) {
  const running = status?.state === 'running';
  const stagesDone = status?.stages_done ?? [];
  const active = running ? status?.stage : null;
  // Idle with a finished run (this server session or any recorded run): show
  // the whole flow completed. Truly fresh: neutral ready state.
  const allDone = !running && (status?.state === 'done' || lastRun != null);

  const stages = STAGES.filter(
    (s) => !('optional' in s && s.optional) || nexusOn || stagesDone.includes(s.id),
  );

  const stateOf = (id: string): StageState => {
    if (running) {
      if (active === id) return 'active';
      return stagesDone.includes(id) ? 'done' : 'pending';
    }
    return allDone ? 'done' : 'pending';
  };

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-300">Training pipeline</h3>
        <span className="text-[11px] text-gray-500 font-mono">
          {running
            ? `running · ${fmtDuration(status?.elapsed_sec)}`
            : lastRun
              ? `Run #${lastRun.run} · ${fmtDuration(lastRun.duration_sec)}`
              : 'ready — run Build artifacts to train'}
        </span>
      </div>

      <div className="flex items-start overflow-x-auto pb-1">
        {stages.map((s, i) => {
          const st = stateOf(s.id);
          const Icon = s.icon;
          return (
            <div key={s.id} className="flex items-start flex-1 min-w-[92px]">
              {i > 0 && (
                <div className="flex-1 pt-4 px-1 min-w-[16px]">
                  <div
                    className={`h-0.5 rounded connector ${
                      st === 'active' ? 'connector-active' : st === 'done' ? 'connector-done' : ''
                    }`}
                  />
                </div>
              )}
              <div className="flex flex-col items-center text-center w-[92px] shrink-0">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center border transition-colors ${
                    st === 'done'
                      ? 'bg-status-green/15 border-status-green/40 text-status-green'
                      : st === 'active'
                        ? 'bg-accent/15 border-accent text-accent animate-pulse-glow-accent'
                        : 'bg-surface-3 border-surface-4 text-gray-600'
                  }`}
                >
                  {st === 'done' ? (
                    <Check className="w-4 h-4" />
                  ) : st === 'active' ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4" />
                  )}
                </div>
                <div
                  className={`mt-1.5 text-[10px] font-medium leading-tight ${
                    st === 'pending' ? 'text-gray-600' : 'text-gray-300'
                  }`}
                >
                  {s.label}
                </div>
                <div className="text-[9px] text-gray-600 font-mono leading-tight">{s.code}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
