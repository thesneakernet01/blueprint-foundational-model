import { Check, Loader2 } from 'lucide-react';
import type { RegistryStatus, RunRecord } from '../../api';

interface Props {
  runs: RunRecord[];
  registry: RegistryStatus | null;
  exportRunning: boolean;
}

type ChipState = 'done' | 'active' | 'pending' | 'unavailable';

/** Develop → Train → Register → Deploy → Observe, each stage backed by the
 *  Cloudera AI primitive that provides it. This is the "how Cloudera improves
 *  the process" story in one strip. */
export default function LifecycleStrip({ runs, registry, exportRunning }: Props) {
  const latest = runs.length ? runs[runs.length - 1] : null;
  const registered = (registry?.versions.length ?? 0) > 0 || !!latest?.registry.registered;
  const deployed = registry?.deployment?.status === 'deployed' || !!latest?.registry.deployed;
  const offCml = registry != null && !registry.available;

  const stages: { label: string; primitive: string; state: ChipState; note?: string }[] = [
    { label: 'Develop', primitive: 'Project · Jobs', state: 'done' },
    {
      label: 'Train',
      primitive: 'GPU job (in-app)',
      state: exportRunning ? 'active' : runs.length ? 'done' : 'pending',
      note: runs.length ? `${runs.length} run${runs.length === 1 ? '' : 's'}` : undefined,
    },
    {
      label: 'Register',
      primitive: 'Model Registry',
      state: offCml ? 'unavailable' : registered ? 'done' : 'pending',
      note: offCml ? (registry?.reason ?? undefined) : undefined,
    },
    {
      label: 'Deploy',
      primitive: 'Model endpoint',
      state: offCml ? 'unavailable' : deployed ? 'done' : 'pending',
      note: offCml ? (registry?.reason ?? undefined) : undefined,
    },
    {
      label: 'Observe',
      primitive: 'Application (this dashboard)',
      state: runs.length ? 'done' : 'pending',
    },
  ];

  const chip = (state: ChipState) =>
    state === 'done'
      ? 'border-status-green/40 text-status-green'
      : state === 'active'
        ? 'border-accent/60 text-accent'
        : state === 'unavailable'
          ? 'border-surface-4 text-gray-400 opacity-60'
          : 'border-surface-4 text-gray-500';

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Lifecycle on Cloudera AI</h3>
      <div className="space-y-2">
        {stages.map((s) => (
          <div
            key={s.label}
            title={s.note}
            className={`flex items-center gap-2.5 border rounded-md px-2.5 py-1.5 ${chip(s.state)}`}
          >
            {s.state === 'done' ? (
              <Check className="w-3.5 h-3.5 shrink-0" />
            ) : s.state === 'active' ? (
              <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" />
            ) : (
              <span className="w-3.5 h-3.5 shrink-0 flex items-center justify-center">
                <span className="w-1.5 h-1.5 rounded-full bg-current opacity-50" />
              </span>
            )}
            <span className="text-xs font-medium">{s.label}</span>
            <span className="ml-auto text-[10px] text-gray-500 truncate">{s.primitive}</span>
          </div>
        ))}
      </div>
      {registry != null && !registry.available && (
        <p className="mt-2.5 text-[10px] text-gray-400 leading-snug">
          Register &amp; Deploy light up when this app runs on Cloudera AI — {registry.reason}.
        </p>
      )}
    </div>
  );
}
