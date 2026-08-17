import type { RunRecord } from '../../api';
import { fmtDuration, fmtRows } from './theme';

interface Props {
  runs: RunRecord[];
}

const fmt = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(4));

export default function RunsTable({ runs }: Props) {
  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Run history</h3>
      {runs.length === 0 ? (
        <p className="text-[11px] text-gray-400 py-6 text-center">
          Run Build artifacts to record run #1.
        </p>
      ) : (
        <div className="overflow-x-auto max-h-[260px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-surface-4 text-[10px] uppercase tracking-wider">
                <th className="text-left py-1.5 px-2 font-medium">Run</th>
                <th className="text-left py-1.5 px-2 font-medium">Finished</th>
                <th className="text-right py-1.5 px-2 font-medium">Rows/split</th>
                <th className="text-right py-1.5 px-2 font-medium">Rounds</th>
                <th className="text-right py-1.5 px-2 font-medium">AUC</th>
                <th className="text-right py-1.5 px-2 font-medium">AP</th>
                <th className="text-right py-1.5 px-2 font-medium">Duration</th>
                <th className="text-center py-1.5 px-2 font-medium">Registry</th>
              </tr>
            </thead>
            <tbody>
              {[...runs].reverse().map((r) => {
                const combined = r.models.find((m) => m.key === 'combined');
                return (
                  <tr key={r.run_id} className="border-b border-surface-4/50 hover:bg-surface-3/50">
                    <td className="py-1.5 px-2 font-mono text-gray-800">#{r.run}</td>
                    <td className="py-1.5 px-2 text-gray-600 font-mono whitespace-nowrap">
                      {r.finished_at?.replace('T', ' ').replace(/\+.*$/, '') ?? '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-800">
                      {fmtRows(r.budget.embed_max)}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-800">
                      {r.budget.n_estimators?.combined ?? '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-800">
                      {fmt(combined?.test_auc)}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-800">
                      {fmt(combined?.test_ap)}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-600">
                      {fmtDuration(r.duration_sec)}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {r.registry.deployed ? (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-status-green/10 text-status-green">
                          v{r.registry.model_version ?? '?'} · live
                        </span>
                      ) : r.registry.registered ? (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-blue/10 text-brand-blue">
                          v{r.registry.model_version ?? '?'}
                        </span>
                      ) : (
                        <span className="text-gray-400">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="text-[10px] text-gray-400 mt-1.5">
            AUC / AP are the combined head's test-set metrics; the trend chart shows all heads.
          </p>
        </div>
      )}
    </div>
  );
}
