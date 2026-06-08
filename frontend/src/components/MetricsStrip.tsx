import { TrendingUp } from 'lucide-react';
import type { Summary } from '../api';

interface Props {
  summary: Summary | null;
}

const fmt = (v: number | null) => (v == null ? '—' : v.toFixed(4));
const liftStr = (v: number | null) => (v == null ? '' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`);

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className="font-mono text-gray-200 text-sm">{value}</span>
    </div>
  );
}

export default function MetricsStrip({ summary }: Props) {
  if (!summary) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-surface-2 rounded-lg border border-surface-3 p-4 h-28 animate-pulse" />
        ))}
      </div>
    );
  }

  const byKey = Object.fromEntries(summary.models.map((m) => [m.key, m]));
  const cards = [
    { m: byKey.raw, tag: 'baseline', lift: null as number | null, featured: false },
    { m: byKey.embed, tag: 'foundation model', lift: summary.lift.embed_ap_pct, featured: true },
    { m: byKey.combined, tag: 'raw + embeddings', lift: summary.lift.combined_ap_pct, featured: false },
  ];

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {cards.map(({ m, tag, lift, featured }) => (
          <div
            key={m.key}
            className={`bg-surface-2 rounded-lg border p-4 ${
              featured ? 'border-accent/40' : 'border-surface-3'
            }`}
          >
            <div className="text-[10px] uppercase tracking-wider text-gray-500">{tag}</div>
            <h3 className="text-sm font-medium text-gray-300 mt-1 mb-3">{m.label}</h3>
            <div className="space-y-1.5">
              <StatRow label="ROC-AUC" value={fmt(m.test_auc)} />
              <StatRow label="Avg precision" value={fmt(m.test_ap)} />
            </div>
            {lift != null && (
              <div
                className={`mt-3 inline-flex items-center gap-1 text-xs font-medium ${
                  lift < 0 ? 'text-status-red' : 'text-status-green'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                {liftStr(lift)} AP vs baseline
              </div>
            )}
          </div>
        ))}
      </div>
      {summary.placeholder && (
        <p className="text-[11px] text-gray-600">
          {summary.note || 'Showing placeholder metrics — run export_for_demo.py on the GPU box for live numbers.'}
        </p>
      )}
    </div>
  );
}
