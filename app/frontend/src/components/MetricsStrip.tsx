import { useState } from 'react';
import { Scale, TrendingUp } from 'lucide-react';
import type { Lift, Summary } from '../api';
import ParadigmCompare from './ParadigmCompare';

interface Props {
  summary: Summary | null;
}

// The two-worlds framing: classic ML (hand-crafted features into XGBoost),
// hybrid (foundation-model representations into the same XGBoost), and fully
// foundational (the model IS the classifier).
const PARADIGMS = {
  classic: { label: 'classic ml', chip: 'bg-surface-3 text-gray-400' },
  hybrid: { label: 'fm + xgboost', chip: 'bg-accent/10 text-accent' },
  foundation: { label: 'foundation model', chip: 'bg-status-purple/10 text-status-purple' },
} as const;

// Per-model card meta, keyed by summary.models[].key — cards render for
// whichever models the export produced (3 today, 4 with the NEXUS head).
const META: Record<
  string,
  { tag: string; featured: boolean; liftKey?: keyof Lift; paradigm: keyof typeof PARADIGMS }
> = {
  raw: { tag: 'baseline', featured: false, paradigm: 'classic' },
  embed: { tag: 'foundation model', featured: true, liftKey: 'embed_ap_pct', paradigm: 'hybrid' },
  combined: { tag: 'raw + embeddings', featured: false, liftKey: 'combined_ap_pct', paradigm: 'hybrid' },
  nexus: { tag: 'large tabular model', featured: true, liftKey: 'nexus_ap_pct', paradigm: 'foundation' },
};

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
  const [compareOpen, setCompareOpen] = useState(false);
  if (!summary) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-surface-2 rounded-lg border border-surface-3 p-4 h-28 animate-pulse" />
        ))}
      </div>
    );
  }

  const cards = summary.models
    .filter((m) => META[m.key])
    .map((m) => {
      const { tag, featured, liftKey, paradigm } = META[m.key];
      return {
        m,
        tag: m.stub ? `${tag} · stub` : tag,
        lift: liftKey ? (summary.lift[liftKey] ?? null) : null,
        featured,
        paradigm: PARADIGMS[paradigm],
      };
    });
  const grid = cards.length > 3 ? 'sm:grid-cols-2 xl:grid-cols-4' : 'sm:grid-cols-3';

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <button
          onClick={() => setCompareOpen(true)}
          className="flex items-center gap-1.5 text-[11px] text-gray-500 hover:text-gray-200 transition-colors"
        >
          <Scale className="w-3.5 h-3.5" /> Compare paradigms
        </button>
      </div>
      <div className={`grid grid-cols-1 ${grid} gap-4`}>
        {cards.map(({ m, tag, lift, featured, paradigm }) => (
          <div
            key={m.key}
            className={`bg-surface-2 rounded-lg border p-4 ${
              featured ? 'border-accent/40' : 'border-surface-3'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="text-[10px] uppercase tracking-wider text-gray-500">{tag}</div>
              <span
                className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded whitespace-nowrap ${paradigm.chip}`}
              >
                {paradigm.label}
              </span>
            </div>
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
      <ParadigmCompare open={compareOpen} onClose={() => setCompareOpen(false)} summary={summary} />
    </div>
  );
}
