import { useState } from 'react';
import type { Summary } from '../api';
import ParadigmCompare from './ParadigmCompare';

interface Props {
  summary: Summary | null;
}

// Per-model annotation, keyed by summary.models[].key — stats render for
// whichever models the export produced (3 today, 4 with the NEXUS head).
const NOTE: Record<string, string> = {
  raw: 'classic ml · hand-picked columns',
  embed: 'fm embeddings · 64-d pca',
  combined: 'raw ⊕ embeddings',
  nexus: 'large tabular model',
};

const fmt = (v: number | null) => (v == null ? '—' : v.toFixed(4));

export default function MetricsStrip({ summary }: Props) {
  const [compareOpen, setCompareOpen] = useState(false);
  if (!summary) {
    return <div className="panel h-28 animate-pulse" />;
  }

  const models = summary.models.filter((m) => NOTE[m.key]);
  const liftPct = summary.lift.combined_ap_pct;

  return (
    <div className="panel px-6 py-5">
      <div className="flex items-baseline justify-between">
        <h2 className="slide-title">Held-out test set</h2>
        <button
          onClick={() => setCompareOpen(true)}
          className="text-[11px] text-gray-500 hover:text-accent transition-colors"
        >
          Compare paradigms →
        </button>
      </div>

      <div className="mt-4 flex flex-wrap items-stretch divide-x divide-surface-3">
        {models.map((m) => (
          <div key={m.key} className="pr-8 pl-8 first:pl-0 py-1 min-w-[170px]">
            <div className="text-xs text-gray-700">
              {m.label}
              {m.stub && <span className="text-gray-400"> · stub</span>}
            </div>
            <div className="mt-1.5 font-mono text-[27px] leading-none text-ink">
              {fmt(m.test_ap)}
            </div>
            <div className="mt-1.5 text-[11px] text-gray-500">
              avg precision · AUC <span className="font-mono">{fmt(m.test_auc)}</span>
            </div>
            <div className="text-[10px] font-mono text-gray-400">{NOTE[m.key]}</div>
          </div>
        ))}

        {liftPct != null && (
          <div className="pl-8 py-1 ml-auto text-right">
            <div className="text-xs text-gray-700">Foundation-model lift</div>
            <div className="mt-1.5 font-mono text-[27px] leading-none text-accent">
              {liftPct >= 0 ? '+' : ''}
              {liftPct.toFixed(1)}%
            </div>
            <div className="mt-1.5 text-[11px] text-gray-500">
              avg precision, combined vs raw baseline
            </div>
          </div>
        )}
      </div>

      {summary.placeholder && (
        <p className="mt-3 text-[11px] text-gray-500">
          {summary.note ||
            'Placeholder metrics — run Build artifacts on the GPU backend for live numbers.'}
        </p>
      )}
      <ParadigmCompare open={compareOpen} onClose={() => setCompareOpen(false)} summary={summary} />
    </div>
  );
}
