import { AlertTriangle, CheckCircle, Cpu } from 'lucide-react';
import type { ScoreResp, Summary } from '../api';

interface Props {
  result: ScoreResp | null;
  summary: Summary | null;
  scoring: boolean;
  error: string | null;
}

// Heads grouped by paradigm: classic ML (hand-crafted features -> XGBoost),
// hybrid (foundation-model embeddings -> the same XGBoost), fully foundational
// (the model IS the classifier). Groups with no scores in the response hide.
const PARADIGMS: { id: 'classic' | 'hybrid' | 'foundation'; label: string; cls: string }[] = [
  { id: 'classic', label: 'classic ml', cls: 'text-gray-500' },
  { id: 'hybrid', label: 'foundation-fed xgboost', cls: 'text-accent/70' },
  { id: 'foundation', label: 'foundation model', cls: 'text-status-purple/70' },
];

const HEADS: {
  key: 'raw' | 'embed' | 'combined' | 'nexus';
  paradigm: 'classic' | 'hybrid' | 'foundation';
  label: string;
  fill: string;
  text: string;
}[] = [
  { key: 'raw', paradigm: 'classic', label: 'Raw features', fill: 'bg-gray-500', text: 'text-gray-200' },
  { key: 'embed', paradigm: 'hybrid', label: 'Embeddings', fill: 'bg-accent', text: 'text-accent' },
  { key: 'combined', paradigm: 'hybrid', label: 'Combined', fill: 'bg-status-amber', text: 'text-status-amber' },
  { key: 'nexus', paradigm: 'foundation', label: 'Large Tabular Model', fill: 'bg-status-purple', text: 'text-status-purple' },
];

const THRESHOLD = 0.5;

function meta(key: string, summary: Summary | null, result: ScoreResp): string {
  if (key === 'raw') return `P(fraud) · ${summary?.n_raw_features ?? '—'}-d tabular`;
  if (key === 'embed') return `P(fraud) · ${summary?.pca_dim ?? 64}-d PCA embedding`;
  if (key === 'nexus') {
    const info = result.nexus;
    if (info?.status === 'timeout') return 'remote LTM timed out — no score';
    if (info?.status === 'unavailable') return 'remote LTM unavailable — no score';
    const lat = info?.latency_ms != null ? ` · ${(info.latency_ms / 1000).toFixed(1)} s remote` : '';
    return `P(fraud) · raw table → NEXUS${lat}`;
  }
  return 'P(fraud) · raw + embedding';
}

export default function ModelHeads({ result, summary, scoring, error }: Props) {
  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-300">Model heads</h2>
        <span className="text-[10px] uppercase tracking-wider text-gray-500">
          {result && 'nexus' in result.scores
            ? 'raw · embeddings · combined · ltm'
            : 'raw · embeddings · combined'}
        </span>
      </div>

      {/* token strip */}
      {result && (
        <div className="flex flex-wrap gap-1.5 mb-4 min-h-[28px]">
          {result.tokens.slice(0, 16).map((t, i) => {
            const special = t.startsWith('<');
            return (
              <span
                key={i}
                className={`font-mono text-[11px] px-2 py-1 rounded border animate-fade-slide-in ${
                  special
                    ? 'bg-accent/10 border-accent/30 text-accent'
                    : 'bg-surface-3 border-surface-4 text-gray-400'
                }`}
                style={{ animationDelay: `${i * 45}ms` }}
              >
                {t}
              </span>
            );
          })}
        </div>
      )}

      {error && (
        <div className="px-4 py-3 bg-status-red-dim/30 border border-status-red/40 rounded-lg text-sm text-status-red">
          {error}
        </div>
      )}

      {!result && !error && (
        <div className="text-center py-12 text-gray-500">
          <Cpu className="w-12 h-12 mx-auto mb-3 opacity-20" />
          <p className="text-gray-400">Load an example or compose a transaction, then run inference.</p>
          <p className="text-[11px] text-gray-600 mt-2">
            Each score is a forward pass through the decoder checkpoint when the backend is in REAL mode.
          </p>
        </div>
      )}

      {result && (
        <>
          <div className="space-y-4">
            {PARADIGMS.map(({ id, label: groupLabel, cls }) => {
              const heads = HEADS.filter((h) => h.paradigm === id && h.key in result.scores);
              if (!heads.length) return null;
              return (
                <div key={id} className="space-y-4">
                  <div className="flex items-center gap-2 -mb-1">
                    <span className={`text-[9px] uppercase tracking-widest ${cls}`}>{groupLabel}</span>
                    <div className="flex-1 h-px bg-surface-3" />
                  </div>
                  {heads.map(({ key, label, fill, text }) => {
                    const p = result.scores[key];
                    const pct = typeof p === 'number' ? (p * 100).toFixed(1) : null;
                    return (
                      <div key={key}>
                        <div className="flex items-baseline justify-between mb-1.5">
                          <span className="text-xs text-gray-300">{label}</span>
                          <span className={`font-mono text-base font-medium ${pct == null ? 'text-gray-600' : text}`}>
                            {pct == null ? '—' : `${pct}%`}
                          </span>
                        </div>
                        <div className="h-2 bg-surface-3 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${fill} transition-[width] duration-700 ease-out`}
                            style={{ width: `${pct ?? 0}%` }}
                          />
                        </div>
                        <div className="text-[10px] font-mono text-gray-600 mt-1">{meta(key, summary, result)}</div>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>

          <Verdict result={result} scoring={scoring} />
        </>
      )}
    </div>
  );
}

function Verdict({ result, scoring }: { result: ScoreResp; scoring: boolean }) {
  const p = result.scores.combined;
  const flagged = p >= THRESHOLD;
  const pct = (p * 100).toFixed(1);
  return (
    <div
      className={`mt-5 flex items-center gap-3 px-4 py-3 rounded-lg border ${
        flagged
          ? 'bg-status-red/10 border-status-red/30 animate-pulse-glow'
          : 'bg-status-green/10 border-status-green/30'
      } ${scoring ? 'opacity-60' : ''}`}
    >
      <span
        className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-[11px] font-medium ${
          flagged ? 'bg-status-red/20 text-status-red' : 'bg-status-green/20 text-status-green'
        }`}
      >
        {flagged ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
        {flagged ? 'FLAG · REVIEW' : 'CLEAR'}
      </span>
      <p className="text-xs text-gray-400 leading-relaxed">
        Combined head returns <span className="font-mono text-gray-200">{pct}%</span> fraud probability.{' '}
        {result.mode === 'real'
          ? `Embedding extracted live from the ${result.embedding_dim}-d decoder checkpoint.`
          : 'Synthetic score — run export_for_demo.py on the GPU box for live model output.'}
      </p>
    </div>
  );
}
