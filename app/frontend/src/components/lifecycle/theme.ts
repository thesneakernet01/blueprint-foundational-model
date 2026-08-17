// Shared chart chrome for the Model Lifecycle dashboard (Cloudera brand
// palette — keep in sync with tailwind.config.js). Recharts needs raw hexes.
export const SERIES: Record<string, { color: string; label: string }> = {
  raw: { color: '#6E7679', label: 'Raw tabular' },
  embed: { color: '#FF550C', label: 'FM embeddings' },
  combined: { color: '#5555F9', label: 'Combined' },
  nexus: { color: '#26177B', label: 'NEXUS LTM' },
};

export const CHART = {
  tick: { fill: '#6b7280', fontSize: 10 },
  grid: '#D9D9E8',
  tooltip: {
    background: '#FFFFFF',
    border: '1px solid #D9D9E8',
    borderRadius: '8px',
    fontSize: '12px',
  } as React.CSSProperties,
};

export const fmtDuration = (sec: number | null | undefined) => {
  if (sec == null) return '—';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m ${s.toString().padStart(2, '0')}s` : `${s}s`;
};

export const fmtRows = (n: number) => (n >= 1000 ? `${n / 1000}k` : String(n));
