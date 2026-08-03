import { Activity, Cpu, Database, Hammer, ShieldAlert } from 'lucide-react';
import type { StatusResp } from '../api';

interface Props {
  status: StatusResp | null;
  error: boolean;
  onBuild: () => void;
  onData: () => void;
}

function StatusDot({ tone }: { tone: 'green' | 'amber' | 'neutral' }) {
  const cls =
    tone === 'green'
      ? 'bg-status-green animate-pulse'
      : tone === 'amber'
        ? 'bg-status-amber'
        : 'bg-surface-4';
  return <span className={`w-2 h-2 rounded-full ${cls}`} />;
}

export default function Header({ status, error, onBuild, onData }: Props) {
  const real = status?.mode === 'real';
  const modeLabel = error
    ? 'backend offline'
    : !status
      ? 'connecting…'
      : real
        ? `REAL · ${status.gpu ? 'GPU' : 'CPU'}`
        : 'DEMO-FALLBACK';
  const modeTone: 'green' | 'amber' | 'neutral' = error
    ? 'neutral'
    : real
      ? 'green'
      : 'amber';

  return (
    <header className="bg-surface-1 border-b border-surface-3 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-accent/10 rounded-lg">
          <ShieldAlert className="w-5 h-5 text-accent" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-white leading-tight">
            Transaction Foundation Model
          </h1>
          <p className="text-xs text-gray-500">
            Live Fraud Inference · NeMo AutoModel + RAPIDS
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xs">
          <StatusDot tone={modeTone} />
          <Activity className="w-3.5 h-3.5 text-gray-400" />
          <span className="text-gray-400">{modeLabel}</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <StatusDot tone={status?.gpu ? 'green' : 'neutral'} />
          <Cpu className="w-3.5 h-3.5 text-gray-400" />
          <span className="text-gray-400">{status?.gpu ? 'CUDA ready' : 'no GPU'}</span>
        </div>
        <button
          onClick={onData}
          className="flex items-center gap-1.5 bg-surface-3 text-gray-300 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
        >
          <Database className="w-3.5 h-3.5" />
          Data
        </button>
        <button
          onClick={onBuild}
          className="flex items-center gap-1.5 bg-surface-3 text-gray-300 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
        >
          <Hammer className="w-3.5 h-3.5" />
          Build artifacts
        </button>
      </div>
    </header>
  );
}
