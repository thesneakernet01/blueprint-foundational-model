import type { StatusResp } from '../api';

export type View = 'inference' | 'lifecycle';

interface Props {
  status: StatusResp | null;
  error: boolean;
  view: View;
  onViewChange: (v: View) => void;
  onBuild: () => void;
  onData: () => void;
}

const VIEWS: { key: View; label: string }[] = [
  { key: 'inference', label: 'Inference' },
  { key: 'lifecycle', label: 'Model Lifecycle' },
];

const GPU_BACKEND_LABEL: Record<string, string> = { cuda: 'CUDA', rocm: 'ROCm' };

function StatusDot({ tone }: { tone: 'green' | 'amber' | 'neutral' }) {
  const cls =
    tone === 'green'
      ? 'bg-status-green animate-pulse'
      : tone === 'amber'
        ? 'bg-status-amber'
        : 'bg-surface-4';
  return <span className={`w-1.5 h-1.5 rounded-full ${cls}`} />;
}

export default function Header({ status, error, view, onViewChange, onBuild, onData }: Props) {
  const real = status?.mode === 'real';
  const modeLabel = error
    ? 'backend offline'
    : !status
      ? 'connecting…'
      : real
        ? `live · ${status.gpu ? 'GPU' : 'CPU'}`
        : 'demo fallback';
  const modeTone: 'green' | 'amber' | 'neutral' = error
    ? 'neutral'
    : real
      ? 'green'
      : 'amber';

  return (
    <header className="bg-surface-1 border-b border-surface-3 px-6 flex items-stretch justify-between gap-6">
      <div className="flex items-center gap-3 py-3.5">
        <span className="w-[3px] self-stretch rounded-full bg-accent" aria-hidden />
        <div>
          <h1 className="text-[15px] font-semibold text-brand-indigo leading-tight tracking-[-0.01em]">
            Transaction Foundation Model
          </h1>
          <p className="text-[11px] text-gray-500">Live fraud inference on Cloudera AI</p>
        </div>
      </div>

      <nav className="flex items-stretch gap-7">
        {VIEWS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => onViewChange(key)}
            className={`relative text-[13px] font-medium transition-colors ${
              view === key ? 'text-brand-indigo' : 'text-gray-500 hover:text-brand-indigo'
            }`}
          >
            {label}
            {view === key && (
              <span className="absolute -bottom-px inset-x-0 h-[2px] bg-accent" aria-hidden />
            )}
          </button>
        ))}
      </nav>

      <div className="flex items-center gap-5 py-3.5">
        <div className="flex items-center gap-4 text-[11px] text-gray-500">
          <span className="flex items-center gap-1.5">
            <StatusDot tone={modeTone} />
            {modeLabel}
          </span>
          <span className="flex items-center gap-1.5">
            <StatusDot tone={status?.gpu ? 'green' : 'neutral'} />
            {status?.gpu ? (GPU_BACKEND_LABEL[status.gpu_backend] ?? status.gpu_backend) : 'no GPU'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onData}
            className="border border-surface-4 text-gray-700 hover:border-accent/60 hover:text-accent px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
          >
            Data
          </button>
          <button
            onClick={onBuild}
            className="bg-accent text-white hover:bg-accent/90 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
          >
            Build artifacts
          </button>
        </div>
      </div>
    </header>
  );
}
