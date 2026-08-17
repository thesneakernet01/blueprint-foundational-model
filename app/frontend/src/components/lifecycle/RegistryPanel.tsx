import { useEffect, useRef } from 'react';
import { BookMarked, CloudUpload, ExternalLink, Loader2, Lock } from 'lucide-react';
import { postDeploy, postRegister, type RegistryStatus } from '../../api';

interface Props {
  registry: RegistryStatus | null;
  /** Re-fetch /api/registry (called after starting an action + while polling). */
  onRefresh: () => void;
  exportRunning: boolean;
}

export default function RegistryPanel({ registry, onRefresh, exportRunning }: Props) {
  const jobRunning = registry?.job.state === 'running';
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [registry?.job.log.length]);

  const start = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
    } catch {
      // The reason lands in the next status poll; the buttons are gated anyway.
    }
    onRefresh();
  };

  if (!registry) {
    return <div className="bg-surface-3 rounded-lg border border-surface-3 p-4 h-40 animate-pulse" />;
  }

  const unavailable = !registry.available;
  const canRegister =
    registry.available && registry.artifacts_ready && !jobRunning && !exportRunning;
  const canDeploy = registry.available && registry.versions.length > 0 && !jobRunning;
  const latestVersion = registry.versions[0]?.version;

  return (
    <div
      className={`bg-surface-2 rounded-lg border border-surface-3 p-4 ${unavailable ? 'opacity-80' : ''}`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-2">
          <BookMarked className="w-4 h-4 text-accent" />
          Cloudera Model Registry
        </h3>
        <span className="text-[10px] font-mono text-gray-500">{registry.model_name}</span>
      </div>

      {unavailable && (
        <div className="flex items-start gap-2 bg-surface-3/60 border border-surface-4 rounded-md px-3 py-2 mb-3">
          <Lock className="w-3.5 h-3.5 text-gray-500 mt-0.5 shrink-0" />
          <p className="text-[11px] text-gray-600 leading-snug">
            Available when running on Cloudera AI — {registry.reason}.
          </p>
        </div>
      )}

      <div className="flex items-center gap-2 mb-3">
        <button
          onClick={() => start(postRegister)}
          disabled={!canRegister}
          title={
            !registry.available
              ? (registry.reason ?? undefined)
              : !registry.artifacts_ready
                ? (registry.artifacts_reason ?? undefined)
                : exportRunning
                  ? 'wait for the export to finish'
                  : undefined
          }
          className="flex items-center gap-1.5 bg-accent text-white hover:bg-accent/90 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {jobRunning && registry.job.action === 'register' ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <BookMarked className="w-3.5 h-3.5" />
          )}
          Register latest run
        </button>
        <button
          onClick={() => start(postDeploy)}
          disabled={!canDeploy}
          title={
            !registry.available
              ? (registry.reason ?? undefined)
              : registry.versions.length === 0
                ? 'register a version first'
                : undefined
          }
          className="flex items-center gap-1.5 bg-surface-3 text-gray-700 hover:bg-surface-4 px-3 py-1.5 text-xs font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {jobRunning && registry.job.action === 'deploy' ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <CloudUpload className="w-3.5 h-3.5" />
          )}
          Deploy {latestVersion != null ? `v${latestVersion}` : ''}
        </button>
        {registry.deployment?.status && (
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded ${
              registry.deployment.status === 'deployed'
                ? 'bg-status-green/10 text-status-green'
                : 'bg-status-amber/10 text-status-amber'
            }`}
          >
            endpoint {registry.deployment.status}
          </span>
        )}
        {registry.deployment?.url && (
          <a
            href={registry.deployment.url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-[11px] text-accent hover:underline"
          >
            endpoint <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>

      {registry.job.error && registry.job.state === 'error' && (
        <p className="text-[11px] text-status-red mb-2">{registry.job.error}</p>
      )}

      {(jobRunning || registry.job.log.length > 0) && (
        <div
          ref={logRef}
          className="bg-surface-1 border border-surface-4 rounded-md p-2 max-h-28 overflow-y-auto font-mono text-[10px] text-gray-600 space-y-0.5 mb-3"
        >
          {registry.job.log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}

      <div>
        <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
          Registered versions
        </div>
        {registry.versions.length === 0 ? (
          <p className="text-[11px] text-gray-400">
            {unavailable
              ? 'Versions appear here once the app runs on Cloudera AI.'
              : 'None yet — register the latest run to create v1.'}
          </p>
        ) : (
          <div className="space-y-1">
            {registry.versions.map((v, i) => (
              <div
                key={`${v.version}-${i}`}
                className="flex items-center justify-between text-[11px] bg-surface-3/50 rounded px-2 py-1"
              >
                <span className="font-mono text-gray-800">v{v.version ?? '?'}</span>
                <span className="text-gray-500 font-mono">{v.created_at}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
