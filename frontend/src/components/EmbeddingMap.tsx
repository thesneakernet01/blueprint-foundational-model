import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, ResponsiveContainer } from 'recharts';
import { Map as MapIcon } from 'lucide-react';
import type { ScoreResp, UmapPoint } from '../api';

interface Props {
  umap: UmapPoint[];
  result: ScoreResp | null;
}

// Recharts gets sluggish past a couple thousand markers; keep all (rare) fraud
// points and stride-sample the normal cloud down to a budget.
const MAX_BG = 1500;

// Glowing accent marker for the live transaction (Recharts won't size a point
// from a Cell prop, so we draw it ourselves).
function LiveDot(props: { cx?: number; cy?: number }) {
  const { cx, cy } = props;
  if (cx == null || cy == null) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={11} fill="#6366f1" fillOpacity={0.15} />
      <circle cx={cx} cy={cy} r={5} fill="#6366f1" stroke="#0a0a0f" strokeWidth={1.5} />
    </g>
  );
}

export default function EmbeddingMap({ umap, result }: Props) {
  const { normal, fraud, dropped } = useMemo(() => {
    const fraudPts = umap.filter((p) => p.fraud);
    const normalPts = umap.filter((p) => !p.fraud);
    const budget = Math.max(0, MAX_BG - fraudPts.length);
    const stride = normalPts.length > budget ? Math.ceil(normalPts.length / budget) : 1;
    const sampled = stride > 1 ? normalPts.filter((_, i) => i % stride === 0) : normalPts;
    return { normal: sampled, fraud: fraudPts, dropped: normalPts.length - sampled.length };
  }, [umap]);

  const live = result?.position ? [result.position] : [];
  const hasData = umap.length > 0 || live.length > 0;

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-300">Embedding map</h2>
        <span className="text-[10px] uppercase tracking-wider text-gray-500">UMAP · test set</span>
      </div>

      <div className="h-[300px] rounded-lg border border-surface-3 bg-surface-0/40">
        {!hasData ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <MapIcon className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-gray-400 text-sm">No embedding background yet</p>
            <p className="text-[11px] text-gray-600 mt-2">
              Available after export_for_demo.py runs on the GPU box.
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
              <XAxis type="number" dataKey="x" hide domain={['dataMin', 'dataMax']} />
              <YAxis type="number" dataKey="y" hide domain={['dataMin', 'dataMax']} />
              <ZAxis range={[16, 16]} />
              <Scatter data={normal} fill="#3a6ea5" fillOpacity={0.45} isAnimationActive={false} />
              <Scatter data={fraud} fill="#ef4444" fillOpacity={0.75} isAnimationActive={false} />
              {live.length > 0 && (
                <Scatter data={live} isAnimationActive={false} shape={<LiveDot />} />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="flex items-center gap-4 mt-3 text-[11px] font-mono text-gray-500">
        <span className="flex items-center gap-1.5">
          <i className="w-2 h-2 rounded-full inline-block" style={{ background: '#3a6ea5' }} /> Normal
        </span>
        <span className="flex items-center gap-1.5">
          <i className="w-2 h-2 rounded-full inline-block" style={{ background: '#ef4444' }} /> Fraud
        </span>
        <span className="flex items-center gap-1.5">
          <i
            className="w-2 h-2 rounded-full inline-block"
            style={{ background: '#6366f1', boxShadow: '0 0 8px #6366f1' }}
          />{' '}
          This transaction
        </span>
      </div>

      <p className="text-[10px] text-gray-600 mt-3 leading-relaxed">
        512-d embeddings projected to 2-D. The decoder learns the geometry from raw sequences — fraud
        clusters emerge with no labels.
        {dropped > 0 && ` ${dropped.toLocaleString()} background points downsampled for rendering.`}
      </p>
    </div>
  );
}
