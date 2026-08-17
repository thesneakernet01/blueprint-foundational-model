import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RunRecord } from '../../api';
import { CHART, SERIES } from './theme';

interface Props {
  runs: RunRecord[];
}

type Metric = 'test_auc' | 'test_ap';
const METRIC_LABEL: Record<Metric, string> = { test_auc: 'ROC-AUC', test_ap: 'Avg precision' };

export default function MetricTrend({ runs }: Props) {
  const [metric, setMetric] = useState<Metric>('test_ap');

  const heads = useMemo(() => {
    const keys = new Set<string>();
    runs.forEach((r) => r.models.forEach((m) => SERIES[m.key] && keys.add(m.key)));
    return ['raw', 'embed', 'combined', 'nexus'].filter((k) => keys.has(k));
  }, [runs]);

  const data = useMemo(
    () =>
      runs.map((r) => {
        const row: Record<string, number | null> = { run: r.run };
        r.models.forEach((m) => {
          if (SERIES[m.key]) row[m.key] = m[metric];
        });
        return row;
      }),
    [runs, metric],
  );

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700">Model quality per training run</h3>
        <div className="flex items-center gap-1 bg-surface-3 rounded-md p-0.5">
          {(Object.keys(METRIC_LABEL) as Metric[]).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                metric === m ? 'bg-accent/20 text-accent' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {METRIC_LABEL[m]}
            </button>
          ))}
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="h-[260px] flex items-center justify-center text-center">
          <div>
            <p className="text-gray-600 text-sm">No training runs recorded yet</p>
            <p className="text-[11px] text-gray-400 mt-1.5">
              Each run of Build artifacts is granted a larger training budget — this chart
              tracks the improvement, with markers for registered model versions.
            </p>
          </div>
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="run"
                tick={CHART.tick}
                tickLine={false}
                axisLine={{ stroke: CHART.grid }}
                tickFormatter={(v) => `#${v}`}
              />
              <YAxis
                tick={CHART.tick}
                tickLine={false}
                axisLine={false}
                domain={['dataMin - 0.01', 'dataMax + 0.005']}
                tickFormatter={(v: number) => v.toFixed(3)}
                width={48}
              />
              <Tooltip
                contentStyle={CHART.tooltip}
                labelFormatter={(v) => `Run #${v}`}
                formatter={(value: number, name: string) => [
                  value?.toFixed(4),
                  SERIES[name]?.label ?? name,
                ]}
              />
              {heads.map((k) => (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={k}
                  stroke={SERIES[k].color}
                  strokeWidth={k === 'combined' ? 2 : 1.5}
                  dot={{ r: 3, fill: SERIES[k].color, strokeWidth: 0 }}
                  connectNulls
                />
              ))}
              {/* Registered-version markers on the combined series. */}
              {runs
                .filter((r) => r.registry.registered)
                .map((r) => {
                  const y = r.models.find((m) => m.key === 'combined')?.[metric];
                  if (y == null) return null;
                  return (
                    <ReferenceDot
                      key={r.run_id}
                      x={r.run}
                      y={y}
                      r={7}
                      fill={r.registry.deployed ? '#10b981' : 'transparent'}
                      fillOpacity={r.registry.deployed ? 0.35 : 0}
                      stroke="#10b981"
                      strokeWidth={1.5}
                      label={{
                        value: r.registry.model_version != null ? `v${r.registry.model_version}` : 'reg',
                        position: 'top',
                        fill: '#10b981',
                        fontSize: 10,
                      }}
                    />
                  );
                })}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-1 flex-wrap">
            {heads.map((k) => (
              <span key={k} className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className="w-2.5 h-0.5 rounded" style={{ background: SERIES[k].color }} />
                {SERIES[k].label}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-[10px] text-gray-500">
              <span className="w-2 h-2 rounded-full border border-status-green" />
              registered version (filled = deployed)
            </span>
          </div>
        </>
      )}
    </div>
  );
}
