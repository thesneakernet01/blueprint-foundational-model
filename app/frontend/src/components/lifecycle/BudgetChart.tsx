import { useMemo } from 'react';
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RunBudget, RunRecord } from '../../api';
import { CHART, fmtDuration, fmtRows } from './theme';

interface Props {
  runs: RunRecord[];
  schedule: RunBudget[];
}

/** The "why it improves" context: every run is granted more data and more
 *  boosting rounds (tfm_demo/runs.py schedule), plus what each run cost. */
export default function BudgetChart({ runs, schedule }: Props) {
  const data = useMemo(
    () =>
      runs.map((r) => ({
        run: r.run,
        rows: r.budget.embed_max,
        rounds: r.budget.n_estimators?.combined ?? null,
        duration: r.duration_sec,
      })),
    [runs],
  );

  if (runs.length === 0) {
    return (
      <div className="panel p-4">
        <h3 className="slide-title mb-2">Training budget schedule</h3>
        <p className="text-[11px] text-gray-400 mb-3">
          Each run earns a bigger budget — rows embedded per split and boosting rounds:
        </p>
        <div className="space-y-1.5">
          {schedule.map((b, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px]">
              <span className="w-10 text-gray-500 font-mono">#{i + 1}</span>
              <div className="flex-1 h-2 bg-surface-3 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent/60 rounded-full"
                  style={{ width: `${(b.embed_max / schedule[schedule.length - 1].embed_max) * 100}%` }}
                />
              </div>
              <span className="w-24 text-right font-mono text-gray-600">
                {fmtRows(b.embed_max)} rows · {Math.round(b.xgb_scale * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="panel p-4 space-y-3">
      <div>
        <h3 className="slide-title mb-1">Training budget per run</h3>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="run"
              tick={CHART.tick}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              tickFormatter={(v) => `#${v}`}
            />
            <YAxis
              yAxisId="rows"
              tick={CHART.tick}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => fmtRows(v)}
              width={40}
            />
            <YAxis yAxisId="rounds" orientation="right" hide />
            <Tooltip
              contentStyle={CHART.tooltip}
              labelFormatter={(v) => `Run #${v}`}
              formatter={(value: number, name: string) =>
                name === 'rows'
                  ? [`${value.toLocaleString()} rows/split`, 'embedded']
                  : [value, 'boosting rounds (combined)']
              }
            />
            <Bar yAxisId="rows" dataKey="rows" fill="#5555F9" fillOpacity={0.55} radius={[4, 4, 0, 0]} />
            <Line
              yAxisId="rounds"
              type="monotone"
              dataKey="rounds"
              stroke="#FF550C"
              strokeWidth={1.5}
              dot={{ r: 2.5, fill: '#FF550C', strokeWidth: 0 }}
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-[11px] text-gray-500 mb-1">Export duration</h3>
        <ResponsiveContainer width="100%" height={70}>
          <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <XAxis
              dataKey="run"
              tick={CHART.tick}
              tickLine={false}
              axisLine={{ stroke: CHART.grid }}
              tickFormatter={(v) => `#${v}`}
            />
            <YAxis hide />
            <Tooltip
              contentStyle={CHART.tooltip}
              labelFormatter={(v) => `Run #${v}`}
              formatter={(value: number) => [fmtDuration(value), 'duration']}
            />
            <Bar dataKey="duration" fill="#8789FB" fillOpacity={0.5} radius={[3, 3, 0, 0]} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
