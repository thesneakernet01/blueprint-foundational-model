import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Area,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RunRecord } from '../../api';
import { CHART, SERIES } from './theme';

interface Props {
  lastRun: RunRecord | null;
}

function Empty({ title, note }: { title: string; note: string }) {
  return (
    <div className="panel p-4">
      <h3 className="slide-title mb-2">{title}</h3>
      <p className="text-[11px] text-gray-400 py-8 text-center">{note}</p>
    </div>
  );
}

const emptyNote = (lastRun: RunRecord | null) =>
  lastRun
    ? 'Recorded from the next training run onward.'
    : 'Recorded with each training run — run Build artifacts to capture run #1.';

/** The gradient-boosting "loss curve": per-round validation AUC per head,
 *  with the round where early stopping kept each model. The cost-governance
 *  argument made visible — training stops when more rounds stop helping. */
export function EvalCurvePanel({ lastRun }: Props) {
  const curves = lastRun?.diagnostics?.eval_curves;
  if (!curves || Object.keys(curves).length === 0) {
    return <Empty title="Validation curve" note={emptyNote(lastRun)} />;
  }
  const heads = ['raw', 'embed', 'combined'].filter((k) => curves[k]);
  const series = heads.map((k) => ({
    key: k,
    data: curves[k].rounds.map((r, i) => ({ round: r, auc: curves[k].auc[i] })),
  }));
  const best = curves.combined?.best_round;

  return (
    <div className="panel p-4">
      <h3 className="slide-title mb-1">Validation curve · run #{lastRun?.run}</h3>
      <p className="text-[11px] text-gray-500 leading-snug mb-2">
        Validation AUC per boosting round. Early stopping halts each head the moment
        extra rounds stop improving it — spend caps itself.
      </p>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart margin={{ top: 6, right: 10, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="round"
            type="number"
            domain={['dataMin', 'dataMax']}
            tick={CHART.tick}
            tickLine={false}
            axisLine={{ stroke: CHART.grid }}
          />
          <YAxis
            tick={CHART.tick}
            tickLine={false}
            axisLine={false}
            domain={['dataMin - 0.004', 'dataMax + 0.002']}
            tickFormatter={(v: number) => v.toFixed(3)}
            width={44}
          />
          <Tooltip
            contentStyle={CHART.tooltip}
            labelFormatter={(v) => `round ${v}`}
            formatter={(value: number, name: string) => [
              value?.toFixed(4),
              SERIES[name]?.label ?? name,
            ]}
          />
          {best != null && (
            <ReferenceLine
              x={best}
              stroke="#10b981"
              strokeDasharray="4 3"
              label={{ value: `stop @ ${best}`, position: 'top', fill: '#10b981', fontSize: 9 }}
            />
          )}
          {series.map((s) => (
            <Line
              key={s.key}
              data={s.data}
              dataKey="auc"
              name={s.key}
              type="monotone"
              stroke={SERIES[s.key].color}
              strokeWidth={s.key === 'combined' ? 2 : 1.5}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-1">
        {heads.map((k) => (
          <span key={k} className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <span className="w-2.5 h-0.5 rounded" style={{ background: SERIES[k].color }} />
            {SERIES[k].label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Fraud vs legitimate score histograms (each normalized to its own class),
 *  for the combined head on the held-out test slice. */
export function SeparationPanel({ lastRun }: Props) {
  const sep = lastRun?.diagnostics?.separation;
  const data = useMemo(() => {
    if (!sep) return [];
    const nL = sep.legit.reduce((a, b) => a + b, 0) || 1;
    const nF = sep.fraud.reduce((a, b) => a + b, 0) || 1;
    return sep.legit.map((v, i) => ({
      p: (sep.edges[i] + sep.edges[i + 1]) / 2,
      legit: (v / nL) * 100,
      fraud: (sep.fraud[i] / nF) * 100,
    }));
  }, [sep]);
  if (!sep) return <Empty title="Score separation" note={emptyNote(lastRun)} />;

  return (
    <div className="panel p-4">
      <h3 className="slide-title mb-1">Score separation · run #{lastRun?.run}</h3>
      <p className="text-[11px] text-gray-500 leading-snug mb-2">
        Share of each class by predicted fraud probability, held-out test set. The
        further apart the humps, the more actionable the score.
      </p>
      <ResponsiveContainer width="100%" height={170}>
        <ComposedChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="p"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={CHART.tick}
            tickLine={false}
            axisLine={{ stroke: CHART.grid }}
          />
          <YAxis
            tick={CHART.tick}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${Math.round(v)}%`}
            width={36}
          />
          <Tooltip
            contentStyle={CHART.tooltip}
            labelFormatter={(v: number) => `score ≈ ${v.toFixed(2)}`}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}% of class`,
              name === 'legit' ? 'legitimate' : 'fraud',
            ]}
          />
          <Area type="step" dataKey="legit" stroke="#5555F9" fill="#5555F9" fillOpacity={0.3} strokeWidth={1.5} />
          <Area type="step" dataKey="fraud" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} strokeWidth={1.5} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-1">
        <span className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: '#5555F9', opacity: 0.5 }} />
          legitimate
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: '#ef4444', opacity: 0.5 }} />
          fraud
        </span>
      </div>
    </div>
  );
}

/** Top feature importances of the combined head — embedding components vs
 *  hand-picked raw columns, the FM thesis from the explainability angle. */
export function ImportancePanel({ lastRun }: Props) {
  const imp = lastRun?.diagnostics?.importance;
  if (!imp || imp.length === 0) {
    return <Empty title="What drives the score" note={emptyNote(lastRun)} />;
  }
  const data = imp.slice(0, 12);

  return (
    <div className="panel p-4">
      <h3 className="slide-title mb-1">What drives the score · run #{lastRun?.run}</h3>
      <p className="text-[11px] text-gray-500 leading-snug mb-2">
        Top features of the combined head. Embedding components crowding out
        hand-picked columns is the foundation-model story, seen from explainability.
      </p>
      <ResponsiveContainer width="100%" height={12 + data.length * 15.5}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 0 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="name"
            width={104}
            interval={0}
            tick={{ fill: '#6b7280', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={CHART.tooltip}
            formatter={(value: number, _n, item) => [
              value.toFixed(4),
              item?.payload?.kind === 'embedding' ? 'embedding component' : 'raw column',
            ]}
          />
          <Bar dataKey="importance" radius={[0, 3, 3, 0]} barSize={9}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.kind === 'embedding' ? '#FF550C' : '#5555F9'} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-1">
        <span className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: '#FF550C', opacity: 0.8 }} />
          embedding component
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: '#5555F9', opacity: 0.8 }} />
          raw column
        </span>
      </div>
    </div>
  );
}
