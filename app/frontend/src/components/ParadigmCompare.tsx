import { X } from 'lucide-react';
import type { ModelSummary, Summary } from '../api';

interface Props {
  open: boolean;
  onClose: () => void;
  summary: Summary | null;
}

const fmt = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(4));
const liftStr = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;

/** Side-by-side "everything XGBoost vs everything foundational" table, with the
 *  hybrid heads shown as the bridge between the two worlds. Qualitative rows
 *  are static; metric rows read the live export summary. */
export default function ParadigmCompare({ open, onClose, summary }: Props) {
  if (!open) return null;

  const byKey: Record<string, ModelSummary> = Object.fromEntries(
    (summary?.models ?? []).map((m) => [m.key, m]),
  );
  const nexus = byKey.nexus ?? null;
  const lift = summary?.lift;

  const hybridPair = (a: string | null, b: string | null) =>
    a == null && b == null ? '—' : `${a ?? '—'} · ${b ?? '—'}`;

  const rows: { label: string; classic: string; hybrid: string; foundation: string }[] = [
    {
      label: 'Input',
      classic: '13 hand-picked columns',
      hybrid: '512-d learned embedding (± raw)',
      foundation: 'raw transaction, untouched',
    },
    {
      label: 'Feature engineering',
      classic: 'humans — domain experts, revisited every fraud shift',
      hybrid: 'none — representations learned by the transaction FM',
      foundation: 'none — pre-trained on tabular corpora',
    },
    {
      label: 'Knowledge source',
      classic: "this institution's labeled history",
      hybrid: 'FM pre-training + your labels',
      foundation: 'billions of prior tabular prediction tasks',
    },
    {
      label: 'Classifier',
      classic: 'XGBoost',
      hybrid: 'the same XGBoost — only the inputs changed',
      foundation: 'the foundation model itself',
    },
    {
      label: 'When fraud mutates',
      classic: 're-engineer features, relabel, retrain',
      hybrid: 'refresh embeddings; heads retrain in minutes',
      foundation: 'pre-trained priors; re-adapt on new data',
    },
    {
      label: 'Governance',
      classic: 'the devil they know',
      hybrid: 'same approved classifier, better inputs',
      foundation: 'frontier — new model-risk territory',
    },
    {
      label: 'Test ROC-AUC',
      classic: fmt(byKey.raw?.test_auc),
      hybrid: hybridPair(fmt(byKey.embed?.test_auc), fmt(byKey.combined?.test_auc)),
      foundation: nexus ? fmt(nexus.test_auc) : 'not enabled',
    },
    {
      label: 'Test avg precision',
      classic: fmt(byKey.raw?.test_ap),
      hybrid: hybridPair(fmt(byKey.embed?.test_ap), fmt(byKey.combined?.test_ap)),
      foundation: nexus ? fmt(nexus.test_ap) : 'not enabled',
    },
    {
      label: 'AP lift vs baseline',
      classic: 'baseline',
      hybrid: hybridPair(liftStr(lift?.embed_ap_pct), liftStr(lift?.combined_ap_pct)),
      foundation: nexus ? liftStr(lift?.nexus_ap_pct) : 'not enabled',
    },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-surface-1 rounded-xl border border-surface-3 shadow-2xl w-[780px] max-w-[95vw] max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-surface-3">
          <h2 className="text-base font-semibold text-white">Classic ML vs. foundation models</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-500 hover:text-gray-300 rounded-lg hover:bg-surface-3"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 overflow-y-auto">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="w-36" />
                  <th className="text-left align-top px-3 pb-3 border-b-2 border-gray-500/60">
                    <div className="text-gray-200 font-medium">Everything XGBoost</div>
                    <div className="text-[10px] font-normal text-gray-500 mt-0.5">raw head · today's world</div>
                  </th>
                  <th className="text-left align-top px-3 pb-3 border-b-2 border-accent/60">
                    <div className="text-accent font-medium">The bridge — hybrid</div>
                    <div className="text-[10px] font-normal text-gray-500 mt-0.5">embeddings · combined heads</div>
                  </th>
                  <th className="text-left align-top px-3 pb-3 border-b-2 border-status-purple/60">
                    <div className="text-status-purple font-medium">Everything foundational</div>
                    <div className="text-[10px] font-normal text-gray-500 mt-0.5">
                      NEXUS large tabular model
                      {nexus?.stub ? ' · stub' : ''}
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ label, classic, hybrid, foundation }) => (
                  <tr key={label} className="border-b border-surface-3 last:border-0">
                    <td className="py-2.5 pr-2 text-gray-500 align-top">{label}</td>
                    <td className="py-2.5 px-3 text-gray-300 align-top">{classic}</td>
                    <td className="py-2.5 px-3 text-gray-300 align-top">{hybrid}</td>
                    <td className={`py-2.5 px-3 align-top ${nexus ? 'text-gray-300' : 'text-gray-600'}`}>
                      {foundation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-[11px] text-gray-500 leading-relaxed">
            The adoption path runs left to right: keep the classifier your model-risk team
            already governs, hand it foundation-model representations for the lift, and use
            the fully-foundational column to evaluate what comes next
            {!nexus && ' (enable it with NEXUS_MODE=stub|live and re-run the export)'}
            {nexus?.stub && ' (NEXUS metrics are stub placeholders pending live access)'}.
          </p>
        </div>
      </div>
    </div>
  );
}
