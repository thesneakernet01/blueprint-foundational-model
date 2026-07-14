import { Loader2, Play } from 'lucide-react';
import type { Example, TxnPayload } from '../api';

export interface FormState {
  amount: string;
  merchant: string;
  city: string;
  state: string;
  channel: string;
  mcc: string;
  zip: string;
  time: string;
  card: string;
}

export const DEFAULT_FORM: FormState = {
  amount: '$842.50',
  merchant: 'DIGITAL-GOODS-LLC',
  city: 'ONLINE',
  state: 'ONLINE',
  channel: 'Online Transaction',
  mcc: '5942',
  zip: '00000',
  time: '03:14',
  card: '0',
};

export function formToPayload(f: FormState): TxnPayload {
  return {
    Amount: f.amount,
    'Merchant Name': f.merchant,
    'Merchant City': f.city,
    'Merchant State': f.state,
    'Use Chip': f.channel,
    MCC: parseInt(f.mcc, 10) || 0,
    Zip: f.zip,
    Time: f.time,
    Year: 2019,
    Month: 11,
    Day: 22,
    Card: parseInt(f.card, 10) || 0,
    User: 0,
  };
}

export function exampleToForm(t: Record<string, string | number>): FormState {
  const s = (k: string, d = '') => (t[k] != null ? String(t[k]) : d);
  return {
    amount: s('Amount'),
    merchant: s('Merchant Name'),
    city: s('Merchant City'),
    state: s('Merchant State'),
    channel: s('Use Chip', 'Online Transaction'),
    mcc: s('MCC'),
    zip: s('Zip'),
    time: s('Time'),
    card: s('Card', '0'),
  };
}

const CHANNELS = ['Online Transaction', 'Swipe Transaction', 'Chip Transaction'];

const labelCls = 'block text-[10px] text-gray-500 uppercase tracking-wider mb-1';
const inputCls =
  'w-full px-2.5 py-1.5 text-xs bg-surface-3 border border-surface-4 rounded-md text-gray-200 ' +
  'placeholder-gray-600 font-mono focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/20';

interface Props {
  form: FormState;
  setForm: (f: FormState) => void;
  examples: Example[];
  /** Example-card click — lets the app track which example the form holds
   *  (cleared again on any manual field edit via setForm). */
  onLoadExample: (ex: Example) => void;
  onRun: () => void;
  scoring: boolean;
}

export default function TransactionComposer({ form, setForm, examples, onLoadExample, onRun, scoring }: Props) {
  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });

  return (
    <div className="bg-surface-2 rounded-lg border border-surface-3 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-300">Transaction input</h2>
        <span className="text-[10px] uppercase tracking-wider text-gray-500">compose</span>
      </div>

      {examples.length > 0 && (
        <div className="flex flex-col gap-2 mb-4">
          {examples.map((ex, i) => (
            <button
              key={i}
              onClick={() => onLoadExample(ex)}
              className="text-left bg-surface-3 border border-surface-4 rounded-lg px-3 py-2 transition-colors hover:border-accent/50 hover:bg-surface-3"
            >
              <div className="text-xs text-gray-200">{ex.label}</div>
              <div className="text-[10px] font-mono text-gray-500 mt-0.5">
                {String(ex.txn['Amount'] ?? '')} ·{' '}
                {String(ex.txn['Use Chip'] ?? '').split(' ')[0].toUpperCase()}
                {ex.is_fraud != null && ` · ground truth: ${ex.is_fraud ? 'FRAUD' : 'LEGIT'}`}
              </div>
            </button>
          ))}
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className={labelCls}>Amount</label>
          <input className={inputCls} value={form.amount} onChange={set('amount')} />
        </div>
        <div>
          <label className={labelCls}>Merchant</label>
          <input className={inputCls} value={form.merchant} onChange={set('merchant')} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>City</label>
            <input className={inputCls} value={form.city} onChange={set('city')} />
          </div>
          <div>
            <label className={labelCls}>State</label>
            <input className={inputCls} value={form.state} onChange={set('state')} />
          </div>
        </div>
        <div>
          <label className={labelCls}>Channel</label>
          <select className={inputCls} value={form.channel} onChange={set('channel')}>
            {CHANNELS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>MCC</label>
            <input className={inputCls} value={form.mcc} onChange={set('mcc')} />
          </div>
          <div>
            <label className={labelCls}>ZIP</label>
            <input className={inputCls} value={form.zip} onChange={set('zip')} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Time (HH:MM)</label>
            <input className={inputCls} value={form.time} onChange={set('time')} />
          </div>
          <div>
            <label className={labelCls}>Card</label>
            <input className={inputCls} value={form.card} onChange={set('card')} />
          </div>
        </div>
      </div>

      <button
        onClick={onRun}
        disabled={scoring}
        className="w-full mt-4 flex items-center justify-center gap-2 bg-accent text-white hover:bg-accent/90 px-3 py-2.5 text-sm font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-wait"
      >
        {scoring ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" /> Scoring…
          </>
        ) : (
          <>
            <Play className="w-4 h-4" /> Run inference
          </>
        )}
      </button>
    </div>
  );
}
