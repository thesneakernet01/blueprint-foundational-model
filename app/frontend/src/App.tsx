import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import Header, { type View } from './components/Header';
import MetricsStrip from './components/MetricsStrip';
import TransactionComposer, {
  DEFAULT_FORM,
  exampleToForm,
  formToPayload,
  type FormState,
} from './components/TransactionComposer';
import ModelHeads from './components/ModelHeads';
import ExportDialog from './components/ExportDialog';
import DataDialog from './components/DataDialog';
import clouderaLogo from './assets/partners/cloudera-white.png';
import vastLogo from './assets/partners/vast-white.svg';
import fundamentalLogo from './assets/partners/fundamental-white.png';

// Recharts is heavy and only the embedding map needs it — load it in its own
// chunk so the first paint (header / composer / heads) isn't blocked on it.
const EmbeddingMap = lazy(() => import('./components/EmbeddingMap'));
// Same deal for the Model Lifecycle dashboard (recharts trend charts).
const LifecycleDashboard = lazy(() => import('./components/lifecycle/LifecycleDashboard'));
import {
  getExamples,
  getStatus,
  getSummary,
  getUmap,
  postScore,
  type Example,
  type ScoreResp,
  type StatusResp,
  type Summary,
  type UmapPoint,
} from './api';

/** Footer partner mark tinted to its brand color. The shipped assets are
 *  white, so the image is used as an alpha mask over a brand-color fill —
 *  Cloudera orange #FF550C (deck), VAST cyan #1FD9FE (their brand spec),
 *  Fundamental violet #3D2E96 (fundamental.tech chrome). The className must
 *  set explicit h/w (masked spans have no intrinsic size). */
function PartnerLogo({
  src,
  label,
  color,
  className,
}: {
  src: string;
  label: string;
  color: string;
  className: string;
}) {
  const mask: React.CSSProperties = {
    backgroundColor: color,
    WebkitMaskImage: `url("${src}")`,
    maskImage: `url("${src}")`,
    WebkitMaskSize: 'contain',
    maskSize: 'contain',
    WebkitMaskRepeat: 'no-repeat',
    maskRepeat: 'no-repeat',
    WebkitMaskPosition: 'center',
    maskPosition: 'center',
  };
  return (
    <span
      role="img"
      aria-label={label}
      className={`inline-block opacity-90 hover:opacity-100 transition-opacity ${className}`}
      style={mask}
    />
  );
}

export default function App() {
  const [status, setStatus] = useState<StatusResp | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [umap, setUmap] = useState<UmapPoint[]>([]);

  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  // The example the form currently holds untouched, if any — its
  // expected_position drives the diagnostic ring on the embedding map.
  const [loadedExample, setLoadedExample] = useState<Example | null>(null);
  const [result, setResult] = useState<ScoreResp | null>(null);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [dataOpen, setDataOpen] = useState(false);
  const [view, setView] = useState<View>('inference');

  // Load (or reload) all dashboard data. Each call degrades on its own.
  const refresh = useCallback(() => {
    getStatus()
      .then((s) => {
        setStatus(s);
        setStatusError(false);
      })
      .catch(() => setStatusError(true));
    getSummary()
      .then(setSummary)
      .catch(() => {});
    getExamples()
      .then(setExamples)
      .catch(() => {});
    getUmap()
      .then(setUmap)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const runInference = useCallback(async () => {
    setScoring(true);
    setScoreError(null);
    try {
      const r = await postScore(formToPayload(form));
      setResult(r);
    } catch (e) {
      setScoreError(e instanceof Error ? e.message : 'Inference request failed');
    } finally {
      setScoring(false);
    }
  }, [form]);

  return (
    <div className="min-h-screen bg-surface-0 flex flex-col">
      <Header
        status={status}
        error={statusError}
        view={view}
        onViewChange={setView}
        onBuild={() => setExportOpen(true)}
        onData={() => setDataOpen(true)}
      />

      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 space-y-6 max-w-[1400px] w-full mx-auto">
        {view === 'inference' ? (
          <>
            <MetricsStrip summary={summary} />

            <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr_400px] gap-6 items-start">
              <TransactionComposer
                form={form}
                setForm={(f) => {
                  setForm(f);
                  setLoadedExample(null); // manual edit — no longer "the" example
                }}
                examples={examples}
                onLoadExample={(ex) => {
                  setForm(exampleToForm(ex.txn));
                  setLoadedExample(ex);
                }}
                onRun={runInference}
                scoring={scoring}
              />
              <ModelHeads result={result} summary={summary} scoring={scoring} error={scoreError} />
              <Suspense
                fallback={
                  <div className="bg-surface-3 rounded-lg border border-surface-3 p-4 h-[420px] animate-pulse" />
                }
              >
                <EmbeddingMap umap={umap} result={result} expected={loadedExample?.expected_position ?? null} />
              </Suspense>
            </div>
          </>
        ) : (
          <Suspense
            fallback={
              <div className="bg-surface-3 rounded-lg border border-surface-3 p-4 h-[420px] animate-pulse" />
            }
          >
            <LifecycleDashboard onBuild={() => setExportOpen(true)} />
          </Suspense>
        )}
      </main>

      {/* partner strip — bottom right, each mark in its brand color */}
      <footer className="px-4 sm:px-6 lg:px-8 pb-4 max-w-[1400px] w-full mx-auto">
        <div className="flex items-center justify-end gap-7">
          <span className="text-[9px] uppercase tracking-[0.2em] text-gray-500">powered by</span>
          <PartnerLogo src={clouderaLogo} label="Cloudera" color="#FF550C" className="h-4 w-[129px]" />
          <PartnerLogo src={vastLogo} label="VAST Data" color="#1FD9FE" className="h-[15px] w-[71px]" />
          <PartnerLogo src={fundamentalLogo} label="Fundamental (NEXUS)" color="#3D2E96" className="h-[18px] w-[134px]" />
        </div>
      </footer>

      {/* brand accent bar — mirrors the orange bottom edge of Cloudera deck
          body slides (#FF550C → ORANGE_LT #FE8756 from the toolkit palette).
          Fixed so it shows on every "slide", like the deck chrome; z-30 keeps
          it under the modals' z-50 backdrops. */}
      <div className="fixed bottom-0 inset-x-0 h-2 bg-gradient-to-r from-accent to-[#FE8756] z-30" />

      <ExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        onExported={refresh}
      />
      <DataDialog open={dataOpen} onClose={() => setDataOpen(false)} />
    </div>
  );
}
