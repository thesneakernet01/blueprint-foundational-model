import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import Header from './components/Header';
import MetricsStrip from './components/MetricsStrip';
import TransactionComposer, {
  DEFAULT_FORM,
  formToPayload,
  type FormState,
} from './components/TransactionComposer';
import ModelHeads from './components/ModelHeads';
import ExportDialog from './components/ExportDialog';
import DataDialog from './components/DataDialog';

// Recharts is heavy and only the embedding map needs it — load it in its own
// chunk so the first paint (header / composer / heads) isn't blocked on it.
const EmbeddingMap = lazy(() => import('./components/EmbeddingMap'));
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

export default function App() {
  const [status, setStatus] = useState<StatusResp | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [umap, setUmap] = useState<UmapPoint[]>([]);

  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [result, setResult] = useState<ScoreResp | null>(null);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [dataOpen, setDataOpen] = useState(false);

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
        onBuild={() => setExportOpen(true)}
        onData={() => setDataOpen(true)}
      />

      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 space-y-6 max-w-[1400px] w-full mx-auto">
        <MetricsStrip summary={summary} />

        <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr_400px] gap-6 items-start">
          <TransactionComposer
            form={form}
            setForm={setForm}
            examples={examples}
            onRun={runInference}
            scoring={scoring}
          />
          <ModelHeads result={result} summary={summary} scoring={scoring} error={scoreError} />
          <Suspense
            fallback={
              <div className="bg-surface-2 rounded-lg border border-surface-3 p-4 h-[420px] animate-pulse" />
            }
          >
            <EmbeddingMap umap={umap} result={result} />
          </Suspense>
        </div>
      </main>

      <ExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        onExported={refresh}
      />
      <DataDialog open={dataOpen} onClose={() => setDataOpen(false)} />
    </div>
  );
}
