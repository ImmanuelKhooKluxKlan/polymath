import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';
import '../mlOperationsConsole.css';

const TABS = [
  ['process', 'Process'],
  ['experiments', 'Experiments & diff'],
  ['tokens', 'Tokens & platforms'],
  ['change', 'Change weights'],
];

const INITIAL_DRAFT = {
  name: 'Piano Phase 2 candidate',
  datasetId: 'phase-2-v001',
  version: 'phase2-v001',
  baseVersion: 'phase1-v002',
  epochs: 1,
  trainLastLayers: 1,
  learningRate: 0.000002,
  timingTokenWeight: 1.15,
  noteOffTokenWeight: 1.25,
  eosTokenWeight: 1.2,
};

function title(value) {
  return String(value || 'unknown')
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function displayCheckpoint(value) {
  return String(value || '—').replaceAll('muscriptor-tester', 'polymath-tester');
}

function number(value, digits = 4) {
  if (!Number.isFinite(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function percent(value, digits = 2) {
  if (!Number.isFinite(Number(value))) return '—';
  return `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(digits)}%`;
}

function downloadJson(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function isImprovement(item) {
  const delta = Number(item?.relativeChangePercent ?? item?.delta);
  if (!Number.isFinite(delta) || delta === 0) return null;
  return item.preferredDirection === 'down' ? delta < 0 : delta > 0;
}

function StatusPill({ value }) {
  const clean = String(value || 'unknown').toLowerCase();
  const tone = ['evaluated', 'trained', 'completed'].includes(clean)
    ? 'good'
    : ['failed', 'cancelled'].includes(clean)
      ? 'danger'
      : ['training', 'evaluating', 'in_queue', 'in_progress'].includes(clean)
        ? 'live'
        : 'neutral';
  return <span className={`mlops-status ${tone}`}>{title(clean)}</span>;
}

function MetricDiff({ label, metric }) {
  const delta = metric.relativeChangePercent ?? (
    Number.isFinite(metric.baseline) && Number.isFinite(metric.candidate) && metric.baseline !== 0
      ? ((metric.candidate - metric.baseline) / Math.abs(metric.baseline)) * 100
      : null
  );
  const improved = isImprovement({ ...metric, relativeChangePercent: delta });
  return (
    <article className={`mlops-metric ${improved === true ? 'improved' : improved === false ? 'regressed' : ''}`}>
      <span>{title(label)}</span>
      <div><strong>{number(metric.candidate, 6)}</strong><b>{percent(delta)}</b></div>
      <small>Original {number(metric.baseline, 6)} · lower is better only where marked</small>
    </article>
  );
}

function ErrorDiffTable({ rows = [] }) {
  if (!rows.length) return <p className="mlops-empty">Detailed decoded error differences appear after frozen evaluation.</p>;
  return (
    <div className="mlops-table-wrap">
      <table className="mlops-table">
        <thead><tr><th>Error / quality measure</th><th>Original</th><th>Candidate</th><th>Difference</th><th>Verdict</th></tr></thead>
        <tbody>
          {rows.map((row) => {
            const improved = isImprovement(row);
            return (
              <tr key={row.label}>
                <td><strong>{row.label}</strong></td>
                <td>{number(row.baseline)}</td>
                <td>{number(row.candidate)}</td>
                <td>{row.delta > 0 ? '+' : ''}{number(row.delta)}</td>
                <td><span className={`mlops-diff-verdict ${improved ? 'good' : improved === false ? 'bad' : ''}`}>{improved ? 'Improved' : improved === false ? 'Regressed' : 'No change'}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ExperimentDetail({ experiment, onRefresh, onEvaluate, onCancel, busy }) {
  if (!experiment) return <p className="mlops-empty">Select an experiment to inspect it.</p>;
  const metrics = Object.entries(experiment.metrics || {});
  const training = experiment.data?.training;
  const validation = experiment.data?.validation;
  return (
    <div className="mlops-detail">
      <header className="mlops-detail-header">
        <div><p className="eyebrow">{experiment.version}</p><h3>{experiment.name}</h3><p>{experiment.stage}</p></div>
        <StatusPill value={experiment.status} />
      </header>

      <div className="mlops-command-row">
        <button type="button" className="ghost" disabled={busy} onClick={onRefresh}>Refresh state</button>
        <button type="button" className="ghost" onClick={() => downloadJson(experiment, `${experiment.version}-experiment-audit.json`)}>Download audit JSON</button>
        {!experiment.readOnly && experiment.status === 'trained' && <button type="button" className="primary" disabled={busy} onClick={onEvaluate}>Run frozen evaluation</button>}
        {!experiment.readOnly && experiment.remote?.jobId && <button type="button" className="danger" disabled={busy} onClick={onCancel}>Cancel GPU job</button>}
      </div>

      <section className="mlops-summary-grid">
        <article><span>Dataset</span><strong>{experiment.datasetId}</strong><small>{experiment.instrument}</small></article>
        <article><span>Base</span><strong>{displayCheckpoint(experiment.baseCheckpoint?.label)}</strong><small>{experiment.baseCheckpoint?.immutable ? 'Immutable' : 'Review required'}</small></article>
        <article><span>Candidate</span><strong>{displayCheckpoint(experiment.candidateCheckpoint?.label)}</strong><small>{experiment.candidateCheckpoint?.promoted ? 'Production' : 'Not promoted'}</small></article>
        <article><span>Compute</span><strong>RunPod GPU</strong><small>{experiment.configuration?.precision || 'BF16'} · {experiment.configuration?.optimizer || 'AdamW'}</small></article>
      </section>

      {experiment.remote && (
        <section className="mlops-live-job">
          <div><span>Live RunPod operation</span><strong>{title(experiment.remote.operation)}</strong></div>
          <div><span>Worker state</span><StatusPill value={experiment.remote.state} /></div>
          <div><span>Private job ID</span><code>{experiment.remote.jobId}</code></div>
        </section>
      )}

      <section className="mlops-two-column">
        <article className="mlops-inset">
          <p className="eyebrow">Weight scope</p><h4>What can change</h4>
          <ul>{(experiment.weightScope?.changed || []).map((item) => <li key={item}>{item}</li>)}</ul>
          <small>{experiment.weightScope?.exactTensorDeltaAvailable ? 'Per-tensor delta norms were captured.' : experiment.weightScope?.note || 'Per-tensor delta norms will be attached by supported future training runs.'}</small>
        </article>
        <article className="mlops-inset">
          <p className="eyebrow">Frozen protection</p><h4>What cannot change</h4>
          <ul>{(experiment.weightScope?.frozen || []).map((item) => <li key={item}>{item}</li>)}</ul>
          <small>Every version writes to a new tester directory. The console has no production-promotion endpoint.</small>
        </article>
      </section>

      {experiment.weightDelta && (
        <section>
          <div className="mlops-section-heading"><div><p className="eyebrow">Numerical weight diff</p><h4>How far trainable tensors moved</h4></div><small>{number(experiment.weightDelta.changedParameters)} of {number(experiment.weightDelta.trainableParameters)} trainable values changed</small></div>
          <div className="mlops-summary-grid">
            <article><span>Trainable tensors</span><strong>{number(experiment.weightDelta.trainableTensorCount)}</strong></article>
            <article><span>Changed values</span><strong>{number(experiment.weightDelta.changedPercent, 6)}%</strong></article>
            <article><span>Overall RMS delta</span><strong>{number(experiment.weightDelta.overallRmsDelta, 10)}</strong></article>
            <article><span>Maximum absolute delta</span><strong>{number(experiment.weightDelta.maximumAbsoluteDelta, 10)}</strong></article>
          </div>
          <div className="mlops-table-wrap"><table className="mlops-table"><thead><tr><th>Tensor</th><th>Parameters</th><th>Changed</th><th>RMS delta</th><th>Relative RMS</th><th>Maximum</th></tr></thead><tbody>{experiment.weightDelta.tensors?.map((tensor) => <tr key={tensor.name}><td><code>{tensor.name}</code></td><td>{number(tensor.parameters)}</td><td>{number(tensor.changedPercent, 6)}%</td><td>{number(tensor.rmsDelta, 10)}</td><td>{number(tensor.relativeRmsDeltaPercent, 8)}%</td><td>{number(tensor.maximumAbsoluteDelta, 10)}</td></tr>)}</tbody></table></div>
        </section>
      )}

      {(training || validation) && (
        <section>
          <div className="mlops-section-heading"><div><p className="eyebrow">Data lineage</p><h4>Exactly what reached the optimizer</h4></div></div>
          <div className="mlops-data-grid">
            {[['Training', training], ['Frozen validation', validation]].map(([label, data]) => data && (
              <article key={label}><span>{label}</span><strong>{number(data.clips)} clips · {number(data.songs)} songs</strong><small>{number(data.audioSeconds)} seconds · {number(data.averageTokens, 2)} average tokens · {number(data.minimumTokens)}–{number(data.maximumTokens)} token range</small></article>
            ))}
          </div>
          {experiment.data?.splitNote && <p className="mlops-callout">{experiment.data.splitNote}</p>}
        </section>
      )}

      <section>
        <div className="mlops-section-heading"><div><p className="eyebrow">Accuracy and loss</p><h4>Original versus candidate</h4></div><small>No single score means “90% accurate”; each metric tests a different failure mode.</small></div>
        <div className="mlops-metric-grid">
          {metrics.map(([label, metric]) => <MetricDiff key={label} label={label} metric={metric} />)}
          {!metrics.length && <p className="mlops-empty">Metrics appear after training and frozen decoding.</p>}
        </div>
      </section>

      <section>
        <div className="mlops-section-heading"><div><p className="eyebrow">Detailed diff</p><h4>What improved, regressed, or was ignored</h4></div></div>
        <ErrorDiffTable rows={experiment.errorDiff} />
      </section>

      <section className="mlops-decision">
        <div><p className="eyebrow">Promotion gate</p><h4>{experiment.decision?.approved ? 'Approved candidate' : 'Research only'}</h4></div>
        <p>{experiment.decision?.summary}</p>
        {!experiment.decision?.commercialUseAllowed && <strong>Commercial use is not cleared.</strong>}
      </section>

      <details className="mlops-audit">
        <summary>Full experiment audit trail ({experiment.audit?.length || 0})</summary>
        <ol>{(experiment.audit || []).map((entry, index) => <li key={`${entry.at}-${index}`}><time>{new Date(entry.at).toLocaleString()}</time><strong>{title(entry.action)}</strong><span>{entry.actor} · {entry.detail}</span></li>)}</ol>
      </details>
      <details className="mlops-audit mlops-raw-record">
        <summary>Raw safe experiment record</summary>
        <pre>{JSON.stringify(experiment, null, 2)}</pre>
      </details>
    </div>
  );
}

export default function MlOperationsConsole() {
  const [tab, setTab] = useState('process');
  const [overview, setOverview] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [selectedId, setSelectedId] = useState('phase1-v002');
  const [draft, setDraft] = useState(INITIAL_DRAFT);
  const [created, setCreated] = useState(null);
  const [confirmVersion, setConfirmVersion] = useState('');
  const [rightsAcknowledged, setRightsAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');

  const selected = useMemo(() => experiments.find((item) => item.id === selectedId) || experiments[0], [experiments, selectedId]);

  async function load() {
    const data = await apiRequest('/api/model-lab/ml/overview');
    setOverview(data.overview);
    setExperiments(data.experiments || []);
    if (!selectedId && data.experiments?.length) setSelectedId(data.experiments[0].id);
  }

  useEffect(() => {
    load().catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    if (!selected?.remote?.jobId) return undefined;
    const timer = window.setInterval(() => {
      refreshExperiment(selected.id, false).catch(() => {});
    }, 5000);
    return () => window.clearInterval(timer);
  }, [selected?.id, selected?.remote?.jobId]);

  function replaceExperiment(experiment) {
    setExperiments((current) => {
      const next = current.filter((item) => item.id !== experiment.id);
      return [experiment, ...next].sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)));
    });
    setSelectedId(experiment.id);
    if (created?.id === experiment.id) setCreated(experiment);
  }

  async function refreshExperiment(id = selected?.id, announce = true) {
    if (!id) return;
    if (announce) setBusy(true);
    try {
      const data = await apiRequest(`/api/model-lab/ml/experiments/${id}`);
      replaceExperiment(data.experiment);
      if (announce) setStatus('Experiment state refreshed.');
    } catch (error) {
      if (announce) setStatus(error.message);
      throw error;
    } finally {
      if (announce) setBusy(false);
    }
  }

  async function createExperiment(event) {
    event.preventDefault();
    setBusy(true);
    setStatus('Saving a guarded candidate configuration…');
    try {
      const data = await apiRequest('/api/model-lab/ml/experiments', {
        method: 'POST',
        body: JSON.stringify(draft),
      });
      setCreated(data.experiment);
      replaceExperiment(data.experiment);
      setConfirmVersion('');
      setRightsAcknowledged(false);
      setStatus('Draft saved. No GPU ran and no weights changed yet.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function startTraining() {
    if (!created) return;
    setBusy(true);
    setStatus('Submitting the candidate to RunPod…');
    try {
      const data = await apiRequest(`/api/model-lab/ml/experiments/${created.id}/train`, {
        method: 'POST',
        body: JSON.stringify({ confirmVersion, rightsAcknowledged }),
      });
      setCreated(data.experiment);
      replaceExperiment(data.experiment);
      setTab('experiments');
      setStatus('Training submitted. This page will refresh the private job automatically.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function evaluateExperiment() {
    if (!selected) return;
    setBusy(true);
    try {
      const data = await apiRequest(`/api/model-lab/ml/experiments/${selected.id}/evaluate`, { method: 'POST' });
      replaceExperiment(data.experiment);
      setStatus('Frozen evaluation submitted to RunPod.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function cancelExperiment() {
    if (!selected || !window.confirm('Cancel this RunPod job? The candidate record and audit history will remain.')) return;
    setBusy(true);
    try {
      const data = await apiRequest(`/api/model-lab/ml/experiments/${selected.id}/cancel`, { method: 'POST' });
      replaceExperiment(data.experiment);
      setStatus('Cancellation requested.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function scrollToFeedData() {
    document.querySelector('.supervised-training-workbench')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return (
    <section className="mlops-console" aria-label="Machine learning operations console">
      <header className="mlops-heading">
        <div><p className="eyebrow">Private machine learning operations</p><h2>Training control room</h2><p>Follow every stage from evidence to candidate weights, with history, metrics, and safety gates kept visible.</p></div>
        <div className="mlops-heading-actions"><StatusPill value={overview?.runpod?.configured ? 'RunPod configured' : 'RunPod unavailable'} /><button type="button" className="ghost" onClick={scrollToFeedData}>Feed supervision data</button></div>
      </header>

      <nav className="mlops-tabs" aria-label="Machine learning console views">
        {TABS.map(([id, label]) => <button key={id} type="button" className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>)}
      </nav>
      {status && <p className="mlops-message">{status}</p>}

      {tab === 'process' && overview && (
        <div className="mlops-view">
          {overview.storage?.warning && <p className="mlops-callout"><strong>Storage warning:</strong> {overview.storage.warning}</p>}
          {!overview.runpod?.configured && <p className="mlops-callout"><strong>GPU unavailable:</strong> the backend is missing {overview.runpod?.missingConfigurationNames?.join(', ') || 'RunPod configuration'}.</p>}
          <div className="mlops-evidence-grid">
            <article><span>Raw tests archived</span><strong>{overview.evidence?.rawModelTests ?? 0}</strong></article>
            <article><span>Alignments archived</span><strong>{overview.evidence?.supervisionAlignments ?? 0}</strong></article>
            <article><span>Training-ready</span><strong>{overview.evidence?.trainingReadyAlignments ?? 0}</strong></article>
            <article><span>Experiment versions</span><strong>{experiments.length}</strong></article>
          </div>
          <section className="mlops-pipeline">
            {overview.pipeline.map((step, index) => (
              <article key={step.id}><b>{index + 1}</b><div><span>{step.platform}</span><h3>{step.label}</h3><p>{step.detail}</p></div></article>
            ))}
          </section>
          <section className="mlops-two-column">
            <article className="mlops-inset"><p className="eyebrow">Platforms</p><h3>Where each responsibility runs</h3><dl className="mlops-platform-list">{overview.platforms.map((platform) => <div key={platform.name}><dt>{platform.name}</dt><dd>{platform.role}</dd></div>)}</dl></article>
            <article className="mlops-inset"><p className="eyebrow">Hard safety rules</p><h3>What the console refuses to do</h3><ul><li>Overwrite the original checkpoint</li><li>Reuse an existing candidate version</li><li>Send API or storage keys to the browser</li><li>Train without typed confirmation and rights acknowledgement</li><li>Promote a checkpoint directly to the public website</li></ul></article>
          </section>
        </div>
      )}

      {tab === 'experiments' && (
        <div className="mlops-experiment-layout">
          <aside className="mlops-experiment-list">
            {experiments.map((experiment) => <button key={experiment.id} type="button" className={selected?.id === experiment.id ? 'active' : ''} onClick={() => setSelectedId(experiment.id)}><span><strong>{experiment.version}</strong><small>{experiment.name}</small></span><StatusPill value={experiment.status} /></button>)}
          </aside>
          <ExperimentDetail experiment={selected} busy={busy} onRefresh={() => refreshExperiment()} onEvaluate={evaluateExperiment} onCancel={cancelExperiment} />
        </div>
      )}

      {tab === 'tokens' && overview && (
        <div className="mlops-view">
          <section className="mlops-token-intro">
            <div><p className="eyebrow">Model language</p><h3>{overview.tokenSystem.name}</h3><p>The network does not write MIDI directly. It predicts numbered events: wait, choose instrument, turn a pitch on, turn it off, tie it, then end.</p></div>
            <div><strong>{number(overview.tokenSystem.modelCardinality)}</strong><span>event IDs</span><small>{overview.tokenSystem.sequenceLimit} maximum per clip · {overview.tokenSystem.timeResolutionHz} timing steps/second</small></div>
          </section>
          <div className="mlops-table-wrap"><table className="mlops-table"><thead><tr><th>Token family</th><th>ID range</th><th>Count</th><th>What it means</th></tr></thead><tbody>{overview.tokenSystem.ranges.map((range) => <tr key={range.label}><td><strong>{range.label}</strong></td><td><code>{range.first}–{range.last}</code></td><td>{range.last - range.first + 1}</td><td>{range.purpose}</td></tr>)}</tbody></table></div>
          <p className="mlops-callout"><strong>Real-life analogy:</strong> the audio encoder is the ear; these tokens are a compact musical sentence; supervised learning compares the sentence with your approved answer and nudges selected weights to make the next sentence less wrong.</p>
        </div>
      )}

      {tab === 'change' && (
        <div className="mlops-change-view">
          <section className="mlops-inset">
            <div className="mlops-section-heading"><div><p className="eyebrow">Step 1</p><h3>Prepare and review evidence</h3></div></div>
            <p>Use the supervision workbench below to compare desired MIDI/JSON with raw model output. Accept, reject, or anchor uncertain five-second regions before building the prepared dataset on the private volume. A trainable candidate requires at least 20 distinct training songs; clips from three songs are not 222 independent songs.</p>
            <button type="button" className="ghost" onClick={scrollToFeedData}>Open data workbench</button>
          </section>
          <form className="mlops-candidate-form" onSubmit={createExperiment}>
            <div className="mlops-section-heading"><div><p className="eyebrow">Step 2</p><h3>Create a new candidate version</h3></div><small>This saves configuration only. It does not start or bill a GPU.</small></div>
            <div className="mlops-form-grid">
              <label>Experiment name<input value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} required /></label>
              <label>Prepared dataset ID<input value={draft.datasetId} onChange={(event) => setDraft({ ...draft, datasetId: event.target.value })} pattern="[a-z0-9][a-z0-9-]{2,50}" required /></label>
              <label>New candidate version<input value={draft.version} onChange={(event) => setDraft({ ...draft, version: event.target.value })} pattern="phase[0-9]+-v[0-9]{3,}" required /></label>
              <label>Continue from checkpoint<input value={draft.baseVersion} onChange={(event) => setDraft({ ...draft, baseVersion: event.target.value })} pattern="original|phase[0-9]+-v[0-9]{3,}" required /><small>Use the current incumbent so a new phase does not forget earlier gains.</small></label>
              <label>Epochs<select value={draft.epochs} onChange={(event) => setDraft({ ...draft, epochs: Number(event.target.value) })}><option value="1">1 — safest trial</option><option value="2">2</option><option value="3">3 — maximum</option></select></label>
              <label>Final transformer blocks<select value={draft.trainLastLayers} onChange={(event) => setDraft({ ...draft, trainLastLayers: Number(event.target.value) })}><option value="1">1 — conservative</option><option value="2">2 — broader change</option></select></label>
              <label>Learning rate<input type="number" min="0.0000001" max="0.00001" step="0.0000001" value={draft.learningRate} onChange={(event) => setDraft({ ...draft, learningRate: Number(event.target.value) })} required /></label>
              <label>Timing-token weight<input type="number" min="0.5" max="3" step="0.05" value={draft.timingTokenWeight} onChange={(event) => setDraft({ ...draft, timingTokenWeight: Number(event.target.value) })} /></label>
              <label>Note-off weight<input type="number" min="0.5" max="3" step="0.05" value={draft.noteOffTokenWeight} onChange={(event) => setDraft({ ...draft, noteOffTokenWeight: Number(event.target.value) })} /></label>
              <label>End-token weight<input type="number" min="0.5" max="3" step="0.05" value={draft.eosTokenWeight} onChange={(event) => setDraft({ ...draft, eosTokenWeight: Number(event.target.value) })} /></label>
            </div>
            <button type="submit" className="primary" disabled={busy}>Save candidate draft</button>
          </form>

          {created && (
            <section className="mlops-train-gate">
              <div className="mlops-section-heading"><div><p className="eyebrow">Step 3 · guarded GPU action</p><h3>Train {created.version}</h3></div><StatusPill value={created.status} /></div>
              <p>The worker validates the dataset and <code>{created.configuration?.baseVersion || 'original'}</code> first. It writes a new tester checkpoint only if validation loss improves; it never overwrites its base.</p>
              <label className="mlops-check"><input type="checkbox" checked={rightsAcknowledged} onChange={(event) => setRightsAcknowledged(event.target.checked)} /><span>I confirm that Polymath has permission to use every source and label in this prepared dataset for model training.</span></label>
              <label>Type <strong>{created.version}</strong> to confirm<input value={confirmVersion} onChange={(event) => setConfirmVersion(event.target.value)} autoComplete="off" /></label>
              <button type="button" className="primary" disabled={busy || !overview?.runpod?.configured || !rightsAcknowledged || confirmVersion !== created.version || created.status !== 'draft'} onClick={startTraining}>Start candidate training on RunPod</button>
              <small>GPU billing begins only when RunPod starts a worker for the submitted job.</small>
            </section>
          )}
        </div>
      )}
    </section>
  );
}
