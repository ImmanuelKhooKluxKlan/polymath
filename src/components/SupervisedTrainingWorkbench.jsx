import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';
import ModelLabPlaybackMixer from './ModelLabPlaybackMixer.jsx';

const NOTE_ACCEPT = '.mid,.midi,.json,application/json,audio/midi,audio/x-midi';

function downloadJson(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function percent(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(digits)}%` : '—';
}

function seconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  const sign = number < 0 ? '−' : '';
  const safe = Math.abs(number);
  return `${sign}${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, '0')}.${Math.floor((safe % 1) * 10)}`;
}

function statusLabel(status) {
  return String(status || 'unknown').replaceAll('-', ' ');
}

function CoordinatePlot({ alignment }) {
  const width = 1000;
  const height = 390;
  const margin = { left: 62, right: 24, top: 28, bottom: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxX = Math.max(1, Number(alignment?.plot?.referenceDurationSeconds) || 1);
  const maxY = Math.max(1, Number(alignment?.plot?.sourceDurationSeconds) || 1);
  const x = (value) => margin.left + Math.max(0, Math.min(1, Number(value) / maxX)) * plotWidth;
  const y = (value) => margin.top + plotHeight - Math.max(0, Math.min(1, Number(value) / maxY)) * plotHeight;
  const anchors = alignment?.anchors || [];
  const matches = alignment?.plot?.matches || [];
  const path = anchors.map((anchor, index) => `${index ? 'L' : 'M'} ${x(anchor.referenceTime)} ${y(anchor.observedTime)}`).join(' ');
  const ticks = Array.from({ length: 7 }, (_, index) => index / 6);
  return (
    <div className="supervision-coordinate-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Reference MIDI time compared with source video time">
        <rect x={margin.left} y={margin.top} width={plotWidth} height={plotHeight} rx="10" className="supervision-plot-background" />
        {ticks.map((fraction) => (
          <g key={fraction}>
            <line x1={x(maxX * fraction)} x2={x(maxX * fraction)} y1={margin.top} y2={margin.top + plotHeight} className="supervision-grid-line" />
            <line x1={margin.left} x2={margin.left + plotWidth} y1={y(maxY * fraction)} y2={y(maxY * fraction)} className="supervision-grid-line" />
            <text x={x(maxX * fraction)} y={height - 20} textAnchor="middle">{Math.round(maxX * fraction)}s</text>
            <text x={margin.left - 10} y={y(maxY * fraction) + 4} textAnchor="end">{Math.round(maxY * fraction)}s</text>
          </g>
        ))}
        {matches.map((match, index) => (
          <circle
            key={`${match.referenceTime}-${match.sourceTime}-${index}`}
            cx={x(match.referenceTime)}
            cy={y(match.sourceTime)}
            r={match.exactPitch ? 2.2 : 2.8}
            className={match.exactPitch ? 'exact' : 'octave'}
          />
        ))}
        {path && <path d={path} className="supervision-warp-path" />}
        {anchors.filter((anchor) => anchor.kind === 'manual').map((anchor, index) => (
          <rect key={`${anchor.referenceTime}-${index}`} x={x(anchor.referenceTime) - 5} y={y(anchor.observedTime) - 5} width="10" height="10" className="manual-anchor-point" />
        ))}
        <text x={margin.left + plotWidth / 2} y={height - 2} textAnchor="middle" className="axis-label">Desired MIDI time</text>
        <text transform={`translate(16 ${margin.top + plotHeight / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">Source video time</text>
      </svg>
      <div className="supervision-plot-legend"><span className="exact">Exact pitch</span><span className="octave">Same pitch class / octave difference</span><span className="warp">Local time map</span><span className="manual">Manual anchor</span></div>
    </div>
  );
}

function Metric({ label, value, detail, tone = '' }) {
  return <article className={`supervision-metric ${tone}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</article>;
}

export default function SupervisedTrainingWorkbench({ job, raw, initialAlignment = null, onAlignmentSaved = null }) {
  const [referenceFile, setReferenceFile] = useState(null);
  const [observedFile, setObservedFile] = useState(null);
  const [sourceDurationSeconds, setSourceDurationSeconds] = useState('');
  const [alignment, setAlignment] = useState(null);
  const [manualAnchors, setManualAnchors] = useState([]);
  const [reviewDecisions, setReviewDecisions] = useState({});
  const [selectedWindowId, setSelectedWindowId] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');

  useEffect(() => {
    const duration = Number(job?.sourceDurationSeconds ?? raw?.sourceDurationSeconds);
    if (Number.isFinite(duration) && duration > 0) setSourceDurationSeconds(duration.toFixed(4));
  }, [job?.sourceDurationSeconds, raw?.sourceDurationSeconds]);

  useEffect(() => {
    if (!initialAlignment?.id) return;
    setAlignment(initialAlignment);
    const duration = Number(initialAlignment.supervisionPackage?.timeline?.sourceDurationSeconds);
    if (Number.isFinite(duration) && duration > 0) setSourceDurationSeconds(duration.toFixed(4));
    setStatus('Loaded from private ML testing history.');
  }, [initialAlignment]);

  useEffect(() => {
    if (!alignment) return;
    setManualAnchors((alignment.manualAnchors || []).map((anchor) => ({
      referenceTime: String(anchor.referenceTime),
      observedTime: String(anchor.observedTime),
    })));
    setReviewDecisions(Object.fromEntries((alignment.qualityWindows || [])
      .filter((window) => window.decision && window.decision !== 'auto')
      .map((window) => [window.id, window.decision])));
    setSelectedWindowId((current) => (
      alignment.qualityWindows?.some((window) => window.id === current)
        ? current
        : alignment.qualityWindows?.find((window) => window.status === 'unsafe' || window.status === 'review')?.id
          || alignment.qualityWindows?.[0]?.id
          || ''
    ));
  }, [alignment]);

  const selectedWindow = useMemo(() => (
    alignment?.qualityWindows?.find((window) => window.id === selectedWindowId) || null
  ), [alignment, selectedWindowId]);
  const alignedPlayback = useMemo(() => {
    const notes = (alignment?.supervisionPackage?.notes || []).map((note) => ({
      ...note,
      instrument: 'acoustic_piano',
    }));
    return { title: `${alignment?.title || 'Desired labels'} aligned to source`, notes };
  }, [alignment]);
  const alignedInstrumentStats = useMemo(() => {
    const notes = alignedPlayback.notes || [];
    if (!notes.length) return [];
    const midis = notes.map((note) => Number(note.midi)).filter(Number.isFinite);
    const noteName = (midi) => {
      const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      const value = Math.round(midi);
      return `${names[((value % 12) + 12) % 12]}${Math.floor(value / 12) - 1}`;
    };
    return [{
      instrument: 'acoustic_piano',
      notes: notes.length,
      minimumNote: noteName(Math.min(...midis)),
      maximumNote: noteName(Math.max(...midis)),
    }];
  }, [alignedPlayback]);

  async function createAlignment() {
    if (!referenceFile || busy) return;
    if (!raw && !observedFile) {
      setStatus('Upload the current MuScriptor MIDI/JSON, or run the raw model test above first.');
      return;
    }
    setBusy(true);
    setStatus('Finding the global line, local tempo drift, pauses, and five-second confidence regions…');
    try {
      const form = new FormData();
      form.append('reference', referenceFile, referenceFile.name);
      if (observedFile) form.append('observed', observedFile, observedFile.name);
      form.append('metadata', JSON.stringify({
        jobId: raw && !observedFile ? job?.id : null,
        title: referenceFile.name.replace(/\.[^.]+$/, ''),
        sourceDurationSeconds: Number(sourceDurationSeconds) || null,
      }));
      const data = await apiRequest('/api/model-lab/alignments', { method: 'POST', body: form });
      setAlignment(data.alignment);
      onAlignmentSaved?.(data.alignment);
      setStatus(data.alignment?.archive?.saved
        ? 'Alignment ready and privately archived for detailed analysis. Unsafe and review regions remain excluded.'
        : 'Alignment ready. Unsafe and review regions are excluded from training until you approve or correct them.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function validManualAnchors() {
    return manualAnchors.map((anchor) => ({
      referenceTime: Number(anchor.referenceTime),
      observedTime: Number(anchor.observedTime),
    })).filter((anchor) => Number.isFinite(anchor.referenceTime) && Number.isFinite(anchor.observedTime));
  }

  async function recalculate(overrides = {}) {
    if (!alignment?.id || busy) return;
    setBusy(true);
    setStatus('Recalculating the nonlinear time map and training eligibility…');
    try {
      const data = await apiRequest(`/api/model-lab/alignments/${alignment.id}`, {
        method: 'PATCH',
        body: JSON.stringify({
          sourceDurationSeconds: Number(sourceDurationSeconds) || null,
          manualAnchors: validManualAnchors(),
          reviewDecisions,
          ...overrides,
        }),
      });
      setAlignment(data.alignment);
      onAlignmentSaved?.(data.alignment);
      setStatus('Corrections applied. Every change is recorded in the exported supervision package.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function addAnchorFromWindow() {
    const referenceTime = selectedWindow
      ? (selectedWindow.referenceStart + selectedWindow.referenceEnd) / 2
      : 0;
    const observedTime = selectedWindow
      ? (selectedWindow.sourceStart + selectedWindow.sourceEnd) / 2
      : 0;
    setManualAnchors((current) => [...current, {
      referenceTime: referenceTime.toFixed(3),
      observedTime: observedTime.toFixed(3),
    }].sort((a, b) => Number(a.referenceTime) - Number(b.referenceTime)));
  }

  function setWindowDecision(decision) {
    if (!selectedWindow) return;
    setReviewDecisions((current) => {
      const next = { ...current };
      if (decision === 'auto') delete next[selectedWindow.id];
      else next[selectedWindow.id] = decision;
      return next;
    });
  }

  const metrics = alignment?.metrics;
  const ready = Boolean(alignment?.supervisionPackage?.review?.readyForTraining);

  return (
    <section className="model-lab-panel supervised-training-workbench">
      <div className="model-lab-panel-heading supervision-heading">
        <div>
          <p className="eyebrow">Supervised learning preparation</p>
          <h2>Align desired piano labels to the real video timeline</h2>
          <p>The desired MIDI supplies pitches and note intent. The audio/video remains the clock. Global offset, tempo drift, pauses, and local uneven speed are estimated separately.</p>
        </div>
        <span className={`supervision-readiness ${ready ? 'ready' : 'review'}`}>{ready ? 'TRAINING READY' : alignment ? 'REVIEW REQUIRED' : 'WAITING FOR PAIR'}</span>
      </div>

      <div className="supervision-input-grid">
        <label className="supervision-file-input"><span>1. Desired / ideal piano labels</span><input type="file" accept={NOTE_ACCEPT} onChange={(event) => { setReferenceFile(event.target.files?.[0] || null); setAlignment(null); }} /><strong>{referenceFile?.name || 'Choose MIDI or note JSON'}</strong></label>
        <label className="supervision-file-input"><span>2. Current model output</span><input type="file" accept={NOTE_ACCEPT} onChange={(event) => { setObservedFile(event.target.files?.[0] || null); setAlignment(null); }} /><strong>{observedFile?.name || (raw ? 'Use the raw Model Lab result above' : 'Choose MuScriptor MIDI or JSON')}</strong></label>
        <label className="supervision-duration"><span>3. Original source length (seconds)</span><input type="number" min="1" step="0.01" value={sourceDurationSeconds} onChange={(event) => setSourceDurationSeconds(event.target.value)} placeholder="Auto from uploaded video" /><small>This bounds labels to the video; it does not stretch labels to fill the ending.</small></label>
        <button type="button" className="primary" disabled={!referenceFile || (!raw && !observedFile) || busy} onClick={createAlignment}>{busy ? 'Analysing patterns…' : 'Build supervision analysis'}</button>
      </div>
      {status && <p className="form-status">{status}</p>}

      {alignment && (
        <>
          <div className="supervision-metric-grid">
            <Metric label="Matched desired notes" value={percent(metrics.matchedReferencePercent)} detail={`${metrics.matchedNotes}/${metrics.referenceNotes}`} tone={metrics.matchedReferencePercent >= 85 ? 'good' : 'review'} />
            <Metric label="Exact pitch among matches" value={percent(metrics.exactPitchPercent)} detail={`${metrics.octaveEquivalentMatches} octave-equivalent`} />
            <Metric label="Median timing error" value={`${metrics.medianTimingResidualMs} ms`} detail={`P95 ${metrics.p95TimingResidualMs} ms`} tone={metrics.medianTimingResidualMs <= 120 ? 'good' : 'review'} />
            <Metric label="Start offset" value={`${metrics.coarseOffsetSeconds.toFixed(2)} s`} detail="Estimated; first note is not blindly trusted" />
            <Metric label="Average speed difference" value={percent(metrics.estimatedAverageSpeedDifferencePercent, 2)} detail={`Scale ${metrics.coarseScale.toFixed(4)}`} />
            <Metric label="Eligible labels" value={percent(metrics.trainingEligiblePercent)} detail={`${metrics.trainingEligibleNotes} notes`} tone={metrics.trainingEligiblePercent >= 90 ? 'good' : 'review'} />
            <Metric label="Trusted windows" value={metrics.trustedWindowCount} detail={`${metrics.qualityWindowCount} total five-second regions`} />
            <Metric label="Unsafe / rejected" value={`${metrics.unsafeWindowCount} / ${metrics.rejectedWindowCount}`} detail={`${metrics.reviewWindowCount} still need review`} tone={metrics.unsafeWindowCount ? 'danger' : 'good'} />
          </div>

          <CoordinatePlot alignment={alignment} />

          <div className="supervision-aligned-playback">
            <p>Playback below uses the <strong>desired notes after time warping</strong>. Scrub to a flagged source-time region and compare it with the original recording before accepting it.</p>
            <ModelLabPlaybackMixer raw={alignedPlayback} instrumentStats={alignedInstrumentStats} />
          </div>

          <section className="supervision-timeline-panel">
            <div className="supervision-subheading"><div><p className="eyebrow">Regional confidence</p><h3>Five-second supervision timeline</h3></div><small>Click a block to inspect, accept, reject, or anchor it.</small></div>
            <div className="supervision-quality-strip">
              {alignment.qualityWindows.map((window) => (
                <button key={window.id} type="button" title={`${window.id}: ${statusLabel(window.status)}`} className={`${window.status} ${selectedWindowId === window.id ? 'selected' : ''}`} onClick={() => setSelectedWindowId(window.id)} aria-label={`${seconds(window.sourceStart)} to ${seconds(window.sourceEnd)} ${statusLabel(window.status)}`} />
              ))}
            </div>
            <div className="supervision-status-legend"><span className="trusted">Trusted automatically</span><span className="review">Needs review</span><span className="unsafe">Unsafe</span><span className="accepted-manually">Accepted manually</span><span className="rejected">Rejected</span><span className="neutral">Reference silence</span></div>
            {selectedWindow && (
              <div className="supervision-window-inspector">
                <div><span>{selectedWindow.id}</span><strong>{seconds(selectedWindow.sourceStart)}–{seconds(selectedWindow.sourceEnd)}</strong><small>Video/source timeline</small></div>
                <div><span>Desired notes matched</span><strong>{percent(selectedWindow.matchedPercent)}</strong><small>{selectedWindow.matchedNotes}/{selectedWindow.referenceNotes}</small></div>
                <div><span>Exact pitch</span><strong>{percent(selectedWindow.exactPitchPercent)}</strong><small>Among matched notes</small></div>
                <div><span>Local speed</span><strong>{percent(selectedWindow.localTempoDifferencePercent, 1)}</strong><small>{selectedWindow.localScale.toFixed(4)}× mapping</small></div>
                <div><span>Timing error</span><strong>{selectedWindow.medianResidualMs ?? '—'} ms</strong><small>Median in region</small></div>
                <div className="supervision-window-flags"><span>Why flagged</span><strong>{selectedWindow.flags.length ? selectedWindow.flags.map(statusLabel).join(' · ') : 'No automatic warning'}</strong></div>
                <div className="supervision-review-buttons">
                  <button type="button" className={reviewDecisions[selectedWindow.id] === 'accept' ? 'primary' : 'ghost'} onClick={() => setWindowDecision('accept')}>Accept labels</button>
                  <button type="button" className={reviewDecisions[selectedWindow.id] === 'reject' ? 'danger' : 'ghost'} onClick={() => setWindowDecision('reject')}>Reject region</button>
                  <button type="button" className={!reviewDecisions[selectedWindow.id] ? 'primary' : 'ghost'} onClick={() => setWindowDecision('auto')}>Use automatic verdict</button>
                  <button type="button" className="ghost" onClick={addAnchorFromWindow}>Add anchor here</button>
                </div>
              </div>
            )}
          </section>

          <section className="supervision-anchor-panel">
            <div className="supervision-subheading"><div><p className="eyebrow">Human correction</p><h3>Manual coordinate anchors</h3></div><small>Enter a clearly matching musical moment on both timelines. Two or more anchors can describe a pause or changing tempo.</small></div>
            <div className="supervision-anchor-list">
              {manualAnchors.map((anchor, index) => (
                <div key={`${index}-${anchor.referenceTime}`}>
                  <label>Desired MIDI second<input type="number" step="0.001" min="0" value={anchor.referenceTime} onChange={(event) => setManualAnchors((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, referenceTime: event.target.value } : item))} /></label>
                  <span>→</span>
                  <label>Source video second<input type="number" step="0.001" min="0" value={anchor.observedTime} onChange={(event) => setManualAnchors((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, observedTime: event.target.value } : item))} /></label>
                  <button type="button" className="ghost" onClick={() => setManualAnchors((current) => current.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
                </div>
              ))}
              {!manualAnchors.length && <p>No manual anchors yet. The purple line is fully automatic.</p>}
            </div>
            <div className="supervision-anchor-actions"><button type="button" className="ghost" onClick={addAnchorFromWindow}>Add anchor</button><button type="button" className="primary" disabled={busy} onClick={() => recalculate()}>Apply anchors and review decisions</button></div>
          </section>

          <section className="supervision-segment-panel">
            <div className="supervision-subheading"><div><p className="eyebrow">Tempo and pause diagnostics</p><h3>Local time-map segments</h3></div><small>Large extra/missing seconds can indicate a recording pause, edit, cut, or bad automatic match.</small></div>
            <div className="model-lab-table-wrap supervision-segment-table"><table className="model-lab-table"><thead><tr><th>Source range</th><th>Desired range</th><th>Local speed</th><th>Extra / missing</th><th>Matches</th><th>Warnings</th></tr></thead><tbody>
              {alignment.tempoSegments.map((segment) => <tr key={segment.id} className={segment.flags.length ? 'flagged' : ''}><td>{seconds(segment.sourceStart)}–{seconds(segment.sourceEnd)}</td><td>{seconds(segment.referenceStart)}–{seconds(segment.referenceEnd)}</td><td>{segment.localScale.toFixed(4)}× ({percent(segment.localTempoDifferencePercent, 1)})</td><td>{segment.extraOrMissingSecondsVsCoarse > 0 ? '+' : ''}{segment.extraOrMissingSecondsVsCoarse.toFixed(3)} s</td><td>{segment.matchedNotes}</td><td>{segment.flags.length ? segment.flags.map(statusLabel).join(', ') : 'Stable'}</td></tr>)}
            </tbody></table></div>
          </section>

          <div className="supervision-export-bar">
            <div><strong>{ready ? 'Quality gate passed' : 'Not ready for training yet'}</strong><span>{ready ? 'Every musical region is trusted, manually accepted, or deliberately rejected. Rejected regions remain outside training.' : 'Correct or reject uncertain regions. Rejected regions stay out of training.'}</span></div>
            <button type="button" className="ghost" onClick={() => downloadJson(alignment.supervisionPackage, `${alignment.title}-supervision-package.json`)}>Full supervision package</button>
            <button type="button" className="ghost" onClick={() => downloadJson({ schema: alignment.supervisionPackage.schema, timeline: alignment.supervisionPackage.timeline, notes: alignment.supervisionPackage.notes.filter((note) => note.trainingEligible) }, `${alignment.title}-eligible-labels.json`)}>Eligible labels only</button>
            <button type="button" className="primary" onClick={() => downloadJson({ metrics: alignment.metrics, anchors: alignment.anchors, tempoSegments: alignment.tempoSegments, qualityWindows: alignment.qualityWindows }, `${alignment.title}-alignment-review.json`)}>Download analytics</button>
          </div>
        </>
      )}
    </section>
  );
}
