import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';
import { downloadSongMidi } from '../utils/exporters.js';
import { INSTRUMENTS } from '../data/instruments.js';
import ModelLabPlaybackMixer from '../components/ModelLabPlaybackMixer.jsx';
import MlOperationsConsole from '../components/MlOperationsConsole.jsx';
import PianoDetailsTester from '../components/PianoDetailsTester.jsx';
import SupervisedTrainingWorkbench from '../components/SupervisedTrainingWorkbench.jsx';

const MEDIA_ACCEPT = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.mp4,.mov,.webm,.mkv,.avi';

function displayName(value) {
  return String(value || 'unknown')
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function displayCheckpoint(value) {
  return String(value || 'Checking…').replaceAll('muscriptor-tester', 'polymath-tester');
}

function polymathLabel(value) {
  return String(value || '').replace(/MuScriptor/gi, 'Polymath');
}

function seconds(value) {
  const number = Number(value) || 0;
  if (number < 1) return `${Math.round(number * 1000)} ms`;
  if (number < 60) return `${number.toFixed(number < 10 ? 2 : 1)} s`;
  return `${Math.floor(number / 60)}m ${(number % 60).toFixed(1)}s`;
}

function downloadJson(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function StatCard({ label, value, detail }) {
  return (
    <article className="model-lab-stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

export default function ModelLabPage({ onNavigate, embedded = false }) {
  const [capability, setCapability] = useState(null);
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [listenAs, setListenAs] = useState('piano');
  const [instrumentFilter, setInstrumentFilter] = useState('all');
  const [history, setHistory] = useState({ rawTests: [], alignments: [] });
  const [historyStatus, setHistoryStatus] = useState('');
  const [archivedAlignment, setArchivedAlignment] = useState(null);
  const pollTimer = useRef(null);

  function clearPolling() {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    pollTimer.current = null;
  }

  useEffect(() => {
    apiRequest('/api/model-lab/capabilities')
      .then(setCapability)
      .catch((error) => setStatus(error.message));
    apiRequest('/api/model-lab/history')
      .then(setHistory)
      .catch((error) => setHistoryStatus(error.message));
    return clearPolling;
  }, []);

  useEffect(() => {
    const options = capability?.checkpoints || [];
    if (!options.length) return;
    if (!options.some((checkpoint) => checkpoint.id === selectedCheckpoint)) {
      setSelectedCheckpoint(capability.defaultCheckpoint || options[0].id);
    }
  }, [capability, selectedCheckpoint]);

  async function refreshHistory() {
    try {
      setHistory(await apiRequest('/api/model-lab/history'));
      setHistoryStatus('');
    } catch (error) {
      setHistoryStatus(error.message);
    }
  }

  async function openRawTest(recordId) {
    setHistoryStatus('Loading archived test…');
    try {
      const data = await apiRequest(`/api/model-lab/history/raw/${recordId}`);
      setCapability(data.capability);
      setFile(null);
      setJob(data.job);
      setSelectedCheckpoint(data.job.checkpoint || data.capability.defaultCheckpoint || 'original');
      setInstrumentFilter('all');
      setHistoryStatus('Archived raw test loaded.');
    } catch (error) {
      setHistoryStatus(error.message);
    }
  }

  async function openAlignment(recordId) {
    setHistoryStatus('Loading archived supervision analysis…');
    try {
      const data = await apiRequest(`/api/model-lab/alignments/${recordId}`);
      setArchivedAlignment(data.alignment);
      setHistoryStatus('Archived supervision analysis loaded.');
    } catch (error) {
      setHistoryStatus(error.message);
    }
  }

  async function poll(jobId) {
    try {
      const data = await apiRequest(`/api/model-lab/jobs/${jobId}`);
      setCapability(data.capability);
      setJob(data.job);
      if (data.job.status === 'processing') {
        pollTimer.current = window.setTimeout(() => poll(jobId), 2000);
      } else if (data.job.status === 'failed') {
        setStatus(data.job.error || 'The model test failed.');
        setBusy(false);
      } else {
        setStatus('Raw model analysis is ready.');
        setBusy(false);
        refreshHistory();
      }
    } catch (error) {
      setStatus(error.message);
      pollTimer.current = window.setTimeout(() => poll(jobId), 4000);
    }
  }

  async function startTest() {
    if (!file || busy) return;
    clearPolling();
    setBusy(true);
    setJob(null);
    setStatus('Uploading to the local Model Lab…');
    try {
      const form = new FormData();
      form.append('media', file, file.name);
      form.append('title', file.name.replace(/\.[^.]+$/, ''));
      form.append('checkpoint', selectedCheckpoint || capability?.defaultCheckpoint || 'original');
      const data = await apiRequest('/api/model-lab/jobs', { method: 'POST', body: form });
      setCapability(data.capability);
      setJob(data.job);
      setStatus('The tester checkpoint is starting. A cold GPU can take several minutes.');
      pollTimer.current = window.setTimeout(() => poll(data.job.id), 1000);
    } catch (error) {
      setStatus(error.message);
      setBusy(false);
    }
  }

  const analysis = job?.result?.analysis;
  const raw = job?.result?.raw;
  const instruments = analysis?.instruments || [];
  const checkpointOptions = capability?.checkpoints || [];
  const selectedCheckpointOption = checkpointOptions.find(
    (checkpoint) => checkpoint.id === (job?.checkpoint || selectedCheckpoint),
  );
  const visibleNotes = useMemo(() => (
    (analysis?.notePreview || [])
      .filter((note) => instrumentFilter === 'all' || note.instrument === instrumentFilter)
      .slice(0, 200)
  ), [analysis, instrumentFilter]);

  const Root = embedded ? 'div' : 'main';

  return (
    <Root className={embedded ? 'model-lab-page model-lab-page-embedded' : 'model-lab-page'}>
      <header className="model-lab-header">
        <div>
          <p className="eyebrow">Private administrator research environment</p>
          <h1>Piano Model Lab</h1>
          <p>Inspect raw notes, test them through Polymath Piano, and judge their suitability for piano-only training.</p>
        </div>
        {!embedded && <button type="button" className="ghost" onClick={() => onNavigate('admin-database')}>Back to admin console</button>}
      </header>

      <MlOperationsConsole />

      <section className="model-lab-upload-card">
        <div className="model-lab-checkpoint">
          <label htmlFor="model-lab-checkpoint-select">Model version</label>
          <select
            id="model-lab-checkpoint-select"
            value={selectedCheckpoint}
            disabled={busy || !checkpointOptions.length}
            onChange={(event) => {
              setSelectedCheckpoint(event.target.value);
              setJob(null);
              setStatus('');
            }}
          >
            {checkpointOptions.map((checkpoint) => (
              <option key={checkpoint.id} value={checkpoint.id}>{checkpoint.label}</option>
            ))}
          </select>
          <small>{checkpointOptions.find((checkpoint) => checkpoint.id === selectedCheckpoint)?.description || 'Loading available checkpoints…'}</small>
        </div>
        <div className="model-lab-checkpoint">
          <label htmlFor="model-lab-listen-as">Hear output as</label>
          <select
            id="model-lab-listen-as"
            value={listenAs}
            disabled={busy}
            onChange={(event) => setListenAs(event.target.value)}
          >
            {INSTRUMENTS.map((instrument) => (
              <option key={instrument.id} value={instrument.id}>{instrument.label}</option>
            ))}
          </select>
          <small>This changes playback only, never the model’s detected notes.</small>
        </div>
        <label className="upload-box">
          <input
            type="file"
            accept={MEDIA_ACCEPT}
            disabled={busy || capability?.enabled === false}
            onChange={(event) => {
              setFile(event.target.files?.[0] || null);
              setJob(null);
              setStatus('');
            }}
          />
          <span>{file?.name || 'Choose a test song, audio recording, or video'}</span>
          <small>Up to {Math.round((capability?.maximumSeconds || 600) / 60)} minutes are analysed</small>
        </label>
        <button
          type="button"
          className="primary"
          disabled={!file || busy || capability?.enabled === false}
          onClick={startTest}
        >
          {busy ? 'Model test running…' : `Run ${checkpointOptions.find((checkpoint) => checkpoint.id === selectedCheckpoint)?.version || 'model'} test`}
        </button>
        {capability?.enabled === false && (
          <p className="form-status error">Missing: {(capability.missing || []).join(', ') || 'Model Lab configuration'}</p>
        )}
        {job && (
          <div className="model-lab-progress">
            <div><strong>{polymathLabel(job.stage)}</strong><span>{job.progress}%</span></div>
            <div className="job-progress-track"><span style={{ width: `${job.progress}%` }} /></div>
          </div>
        )}
        {status && <p className="form-status">{status}</p>}
      </section>

      <details className="model-lab-panel model-lab-history">
        <summary>
          <span><strong>ML testing history</strong><small>{history.rawTests.length} model tests · {history.alignments.length} supervision comparisons</small></span>
          <b>Open history</b>
        </summary>
        {historyStatus && <p className="form-status">{historyStatus}</p>}
        <div className="model-lab-history-columns">
          <section>
            <div className="supervision-subheading"><div><p className="eyebrow">Model input history</p><h3>Raw song tests</h3></div></div>
            <div className="model-lab-history-list">
              {history.rawTests.map((record) => (
                <button type="button" key={record.id} onClick={() => openRawTest(record.id)}>
                  <span><strong>{record.title}</strong><small>{displayCheckpoint(record.checkpoint || 'original')} · {new Date(record.completedAt).toLocaleString()}</small></span>
                  <span><b>{record.noteCount} notes</b><small>{record.instrumentCount} instruments · {record.rapidRepeats75ms} rapid repeats</small></span>
                </button>
              ))}
              {!history.rawTests.length && <p className="muted">New raw tests will be preserved here automatically.</p>}
            </div>
          </section>
          <section>
            <div className="supervision-subheading"><div><p className="eyebrow">Desired versus model</p><h3>Supervision comparisons</h3></div></div>
            <div className="model-lab-history-list">
              {history.alignments.map((record) => (
                <button type="button" key={record.id} onClick={() => openAlignment(record.id)}>
                  <span><strong>{record.title}</strong><small>{new Date(record.updatedAt).toLocaleString()}</small></span>
                  <span><b>{record.matchedPercent ?? 0}% matched</b><small>{record.exactPitchPercent ?? 0}% exact · {record.trainingEligiblePercent ?? 0}% eligible</small></span>
                </button>
              ))}
              {!history.alignments.length && <p className="muted">Desired/model comparisons will appear here after analysis.</p>}
            </div>
          </section>
        </div>
        <p className="privacy-note">Private administrator history. Note events, analytics, hashes, and review decisions are retained; the uploaded source song/video is deleted after transcription.</p>
      </details>

      <SupervisedTrainingWorkbench
        job={job}
        raw={raw}
        initialAlignment={archivedAlignment}
        onAlignmentSaved={refreshHistory}
      />

      {analysis && raw && (
        <>
          <section className="model-lab-result-heading">
            <div>
              <p className="eyebrow">Raw result</p>
              <h2>{raw.title || job.title}</h2>
              <p>{polymathLabel(analysis.model.provider)}</p>
              <p>
                <strong>{selectedCheckpointOption?.label || displayCheckpoint(job.checkpoint)}</strong>
                {' · '}listening as {INSTRUMENTS.find((instrument) => instrument.id === listenAs)?.label || listenAs}
              </p>
            </div>
            <div className="model-lab-actions">
              <button type="button" className="ghost" onClick={() => downloadJson(analysis, `${job.title}-analysis.json`)}>Analysis JSON</button>
              <button type="button" className="ghost" onClick={() => downloadJson(raw, `${job.title}-raw.json`)}>Raw notes JSON</button>
              <button type="button" className="primary" onClick={() => downloadSongMidi(raw)}>Download MIDI</button>
            </div>
          </section>

          <section className="model-lab-detected-instruments" aria-label="Automatically detected instruments">
            <div className="model-lab-detected-heading">
              <div>
                <p className="eyebrow">Automatically detected</p>
                <h2>Instruments in this recording</h2>
              </div>
              <strong>{instruments.length}</strong>
            </div>
            <div className="model-lab-instrument-cards">
              {instruments.map((instrument) => (
                <button
                  type="button"
                  key={instrument.instrument}
                  className={instrumentFilter === instrument.instrument ? 'active' : ''}
                  onClick={() => setInstrumentFilter((current) => (
                    current === instrument.instrument ? 'all' : instrument.instrument
                  ))}
                >
                  <span>{displayName(instrument.instrument)}</span>
                  <strong>{instrument.notes} notes</strong>
                  <small>{instrument.noteSharePercent}% · {instrument.minimumNote}–{instrument.maximumNote}</small>
                </button>
              ))}
            </div>
            <small>Model predictions appear automatically. Click an instrument to filter the MIDI inspection table.</small>
          </section>

          <section className="model-lab-stat-grid" aria-label="Major model statistics">
            <StatCard label="Valid notes" value={analysis.headline.validNotes} detail={`${analysis.midi.totalChannelEvents} MIDI note events`} />
            <StatCard label="Instruments" value={analysis.headline.detectedInstrumentGroups} detail="Model-predicted groups" />
            <StatCard label="Pitch range" value={analysis.headline.pitchRange} detail={`${analysis.pitch.rangeSemitones} semitones`} />
            <StatCard label="Detected length" value={seconds(analysis.headline.recordingSeconds)} />
            <StatCard label="Maximum polyphony" value={analysis.headline.maximumPolyphony} detail={`${analysis.timing.averagePolyphony} average`} />
            <StatCard label="Rapid repeats" value={analysis.headline.rapidRepeats75ms} detail="Same pitch within 75 ms" />
            <StatCard label="Note density" value={analysis.timing.noteDensityPerSecond} detail="Notes per second" />
            <StatCard label="Median sustain" value={seconds(analysis.timing.medianDurationSeconds)} />
          </section>

          <ModelLabPlaybackMixer raw={raw} instrumentStats={instruments} initialSound={listenAs} />

          <PianoDetailsTester
            raw={raw}
            maximumPolyphony={analysis.headline.maximumPolyphony}
          />

          <section className="model-lab-panel">
            <div className="model-lab-panel-heading">
              <div><p className="eyebrow">Instrument detection</p><h2>Predicted instrument groups</h2></div>
              <small>These labels require human verification.</small>
            </div>
            <div className="model-lab-table-wrap">
              <table className="model-lab-table">
                <thead><tr><th>Instrument</th><th>Notes</th><th>Share</th><th>Pitch range</th><th>Unique pitches</th><th>Average sustain</th></tr></thead>
                <tbody>
                  {instruments.map((instrument) => (
                    <tr key={instrument.instrument}>
                      <td><strong>{displayName(instrument.instrument)}</strong></td>
                      <td>{instrument.notes}</td>
                      <td>{instrument.noteSharePercent}%</td>
                      <td>{instrument.minimumNote}–{instrument.maximumNote}</td>
                      <td>{instrument.uniquePitches}</td>
                      <td>{seconds(instrument.averageDurationSeconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="model-lab-two-column">
            <article className="model-lab-panel">
              <p className="eyebrow">Pitch-class histogram</p>
              <h2>Which notes dominate?</h2>
              <div className="model-lab-bars">
                {analysis.pitch.pitchClasses.map((pitchClass) => (
                  <div key={pitchClass.label}>
                    <strong>{pitchClass.label}</strong>
                    <span><i style={{ width: `${pitchClass.percent}%` }} /></span>
                    <small>{pitchClass.count} · {pitchClass.percent}%</small>
                  </div>
                ))}
              </div>
            </article>
            <article className="model-lab-panel">
              <p className="eyebrow">Most frequent pitches</p>
              <h2>Top detected keys</h2>
              <div className="model-lab-pitch-list">
                {analysis.pitch.topPitches.slice(0, 12).map((pitch) => (
                  <div key={pitch.midi}><strong>{pitch.note}</strong><span>MIDI {pitch.midi}</span><small>{pitch.count} notes · {pitch.percent}%</small></div>
                ))}
              </div>
            </article>
          </section>

          <section className="model-lab-two-column">
            <article className="model-lab-panel">
              <p className="eyebrow">Timing health</p>
              <h2>Stutter and sustain checks</h2>
              <dl className="model-lab-definition-list">
                <div><dt>Shortest note</dt><dd>{seconds(analysis.timing.minimumDurationSeconds)}</dd></div>
                <div><dt>Average note</dt><dd>{seconds(analysis.timing.averageDurationSeconds)}</dd></div>
                <div><dt>Longest note</dt><dd>{seconds(analysis.timing.maximumDurationSeconds)}</dd></div>
                <div><dt>Notes under 100 ms</dt><dd>{analysis.timing.notesUnder100ms}</dd></div>
                <div><dt>Same-pitch overlaps</dt><dd>{analysis.timing.samePitchOverlaps}</dd></div>
                <div><dt>Largest onset cluster</dt><dd>{analysis.timing.largestOnsetCluster}</dd></div>
              </dl>
            </article>
            <article className="model-lab-panel">
              <p className="eyebrow">Honest limitations</p>
              <h2>What these numbers cannot prove</h2>
              <ul className="model-lab-notes">
                {analysis.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
              </ul>
              {analysis.warnings.length > 0 && (
                <div className="model-lab-warning-list">
                  {analysis.warnings.map((warning) => <p key={warning}>{warning}</p>)}
                </div>
              )}
            </article>
          </section>

          <section className="model-lab-panel">
            <div className="model-lab-panel-heading">
              <div><p className="eyebrow">MIDI inspection</p><h2>First detected notes</h2></div>
              <select value={instrumentFilter} onChange={(event) => setInstrumentFilter(event.target.value)}>
                <option value="all">All instruments</option>
                {instruments.map((instrument) => (
                  <option key={instrument.instrument} value={instrument.instrument}>{displayName(instrument.instrument)}</option>
                ))}
              </select>
            </div>
            <div className="model-lab-table-wrap model-lab-note-table">
              <table className="model-lab-table">
                <thead><tr><th>Instrument</th><th>Pitch</th><th>MIDI</th><th>Start</th><th>Duration</th><th>Velocity*</th></tr></thead>
                <tbody>
                  {visibleNotes.map((note, index) => (
                    <tr key={`${note.instrument}-${note.midi}-${note.time}-${index}`}>
                      <td>{displayName(note.instrument)}</td>
                      <td><strong>{note.note}</strong></td>
                      <td>{note.midi}</td>
                      <td>{seconds(note.time)}</td>
                      <td>{seconds(note.duration)}</td>
                      <td>{note.velocity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <small>* Velocity is currently synthetic, not measured from the performance.</small>
          </section>
        </>
      )}
    </Root>
  );
}
