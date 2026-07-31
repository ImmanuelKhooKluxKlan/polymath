import { useEffect, useState } from 'react';
import { transcribeMediaFile } from '../utils/mediaTranscriber.js';
import { midiToNote } from '../engine/noteMath.js';

const MAX_MEDIA_BYTES = 350 * 1024 * 1024;
const MEDIA_ACCEPT = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.mp4,.webm,.mov';
const INSTRUMENT_RANGES = {
  piano: [21, 108],
  guitar: [40, 79],
  'electric-guitar': [40, 79],
  fiddle: [48, 88],
  violin: [55, 88],
  cello: [36, 75],
  banjo: [50, 79],
  mandolin: [55, 88],
  dobro: [43, 74],
  'upright-bass': [28, 61],
  ukulele: [60, 81],
  synth: [48, 84],
  flute: [60, 96],
  saxophone: [46, 82],
  trumpet: [52, 82],
  clarinet: [50, 88],
};

function percent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function defaultTarget(instrument) {
  if (instrument === 'drums') return 'drums';
  if (['upright-bass', 'bass-guitar', 'cello'].includes(instrument)) return 'bass';
  return 'melody';
}

function adaptDraftToInstrument(song, instrument) {
  if (instrument === 'drums' || !INSTRUMENT_RANGES[instrument]) {
    return { ...song, instrument };
  }
  const [minimum, maximum] = INSTRUMENT_RANGES[instrument];
  return {
    ...song,
    instrument,
    notes: song.notes.map((note) => {
      const originalMidi = Number(note.midi);
      let midi = originalMidi;
      while (midi < minimum) midi += 12;
      while (midi > maximum) midi -= 12;
      if (midi < minimum) midi = minimum;
      if (midi > maximum) midi = maximum;
      return {
        ...note,
        note: midiToNote(midi),
        midi,
        originalMidi: midi === originalMidi ? undefined : originalMidi,
      };
    }),
    transcription: {
      ...song.transcription,
      destinationInstrument: instrument,
      playableRange: [minimum, maximum],
    },
  };
}

export default function MediaTranslationPanel({ onReadyFile, instrument = 'piano' }) {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [target, setTarget] = useState(defaultTarget(instrument));
  const [busy, setBusy] = useState(false);
  const [hasRights, setHasRights] = useState(false);
  const [status, setStatus] = useState('Choose a recording you own or have permission to process.');
  const [result, setResult] = useState(null);

  useEffect(() => {
    setTarget(defaultTarget(instrument));
    setResult(null);
  }, [instrument]);

  function chooseFile(event) {
    const selected = event.target.files?.[0] || null;
    setResult(null);
    if (!selected) {
      setFile(null);
      return;
    }
    if (selected.size > MAX_MEDIA_BYTES) {
      setFile(null);
      setStatus('That file is larger than the 350 MB first-version limit.');
      event.target.value = '';
      return;
    }
    setFile(selected);
    if (!title) setTitle(selected.name.replace(/\.[^.]+$/, ''));
    setStatus(`${selected.name} is ready for private on-device analysis.`);
  }

  async function translate() {
    if (!file || busy) return;
    setBusy(true);
    setResult(null);
    setStatus('Listening for stable pitches, note changes, attacks, and tempo. Keep this tab open…');
    try {
      const rawSong = await transcribeMediaFile(file, { target, title, youtubeUrl });
      const song = adaptDraftToInstrument(rawSong, instrument);
      setResult(song);
      setStatus(`Draft complete: ${song.notes.length} playable notes at approximately ${song.bpm} BPM.`);
    } catch (error) {
      setStatus(error.message || 'The recording could not be analyzed.');
    } finally {
      setBusy(false);
    }
  }

  async function loadDraft() {
    if (!result) return;
    setBusy(true);
    try {
      await onReadyFile(result);
      setStatus(`Loaded ${result.title} into Polymath Musician. Press Play to audition the draft.`);
    } catch (error) {
      setStatus(error.message || 'The playable draft could not be loaded into this instrument.');
    } finally {
      setBusy(false);
    }
  }

  function downloadDraft() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${result.title.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '') || 'audio-draft'}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className="media-translation-panel" role="tabpanel">
      <div>
        <p className="eyebrow">Audio/video to playable draft</p>
        <h3>Let Polymath listen to your recording</h3>
        <p className="muted">
          This first version privately detects one dominant melody or bass line in your browser.
          Clear solo recordings work best; full-band recordings need review.
        </p>
      </div>

      <label className="upload-box compact media-file-picker">
        <input type="file" accept={MEDIA_ACCEPT} onChange={chooseFile} disabled={busy} />
        <span>{file ? file.name : 'Choose audio or video'}</span>
        <small>MP3 · WAV · M4A · MP4 · WebM · up to 350 MB</small>
      </label>

      <div className="media-analysis-grid">
        <label>
          Song title
          <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="My song" disabled={busy} />
        </label>
        <label>
          Listen for
          <select value={target} onChange={(event) => setTarget(event.target.value)} disabled={busy}>
            <option value="melody">Lead melody</option>
            <option value="bass">Bass line</option>
            <option value="drums">Rhythm / drum hits</option>
          </select>
        </label>
      </div>

      <label className="media-youtube-reference">
        YouTube reference link <span>(optional)</span>
        <input
          type="url"
          value={youtubeUrl}
          onChange={(event) => setYoutubeUrl(event.target.value)}
          placeholder="https://www.youtube.com/watch?v=…"
          disabled={busy}
        />
        <small>The link is saved for comparison only. Polymath does not download YouTube media.</small>
      </label>

      <label className="rights-confirmation">
        <input
          type="checkbox"
          checked={hasRights}
          onChange={(event) => setHasRights(event.target.checked)}
          disabled={busy}
        />
        <span>I own this recording or have permission to process it.</span>
      </label>

      <button
        type="button"
        className="primary full"
        disabled={!file || busy || !hasRights}
        onClick={translate}
      >
        {busy ? 'Listening and building draft…' : 'Create playable draft'}
      </button>

      <p className="form-status" aria-live="polite">{status}</p>

      {result && (
        <div className="media-result-card">
          <div>
            <strong>{result.title}</strong>
            <span>{result.notes.length} notes · {result.bpm} BPM · {result.duration.toFixed(1)} seconds</span>
          </div>
          <div className="media-confidence">
            <span>Note confidence <strong>{percent(result.transcription.noteConfidence)}</strong></span>
            <span>Tempo confidence <strong>{percent(result.transcription.tempoConfidence)}</strong></span>
          </div>
          <p>
            Confidence measures signal stability, not guaranteed musical correctness. Audition and edit dense band recordings.
          </p>
          <div className="media-result-actions">
            <button type="button" className="primary" onClick={loadDraft}>Load and play draft</button>
            <button type="button" className="ghost" onClick={downloadDraft}>Download JSON</button>
          </div>
        </div>
      )}
    </div>
  );
}
