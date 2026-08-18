import { useState } from 'react';
import PdfTranslationPanel from './PdfTranslationPanel.jsx';
import MediaTranscriptionPanel from './MediaTranscriptionPanel.jsx';

export default function MusicUploadPanel({
  user,
  setUser,
  onNavigate,
  instrument,
  onReadyFile,
  readyAccept = '.json,.mid,.midi',
  compact = false,
}) {
  const [mode, setMode] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  async function handleReadyFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setMode('ready');
    setBusy(true);
    setStatus(`Reading ${file.name}…`);
    try {
      const result = await onReadyFile(file);
      const title = result?.title || file.name;
      const count = result?.notes?.length ?? result?.events?.length;
      setStatus(`Loaded ${title}${Number.isFinite(count) ? ` · ${count} playable events` : ''}.`);
    } catch (error) {
      setStatus(error.message || 'The ready-to-play sheet could not be loaded.');
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  }

  return (
    <div className={`music-upload-panel ${compact ? 'compact-panel' : ''}`}>
      <div className="upload-mode-grid" aria-label="Music sheet upload options">
        <label className={`upload-mode-option ${mode === 'ready' ? 'active' : ''}`}>
          <input type="file" accept={readyAccept} onChange={handleReadyFile} disabled={busy} />
          <strong>{busy ? 'Loading…' : 'Upload Ready to Play Sheet (JSON/MIDI)'}</strong>
        </label>
        <button
          type="button"
          className={mode === 'pdf' ? 'active' : ''}
          onClick={() => setMode('pdf')}
        >
          <strong>Translate to Ready to Play Sheet (PDF)</strong>
        </button>
        <button
          type="button"
          className={mode === 'media' ? 'active' : ''}
          onClick={() => setMode('media')}
        >
          <strong>Transcribe Music Audio/Video (MuScriptor)</strong>
        </button>
      </div>

      {mode === 'pdf' ? (
        <PdfTranslationPanel user={user} setUser={setUser} instrument={instrument} onNavigate={onNavigate} />
      ) : mode === 'media' ? (
        <MediaTranscriptionPanel
          user={user}
          onNavigate={onNavigate}
          instrument={instrument}
          onReadyFile={onReadyFile}
        />
      ) : (
        status && <p className="form-status">{status}</p>
      )}
    </div>
  );
}
