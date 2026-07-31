import { useState } from 'react';
import PdfTranslationPanel from './PdfTranslationPanel.jsx';
import MediaTranslationPanel from './MediaTranslationPanel.jsx';

export default function MusicUploadPanel({
  user,
  setUser,
  onNavigate,
  instrument,
  onReadyFile,
  readyAccept = '.json,.csv,.mid,.midi,.musicxml,.xml',
  readyFormats = 'JSON · CSV · MIDI · MusicXML',
  readyDescription = 'A ready-to-play sheet can be read and played immediately by Polymath Musician.',
  compact = false,
  enableMedia = false,
}) {
  const [mode, setMode] = useState('ready');
  const [status, setStatus] = useState('Choose a ready-to-play sheet or translate a PDF music sheet.');
  const [busy, setBusy] = useState(false);

  async function handleReadyFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
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

  function handleMediaDraft(song) {
    const result = onReadyFile(song);
    return result || song;
  }

  return (
    <div className={`music-upload-panel ${compact ? 'compact-panel' : ''}`}>
      <div className="upload-mode-grid" role="tablist" aria-label="Music sheet upload options">
        <button
          type="button"
          className={mode === 'ready' ? 'active' : ''}
          onClick={() => setMode('ready')}
          role="tab"
          aria-selected={mode === 'ready'}
        >
          <strong>Upload Ready-to-Play Sheet</strong>
          <small>Play JSON, MIDI, and supported files now</small>
        </button>
        <button
          type="button"
          className={mode === 'pdf' ? 'active' : ''}
          onClick={() => setMode('pdf')}
          role="tab"
          aria-selected={mode === 'pdf'}
        >
          <strong>Translate to a Ready-to-Play Sheet</strong>
          <small>Convert an instrumental PDF music sheet</small>
        </button>
        {enableMedia && (
          <button
            type="button"
            className={mode === 'media' ? 'active' : ''}
            onClick={() => setMode('media')}
            role="tab"
            aria-selected={mode === 'media'}
          >
            <strong>Listen to Audio or Video</strong>
            <small>Create a playable draft from your recording</small>
          </button>
        )}
      </div>

      {mode === 'ready' ? (
        <div className="ready-upload-section" role="tabpanel">
          <div>
            <p className="eyebrow">Ready-to-play upload</p>
            <h3>Upload Ready-to-Play Sheet</h3>
            <p className="muted">{readyDescription}</p>
          </div>
          <label className="upload-box compact">
            <input type="file" accept={readyAccept} onChange={handleReadyFile} disabled={busy} />
            <span>{busy ? 'Reading sheet…' : 'Choose ready-to-play sheet'}</span>
            <small>{readyFormats}</small>
          </label>
          <div className="ready-sheet-help">
            <strong>What is a ready-to-play sheet?</strong>
            <p>It is the playable version of a music sheet. The platform uses formats such as JSON or MIDI internally, but you do not need to understand the code.</p>
          </div>
          <p className="form-status">{status}</p>
        </div>
      ) : mode === 'pdf' ? (
        <PdfTranslationPanel user={user} setUser={setUser} instrument={instrument} onNavigate={onNavigate} />
      ) : enableMedia ? (
        <MediaTranslationPanel instrument={instrument} onReadyFile={handleMediaDraft} />
      ) : (
        <PdfTranslationPanel user={user} setUser={setUser} instrument={instrument} onNavigate={onNavigate} />
      )}
    </div>
  );
}
