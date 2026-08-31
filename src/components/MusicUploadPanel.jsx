import { useState } from 'react';
import { apiRequest } from '../services/api.js';
import PdfTranslationPanel from './PdfTranslationPanel.jsx';
import MediaTranscriptionPanel from './MediaTranscriptionPanel.jsx';

const GUEST_READY_UPLOAD_KEY = 'polymath_guest_ready_upload_month_v2';
const GUEST_READY_UPLOAD_LIMIT = 2;

function currentGuestPeriod(now = new Date()) {
  return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}`;
}

function readGuestAllowance() {
  const period = currentGuestPeriod();
  if (typeof window === 'undefined') return { period, used: 0, remaining: GUEST_READY_UPLOAD_LIMIT };
  try {
    const stored = JSON.parse(window.localStorage.getItem(GUEST_READY_UPLOAD_KEY) || '{}');
    const used = stored.period === period ? Math.max(0, Number(stored.used || 0)) : 0;
    return { period, used, remaining: Math.max(0, GUEST_READY_UPLOAD_LIMIT - used) };
  } catch {
    return { period, used: 0, remaining: GUEST_READY_UPLOAD_LIMIT };
  }
}

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
  const [guestAllowance, setGuestAllowance] = useState(readGuestAllowance);
  const [showUploadAllowance, setShowUploadAllowance] = useState(false);

  async function handleReadyFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setMode('ready');
    setBusy(true);
    setStatus(`Reading ${file.name}…`);
    try {
      const prepared = await onReadyFile(file, { commit: false });
      let payment;
      if (user) {
        payment = await apiRequest('/api/ready-sheet-uploads', {
          method: 'POST',
          body: JSON.stringify({ filename: file.name }),
        });
        if (payment.user) setUser?.(payment.user);
      } else {
        const current = readGuestAllowance();
        if (current.remaining <= 0) {
          onNavigate?.('signin');
          throw new Error('Your 2 free guest uploads are used for this month. Sign in to continue for 0.5 Mcoin per upload.');
        }
        const next = {
          period: current.period,
          used: current.used + 1,
          remaining: current.remaining - 1,
        };
        window.localStorage.setItem(GUEST_READY_UPLOAD_KEY, JSON.stringify({ period: next.period, used: next.used }));
        setGuestAllowance(next);
        payment = { costMcoins: 0, paymentMethod: 'free_attempt' };
      }
      const result = await onReadyFile(file, { commit: true, prepared });
      const title = result?.title || file.name;
      const updatedAllowance = payment.user?.readySheetAllowance;
      const remaining = payment.user
        ? updatedAllowance?.remaining
        : readGuestAllowance().remaining;
      const resultLabel = updatedAllowance?.unlimited || payment.paymentMethod === 'unlimited'
        ? '∞'
        : payment.costMcoins > 0
          ? `${payment.costMcoins} Mcoin`
          : `${remaining} free left`;
      setStatus(`Loaded ${title} · ${resultLabel}`);
    } catch (error) {
      setStatus(error.message || 'The ready-to-play sheet could not be loaded.');
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  }

  const readyAllowance = user?.readySheetAllowance;
  const unlimitedUploads = Boolean(user && (user.admin || readyAllowance?.unlimited));
  const remainingUploads = user
    ? Math.max(0, Number(readyAllowance?.remaining ?? 2))
    : guestAllowance.remaining;

  const modeLabels = {
    ready: 'Ready-to-play sheet',
    pdf: 'PDF music sheet',
    media: 'Audio or music video',
  };

  function chooseAnotherType() {
    if (busy) return;
    setMode(null);
    setStatus('');
  }

  return (
    <div className={`music-upload-panel ${compact ? 'compact-panel' : ''}`}>
      {mode === null ? (
        <>
          <p className="upload-choice-prompt">What are you uploading?</p>
          <div className="upload-mode-grid" aria-label="Choose what to upload">
            <label className="upload-mode-option" onClick={() => setShowUploadAllowance(true)}>
              <input type="file" accept={readyAccept} onChange={handleReadyFile} disabled={busy} />
              <strong>Ready-to-play sheet</strong>
              <small>
                JSON or MIDI
                {showUploadAllowance && (
                  unlimitedUploads
                    ? ' · unlimited'
                    : remainingUploads > 0
                      ? ` · ${remainingUploads} free`
                      : user
                        ? ' · 0.5 Mcoin'
                        : ' · sign in'
                )}
              </small>
            </label>
            <button type="button" onClick={() => setMode('pdf')}>
              <strong>PDF music sheet</strong>
              <small>Turn a PDF into playable notes</small>
            </button>
            <button type="button" onClick={() => setMode('media')}>
              <strong>Audio or music video</strong>
              <small>Create a playable transcription</small>
            </button>
          </div>
        </>
      ) : (
        <div className="upload-active-mode">
          <div>
            <small>Upload type</small>
            <strong>{modeLabels[mode]}</strong>
          </div>
          <button className="ghost" type="button" onClick={chooseAnotherType} disabled={busy}>
            Choose another type
          </button>
        </div>
      )}

      {mode === 'pdf' ? (
        <PdfTranslationPanel user={user} setUser={setUser} instrument={instrument} onNavigate={onNavigate} onReadyFile={onReadyFile} />
      ) : mode === 'media' ? (
        <MediaTranscriptionPanel
          user={user}
          setUser={setUser}
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
