import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest, downloadProtectedFile, fetchProtectedFile, fileToBase64 } from '../services/api.js';
import { instrumentLabel } from '../data/instruments.js';

const POLL_INTERVAL_MS = 10000;
const MAX_PDF_BYTES = 10 * 1024 * 1024;

function formatRemaining(seconds) {
  const safe = Math.max(0, Math.ceil(Number(seconds) || 0));
  const minutes = Math.floor(safe / 60);
  const remainder = safe % 60;
  if (minutes <= 0) return `${remainder}s`;
  return `${minutes}m ${String(remainder).padStart(2, '0')}s`;
}

function statusLabel(job) {
  if (!job) return 'Ready to translate';
  if (job.status === 'completed') return 'Ready to download';
  if (job.status === 'failed') return 'Translation failed';
  return job.stage || 'Translating music sheet';
}

async function verifyPdfFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Invalid PDF music sheet. Please upload a PDF file.');
  }
  if (!file.size || file.size > MAX_PDF_BYTES) {
    throw new Error('Invalid PDF music sheet. The PDF must be smaller than 10 MB.');
  }
  const signature = new Uint8Array(await file.slice(0, 5).arrayBuffer());
  const header = String.fromCharCode(...signature);
  if (header !== '%PDF-') {
    throw new Error('Invalid PDF music sheet. The selected file is not a valid PDF.');
  }
}

export default function PdfTranslationPanel({ user, setUser, instrument, onNavigate, onReadyFile }) {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('Upload a readable instrumental PDF music sheet.');
  const [busy, setBusy] = useState(false);
  const pollTimer = useRef(null);

  const allowance = user?.translationAllowance || null;
  const unlimited = Boolean(user?.admin || allowance?.unlimited);
  const remaining = unlimited ? null : Number(allowance?.remaining || 0);
  const planLabel = user?.pro ? 'Pro monthly translation' : 'Free monthly translation';
  const balanceAfter = Math.max(0, Number(user?.mcoins || 0) - 30);

  const progress = useMemo(() => {
    if (!job) return 0;
    if (job.status === 'completed') return 100;
    if (job.status === 'failed') return Math.max(5, Number(job.progress || 0));
    return Math.max(5, Math.min(95, Number(job.progress || 5)));
  }, [job]);

  function clearPolling() {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    pollTimer.current = null;
  }

  async function refreshJob(jobId) {
    try {
      const data = await apiRequest(`/api/score-translations/${jobId}`);
      setJob(data.job);
      if (data.user) setUser(data.user);
      if (data.job.status === 'completed') {
        setStatus('Your ready-to-play sheet is complete. Opening it in the piano studio...');
        clearPolling();
        await openReadySheet(data.job);
        return;
      }
      if (data.job.status === 'failed') {
        setStatus(data.job.error || 'Translation failed. Your payment or monthly attempt has been restored.');
        clearPolling();
        return;
      }
      pollTimer.current = window.setTimeout(() => refreshJob(jobId), POLL_INTERVAL_MS);
    } catch (error) {
      setStatus(error.message);
      pollTimer.current = window.setTimeout(() => refreshJob(jobId), POLL_INTERVAL_MS);
    }
  }

  useEffect(() => () => clearPolling(), []);

  async function choosePdf(event) {
    const selected = event.target.files?.[0] || null;
    clearPolling();
    setJob(null);
    if (!selected) {
      setFile(null);
      return;
    }
    try {
      await verifyPdfFile(selected);
      setFile(selected);
      setStatus(unlimited
        ? `${selected.name} is ready. Administrator access includes this translation.`
        : `${selected.name} is ready. Choose how to pay for this translation.`);
    } catch (error) {
      setFile(null);
      setStatus(error.message);
    } finally {
      event.target.value = '';
    }
  }

  async function startTranslation(paymentMethod) {
    if (!user) {
      onNavigate('account');
      return;
    }
    if (!file || busy) return;

    setBusy(true);
    setStatus('Validating and securely creating your translation job…');
    try {
      const contentBase64 = await fileToBase64(file);
      const data = await apiRequest('/api/score-translations', {
        method: 'POST',
        body: JSON.stringify({
          filename: file.name,
          instrument,
          paymentMethod,
          contentBase64,
        }),
      });
      setJob(data.job);
      setUser(data.user);
      setStatus('Translation started. The initial estimated time is approximately 20 minutes.');
      clearPolling();
      pollTimer.current = window.setTimeout(() => refreshJob(data.job.id), POLL_INTERVAL_MS);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function downloadResult() {
    if (!job?.id) return;
    try {
      await downloadProtectedFile(
        `/api/score-translations/${job.id}/download`,
        `${file?.name?.replace(/\.pdf$/i, '') || 'ready-to-play-sheet'}.json`,
      );
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function openReadySheet(completedJob = job) {
    if (!completedJob?.id || !onReadyFile) return;
    setBusy(true);
    try {
      const readyFile = await fetchProtectedFile(
        `/api/score-translations/${completedJob.id}/download`,
        `${file?.name?.replace(/\.pdf$/i, '') || 'ready-to-play-sheet'}.json`,
      );
      await onReadyFile(readyFile);
      setStatus('Loaded into the piano studio and ready to play.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  if (!user) {
    return (
      <div className="translation-panel">
        <div className="translation-heading">
          <div>
            <p className="eyebrow">PDF translation</p>
            <h3>Translate to a Ready-to-Play Sheet</h3>
          </div>
          <span className="price-chip">30 Mcoins · $3</span>
        </div>
        <p className="muted">Sign in to use your monthly translation allowance or pay with Mcoins.</p>
        <button className="primary full" type="button" onClick={() => onNavigate('account')}>Sign in to translate</button>
      </div>
    );
  }

  return (
    <div className="translation-panel">
      <div className="translation-heading">
        <div>
          <p className="eyebrow">PDF translation</p>
          <h3>Translate to a Ready-to-Play Sheet</h3>
        </div>
        <span className="price-chip">{unlimited ? 'Administrator - Unlimited' : '30 Mcoins · $3'}</span>
      </div>

      <p className="muted">
        Upload a readable {instrumentLabel(instrument)} PDF music sheet. Non-music documents, corrupted PDFs, and unsupported sheets are rejected without charging you.
      </p>

      <label className="upload-box compact translation-file-picker">
        <input type="file" accept="application/pdf,.pdf" onChange={choosePdf} disabled={busy || Boolean(job && job.status === 'processing')} />
        <span>{file ? file.name : 'Choose PDF music sheet'}</span>
        <small>PDF only · maximum 10 MB</small>
      </label>

      <div className="allowance-summary">
        <div>
          <span>{unlimited ? 'Administrator access' : user.pro ? 'Pro allowance' : 'Free allowance'}</span>
          <strong>{unlimited ? 'Unlimited PDF translations' : `${remaining} of ${allowance?.limit ?? (user.pro ? 20 : 1)} remaining`}</strong>
        </div>
        <div>
          <span>{unlimited ? 'Translation cost' : 'Mcoin balance'}</span>
          <strong>{unlimited ? 'No charge' : `${Number(user.mcoins || 0).toLocaleString()} Mcoins`}</strong>
        </div>
      </div>

      {!job && (
        <div className="translation-payment-grid">
          {unlimited ? (
            <button
              className="primary"
              type="button"
              onClick={() => startTranslation('admin')}
              disabled={!file || busy}
            >
              <span>Translate with admin access</span>
              <small>Unlimited - no Mcoins charged</small>
            </button>
          ) : (
            <>
          <button
            className="primary"
            type="button"
            onClick={() => startTranslation('allowance')}
            disabled={!file || busy || remaining <= 0}
          >
            <span>{planLabel}</span>
            <small>{remaining > 0 ? `Use 1 · ${remaining} remaining` : '0 remaining · unavailable'}</small>
          </button>
          <button
            className="ghost mcoin-pay-button"
            type="button"
            onClick={() => startTranslation('mcoins')}
            disabled={!file || busy || Number(user.mcoins || 0) < 30}
          >
            <span>Pay 30 Mcoins</span>
            <small>{Number(user.mcoins || 0) >= 30 ? `${balanceAfter} Mcoins after payment` : 'Insufficient Mcoins'}</small>
          </button>
            </>
          )}
        </div>
      )}

      {!unlimited && remaining <= 0 && !job && (
        <div className="quota-warning">
          <strong>0 translations remaining.</strong>
          <span>Pay 30 Mcoins{user.pro ? ' to continue.' : ' or buy Pro for 20 monthly translations.'}</span>
          {!user.pro && <button className="ghost" type="button" onClick={() => onNavigate('payment', { productId: 'polymath-pro' })}>Buy Pro</button>}
        </div>
      )}

      {job && (
        <div className={`translation-job ${job.status}`}>
          <div className="job-status-row">
            <div>
              <span>{statusLabel(job)}</span>
              <strong>{job.filename}</strong>
            </div>
            <span className="job-state-badge">{job.status}</span>
          </div>
          <div className="job-progress-track" aria-label={`Translation progress ${progress}%`}>
            <span style={{ width: `${progress}%` }} />
          </div>
          <div className="job-metrics">
            <span>Estimated time remaining</span>
            <strong>{job.status === 'processing' ? formatRemaining(job.estimatedRemainingSeconds) : job.status === 'completed' ? 'Ready' : 'Stopped'}</strong>
          </div>
          {Number(job.estimateExtensionCount || 0) > 0 && job.status === 'processing' && (
            <p className="estimate-note">Processing is taking longer than expected. The estimate has been extended by {Number(job.estimateExtensionCount) * 5} minutes.</p>
          )}
          <p className="muted job-payment-line">
            Payment method: {job.paymentMethod === 'admin' ? 'unlimited administrator access' : job.paymentMethod === 'mcoins' ? '30 Mcoins' : 'monthly translation allowance'}.
          </p>
          {job.status === 'completed' && <button className="primary full" type="button" onClick={downloadResult}>Download Ready-to-Play Sheet</button>}
          {job.status === 'failed' && <p className="form-status">{job.error || 'The translation could not be completed. Your payment or allowance was restored.'}</p>}
        </div>
      )}

      <p className="form-status">{status}</p>
      <p className="translation-footnote">The estimate begins at about 20 minutes. If needed, it automatically extends in five-minute blocks until the job completes or fails.</p>
    </div>
  );
}
