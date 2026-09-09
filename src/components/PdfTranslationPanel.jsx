import { useEffect, useMemo, useRef, useState } from 'react';
import {
  apiRequest,
  downloadProtectedFile,
  fetchProtectedFile,
  fileToBase64,
  uploadProtectedArtifact,
} from '../services/api.js';
import { instrumentLabel } from '../data/instruments.js';
import { userFacingError } from '../utils/userFacingError.js';
import TaskProgress from './TaskProgress.jsx';

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
  return 'Creating your playable sheet';
}

function progressDetail(job) {
  const remaining = Number(job?.estimatedRemainingSeconds);
  return Number.isFinite(remaining) && remaining > 0
    ? `About ${formatRemaining(remaining)} remaining · You may leave this page`
    : 'You may leave this page while we finish.';
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

export default function PdfTranslationPanel({ user, setUser, instrument, onNavigate, onReadyFile, onPersonalSongSaved }) {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('Upload a readable instrumental PDF music sheet.');
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const pollTimer = useRef(null);

  const allowance = user?.translationAllowance || null;
  const unlimited = Boolean(user?.admin || allowance?.unlimited);
  const remaining = unlimited ? null : Number(allowance?.remaining || 0);
  const overageCost = Number(allowance?.overageCostMcoins ?? (user?.pro ? 0.5 : 2));
  const planLabel = user?.subscriptionTier
    ? `${user.subscriptionTier === 'musician' ? 'Musician' : 'Chill'} included translation`
    : 'Included translation';
  const balanceAfter = Math.max(0, Number(user?.mcoins || 0) - overageCost);

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
        setStatus('We couldn’t translate this sheet. Your payment or monthly attempt has been restored.');
        clearPolling();
        return;
      }
      pollTimer.current = window.setTimeout(() => refreshJob(jobId), POLL_INTERVAL_MS);
    } catch {
      setStatus('');
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
    setUploadProgress(null);
    setStatus('');
    try {
      const directUpload = await uploadProtectedArtifact(file, 'score-translation', {
        onProgress: setUploadProgress,
      });
      const contentBase64 = directUpload ? '' : await fileToBase64(file);
      const data = await apiRequest('/api/score-translations', {
        method: 'POST',
        body: JSON.stringify({
          filename: file.name,
          instrument,
          paymentMethod,
          contentBase64,
          uploadReceipt: directUpload?.receipt || '',
        }),
      });
      setJob(data.job);
      setUser(data.user);
      setStatus('');
      setUploadProgress(null);
      clearPolling();
      pollTimer.current = window.setTimeout(() => refreshJob(data.job.id), POLL_INTERVAL_MS);
    } catch (error) {
      setStatus(userFacingError(error, 'We couldn’t start this translation. Please try again.'));
      setUploadProgress(null);
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
      setStatus(userFacingError(error, 'The finished sheet could not be downloaded. Please try again.'));
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
      if (completedJob.personalSongId) {
        onPersonalSongSaved?.({
          id: completedJob.personalSongId,
          title: String(completedJob.outputFilename || completedJob.filename || 'Ready-to-play song').replace(/\.json$/i, ''),
          artist: '',
          instrument,
          format: 'JSON',
          filename: completedJob.outputFilename || readyFile.name,
          createdAt: completedJob.completedAt,
        });
      }
      setStatus('Loaded into the piano studio and ready to play.');
    } catch (error) {
      setStatus(userFacingError(error, 'The finished sheet could not be opened. Please try again.'));
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
          <span className="price-chip">2 Mcoins · $2</span>
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
        <span className="price-chip">{unlimited ? 'Administrator - Unlimited' : `${overageCost} Mcoins · $${overageCost}`}</span>
      </div>

      <p className="muted">
        Upload a readable {instrumentLabel(instrument)} PDF music sheet. Polymath reads it locally without a paid translation API. Unreadable sheets are rejected instead of guessing.
      </p>

      <label className="upload-box compact translation-file-picker">
        <input type="file" accept="application/pdf,.pdf" onChange={choosePdf} disabled={busy || Boolean(job && job.status === 'processing')} />
        <span>{file ? file.name : 'Choose PDF music sheet'}</span>
        <small>PDF only · maximum 10 MB and 20 pages</small>
      </label>

      <div className="allowance-summary">
        <div>
          <span>{unlimited ? 'Administrator access' : user.subscriptionTier ? `${user.subscriptionTier === 'musician' ? 'Musician' : 'Chill'} allowance` : 'No subscription'}</span>
          <strong>{unlimited ? 'Unlimited translations' : `${remaining} of ${allowance?.limit ?? 0} remaining`}</strong>
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
            disabled={!file || busy || Number(user.mcoins || 0) < overageCost}
          >
            <span>Pay {overageCost} Mcoins</span>
            <small>{Number(user.mcoins || 0) >= overageCost ? `${balanceAfter} Mcoins after payment` : 'Insufficient Mcoins'}</small>
          </button>
            </>
          )}
        </div>
      )}

      {!unlimited && remaining <= 0 && !job && (
        <div className="quota-warning">
          <strong>0 translations remaining.</strong>
          <span>Pay {overageCost} Mcoins{user.pro ? ' to continue.' : ' or choose a subscription for monthly translations.'}</span>
          {!user.pro && <button className="ghost" type="button" onClick={() => onNavigate('payment', { productId: 'polymath-chill-monthly' })}>See subscriptions</button>}
        </div>
      )}

      {busy && !job && (
        <TaskProgress
          compact
          label='Uploading your music sheet…'
          progress={uploadProgress}
          detail='Keep this page open until the upload finishes.'
          ariaLabel='Music sheet upload progress'
        />
      )}

      {job && (
        <div className={`translation-job ${job.status}`}>
          {job.status === 'processing' ? (
            <TaskProgress
              label='Creating your playable sheet…'
              progress={progress}
              detail={progressDetail(job)}
              ariaLabel='Music sheet translation progress'
            />
          ) : (
            <div className="job-status-row">
              <div>
                <span>{statusLabel(job)}</span>
                <strong>{job.filename}</strong>
              </div>
              <span className="job-state-badge">{job.status === 'completed' ? 'Ready' : 'Stopped'}</span>
            </div>
          )}
          {job.status === 'completed' && (
            <div className="job-metrics">
              <span>Notation confidence</span>
              <strong>{Math.round(Number(job.confidence || 0) * 100)}% notation confidence</strong>
            </div>
          )}
          {job.status === 'completed' && job.pianoPerformance && (
            <div className="job-metrics">
              <span>Piano interpretation</span>
              <strong>
                {job.pianoPerformance.pedalSource === 'printed-score'
                  || job.pianoPerformance.pedalSource === 'printed-smufl-pedal'
                  ? 'Printed pedal'
                  : job.pianoPerformance.pedalSource === 'inferred-score-pedaling'
                    ? 'Musical pedal inferred'
                    : 'No pedal'}
                {' · '}{Number(job.pianoPerformance.restrikesGivenReleaseGap || 0)} clean restrikes
              </strong>
            </div>
          )}
          {job.status === 'completed' && Array.isArray(job.warnings) && job.warnings.length > 0 && (
            <details className="advanced-controls">
              <summary>Review {job.warnings.length} notation warning{job.warnings.length === 1 ? '' : 's'}</summary>
              {job.warnings.map((warning) => <p className="muted" key={warning}>{warning}</p>)}
            </details>
          )}
          {Number(job.estimateExtensionCount || 0) > 0 && job.status === 'processing' && (
            <p className="estimate-note">This is taking longer than the first estimate. The updated time appears above.</p>
          )}
          <p className="muted job-payment-line">
            Payment method: {job.paymentMethod === 'admin' ? 'unlimited administrator access' : job.paymentMethod === 'mcoins' ? `${job.costMcoins} Mcoins` : 'monthly translation allowance'}.
          </p>
          {job.status === 'completed' && <button className="primary full" type="button" onClick={downloadResult}>Download Ready-to-Play Sheet</button>}
          {job.status === 'failed' && <p className="form-status">The translation could not be completed. Your payment or allowance was restored.</p>}
        </div>
      )}

      {status && !busy && job?.status !== 'processing' && <p className="form-status">{status}</p>}
      <p className="translation-footnote">The estimate begins at about 5 minutes. Large or scanned scores can extend in five-minute blocks until the job completes or fails.</p>
    </div>
  );
}
