import { useEffect, useRef, useState } from 'react';
import {
  apiRequest,
  downloadProtectedFile,
  fetchProtectedFile,
  uploadProtectedArtifact,
} from '../services/api.js';
import { trackProductEvent, uploadSizeBucket } from '../services/productAnalytics.js';

const MEDIA_ACCEPT = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.mp4,.mov,.webm,.mkv,.avi';
const ACTIVE_JOB_KEY_PREFIX = 'polymath-active-media-transcription-v1:';

function activeJobKey(userId) {
  return `${ACTIVE_JOB_KEY_PREFIX}${String(userId || 'guest')}`;
}

function rememberActiveJob(userId, jobId) {
  try {
    if (jobId) window.localStorage.setItem(activeJobKey(userId), jobId);
    else window.localStorage.removeItem(activeJobKey(userId));
  } catch {
    // Server-side job history still restores progress when storage is blocked.
  }
}

function recalledActiveJob(userId) {
  try {
    return window.localStorage.getItem(activeJobKey(userId)) || '';
  } catch {
    return '';
  }
}

function elapsedLabel(startedAt, now = Date.now()) {
  const started = Date.parse(startedAt || '');
  if (!Number.isFinite(started)) return '';
  const seconds = Math.max(0, Math.floor((now - started) / 1000));
  if (seconds < 60) return `${seconds}s elapsed`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s elapsed`;
}

function polymathLabel(value) {
  return String(value || '').replace(/MuScriptor/gi, 'Polymath');
}

export default function MediaTranscriptionPanel({
  user,
  setUser,
  onNavigate,
  instrument,
  onReadyFile,
  onPersonalSongSaved,
}) {
  const [capability, setCapability] = useState(null);
  const [file, setFile] = useState(null);
  const [playbackMode, setPlaybackMode] = useState(instrument === 'piano' ? 'full' : 'instrumental');
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [restoredJob, setRestoredJob] = useState(false);
  const [clock, setClock] = useState(() => Date.now());
  const pollTimer = useRef(null);
  const pollFailures = useRef(0);
  const mounted = useRef(true);
  const autoOpenJobId = useRef('');
  const currentUserId = useRef(String(user?.user_id || ''));
  currentUserId.current = String(user?.user_id || '');
  const transcriptionUnavailable = capability?.enabled === false
    || Boolean(capability?.adminOnly && !user?.admin);
  const allowance = user?.translationAllowance;
  const remaining = allowance?.unlimited ? null : Math.max(0, Number(allowance?.remaining || 0));
  const paymentMethod = user?.admin ? 'admin' : remaining > 0 ? 'allowance' : 'mcoins';
  const overageCost = Number(allowance?.overageCostMcoins ?? (user?.pro ? 0.5 : 2));
  const insufficientMcoins = paymentMethod === 'mcoins'
    && Number(user?.mcoins || 0) < overageCost;

  function clearPolling() {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    pollTimer.current = null;
  }

  function schedulePoll(jobId, delay = 4000) {
    clearPolling();
    pollTimer.current = window.setTimeout(() => refreshJob(jobId), delay);
  }

  useEffect(() => {
    let cancelled = false;
    mounted.current = true;
    const capabilityRequest = apiRequest('/api/media-transcriptions/capabilities');
    const historyRequest = user?.user_id
      ? apiRequest('/api/media-transcriptions')
      : Promise.resolve(null);
    Promise.all([capabilityRequest, historyRequest])
      .then(([capabilityData, historyData]) => {
        if (cancelled) return;
        setCapability(capabilityData);
        if (historyData?.user && setUser) setUser(historyData.user);
        const rememberedId = recalledActiveJob(user?.user_id);
        const jobs = Array.isArray(historyData?.jobs) ? historyData.jobs : [];
        const active = jobs.find((candidate) => candidate.id === rememberedId && candidate.status === 'processing')
          || jobs.find((candidate) => candidate.status === 'processing');
        if (!active) return;
        setJob(active);
        setRestoredJob(true);
        autoOpenJobId.current = active.id;
        rememberActiveJob(user?.user_id, active.id);
        setStatus('Reconnected to your transcription. It kept working safely on the server.');
        trackProductEvent('transcription_restored', {
          instrument: active.instrument || instrument || 'band',
          playbackMode: active.playbackMode || playbackMode,
          restored: true,
        });
        schedulePoll(active.id, 250);
      })
      .catch((error) => {
        if (!cancelled) setStatus(error.message);
      });

    const resumePolling = () => {
      if (!autoOpenJobId.current || document.visibilityState === 'hidden') return;
      pollFailures.current = 0;
      schedulePoll(autoOpenJobId.current, 50);
    };
    window.addEventListener('online', resumePolling);
    document.addEventListener('visibilitychange', resumePolling);
    return () => {
      cancelled = true;
      mounted.current = false;
      clearPolling();
      window.removeEventListener('online', resumePolling);
      document.removeEventListener('visibilitychange', resumePolling);
    };
  }, [user?.user_id]);

  useEffect(() => {
    if (job?.status !== 'processing') return undefined;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [job?.status]);

  async function refreshJob(jobId) {
    const requestUserId = String(user?.user_id || '');
    try {
      const data = await apiRequest(`/api/media-transcriptions/${jobId}`);
      if (!mounted.current || currentUserId.current !== requestUserId) return;
      pollFailures.current = 0;
      setJob(data.job);
      if (data.user && setUser) setUser(data.user);
      if (data.job.status === 'completed') {
        setStatus('Your ready-to-play sheet is ready. Opening the piano studio...');
        clearPolling();
        rememberActiveJob(user?.user_id, '');
        if (autoOpenJobId.current === data.job.id) await openReadySheet(data.job);
        autoOpenJobId.current = '';
        return;
      }
      if (data.job.status === 'failed') {
        const refund = data.job.refunded ? ' Your translation was refunded.' : '';
        setStatus(`${polymathLabel(data.job.error) || 'Polymath could not transcribe this recording.'}${refund}`);
        clearPolling();
        autoOpenJobId.current = '';
        rememberActiveJob(user?.user_id, '');
        return;
      }
      schedulePoll(jobId, document.visibilityState === 'hidden' ? 15000 : 4000);
    } catch {
      if (!mounted.current || currentUserId.current !== requestUserId) return;
      pollFailures.current += 1;
      const retrySeconds = Math.min(30, 4 * (2 ** Math.min(3, pollFailures.current)));
      setStatus(`The status connection paused, but your server job is safe. Reconnecting in ${retrySeconds} seconds…`);
      schedulePoll(jobId, retrySeconds * 1000);
    }
  }

  function chooseFile(event) {
    const selected = event.target.files?.[0] || null;
    setJob(null);
    setRestoredJob(false);
    setStatus('');
    setRightsConfirmed(false);
    setFile(selected);
    if (selected) {
      trackProductEvent('transcription_file_selected', {
        instrument: instrument || 'band',
        sizeBucket: uploadSizeBucket(selected.size),
        sourceKind: selected.type.startsWith('video/') ? 'video' : 'audio',
      });
    }
    event.target.value = '';
  }

  async function startTranscription() {
    if (!user) {
      onNavigate('account');
      return;
    }
    if (!file || busy || !rightsConfirmed) return;
    setBusy(true);
    setStatus('Uploading securely…');
    try {
      const directUpload = await uploadProtectedArtifact(file, 'media-transcription', {
        onProgress: (percent) => setStatus(`Uploading securely… ${percent}%`),
      });
      let data;
      if (directUpload) {
        data = await apiRequest('/api/media-transcriptions/direct', {
          method: 'POST',
          body: JSON.stringify({
            uploadReceipt: directUpload.receipt,
            instrument: instrument || 'band',
            title: file.name.replace(/\.[^.]+$/, ''),
            playbackMode,
            rightsConfirmed: 'true',
            paymentMethod,
          }),
        });
      } else {
        const form = new FormData();
        form.append('media', file, file.name);
        form.append('instrument', instrument || 'band');
        form.append('title', file.name.replace(/\.[^.]+$/, ''));
        form.append('playbackMode', playbackMode);
        form.append('rightsConfirmed', 'true');
        form.append('paymentMethod', paymentMethod);
        data = await apiRequest('/api/media-transcriptions', {
          method: 'POST',
          body: form,
        });
      }
      setCapability(data.capability);
      if (data.user && setUser) setUser(data.user);
      setJob(data.job);
      setRestoredJob(false);
      setStatus('Polymath is preparing your recording.');
      autoOpenJobId.current = data.job.id;
      rememberActiveJob(user?.user_id, data.job.id);
      pollFailures.current = 0;
      clearPolling();
      schedulePoll(data.job.id, 1500);
    } catch (error) {
      setStatus(error.message);
      if (error.details?.capability) setCapability(error.details.capability);
    } finally {
      setBusy(false);
    }
  }

  function resetFailedJob() {
    clearPolling();
    setJob(null);
    setRestoredJob(false);
    setStatus(file
      ? 'Ready to retry this file. Nothing will be charged until a new job is accepted.'
      : 'Choose the recording again to retry.');
  }

  async function submitFeedback(feedback) {
    if (!job?.id || job.feedback || feedbackBusy) return;
    setFeedbackBusy(true);
    try {
      const data = await apiRequest(`/api/media-transcriptions/${job.id}/feedback`, {
        method: 'POST',
        body: JSON.stringify({ feedback }),
      });
      setJob(data.job);
      setStatus('Thank you. This review goes directly into Polymath quality tracking.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setFeedbackBusy(false);
    }
  }

  async function openReadySheet(completedJob = null) {
    const readyJob = completedJob?.id ? completedJob : job;
    if (!readyJob?.id || !onReadyFile) return;
    setBusy(true);
    try {
      const readyFile = await fetchProtectedFile(
        `/api/media-transcriptions/${readyJob.id}/download`,
        readyJob.outputFilename || 'polymath-ready-to-play.json',
      );
      await onReadyFile(readyFile);
      if (readyJob.personalSongId) {
        onPersonalSongSaved?.({
          id: readyJob.personalSongId,
          title: readyJob.title || String(readyJob.outputFilename || 'Ready-to-play song').replace(/\.json$/i, ''),
          artist: '',
          instrument: readyJob.instrument || instrument,
          format: 'JSON',
          filename: readyJob.outputFilename || readyFile.name,
          createdAt: readyJob.completedAt,
        });
      }
      setStatus('Loaded into the studio and ready to play.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  if (!user) {
    return (
      <div className="media-transcription-panel">
        <p>Sign in to turn an MP3, audio file, or music video into a ready-to-play sheet.</p>
        <button className="primary full" type="button" onClick={() => onNavigate('account')}>
          Sign in to transcribe
        </button>
      </div>
    );
  }

  return (
    <div className="media-transcription-panel">
      {capability && !capability.enabled && (
        <div className="quota-warning">
          <strong>Polymath transcription is not enabled on this server.</strong>
          <span>{polymathLabel(capability.reason)}</span>
        </div>
      )}

      {capability?.enabled && capability.adminOnly && !user.admin && (
        <div className={'quota-warning'}>
          <strong>Polymath model testing is restricted.</strong>
          <span>Administrator access is required during the trial-and-error testing phase.</span>
        </div>
      )}

      {user.admin && (
        <p className="muted"><strong>Administrator access:</strong> unlimited audio and video transcriptions.</p>
      )}
      {!user.admin && (
        <p className="muted">
          {remaining > 0
            ? `${remaining} included translation${remaining === 1 ? '' : 's'} remaining this month.`
            : `This translation costs ${overageCost} Mcoin${overageCost === 1 ? '' : 's'}.`}
        </p>
      )}
      {insufficientMcoins && (
        <div className="quota-warning">
          <strong>Not enough Mcoins.</strong>
          <span>You need {overageCost} Mcoins for this translation.</span>
          <button className="ghost" type="button" onClick={() => onNavigate('payment', { productId: 'mcoins-50' })}>Open wallet options</button>
        </div>
      )}

      <label className="upload-box compact">
        <input type="file" accept={MEDIA_ACCEPT} onChange={chooseFile} disabled={busy || capability?.enabled === false} />
        <span>{file ? file.name : 'Choose MP3, audio, or music video'}</span>
        <small>No file-size limit · first 10 minutes processed</small>
      </label>

      {file && (
        <label className="media-rights-check">
          <input
            type="checkbox"
            checked={rightsConfirmed}
            onChange={(event) => setRightsConfirmed(event.target.checked)}
          />
          <span>I have permission to transcribe this recording.</span>
        </label>
      )}

      {file && rightsConfirmed && instrument === 'piano' && (
        <fieldset className={'media-playback-options'}>
          <legend>Playback version</legend>
          <label className={'media-playback-option ' + (playbackMode === 'full' ? 'active' : '')}>
            <input
              type={'radio'}
              name={'media-playback-mode'}
              value={'full'}
              checked={playbackMode === 'full'}
              onChange={() => setPlaybackMode('full')}
              disabled={busy}
            />
            <span>
              <strong>Full song</strong>
              <small>All detected parts plus the singer's melody, emphasized on piano.</small>
            </span>
          </label>
          <label className={'media-playback-option ' + (playbackMode === 'instrumental' ? 'active' : '')}>
            <input
              type={'radio'}
              name={'media-playback-mode'}
              value={'instrumental'}
              checked={playbackMode === 'instrumental'}
              onChange={() => setPlaybackMode('instrumental')}
              disabled={busy}
            />
            <span>
              <strong>Pure instrumental</strong>
              <small>All detected instrument parts, with the vocal melody excluded.</small>
            </span>
          </label>
        </fieldset>
      )}

      {!job && file && rightsConfirmed && (
        <button
          className="primary full"
          type="button"
          onClick={startTranscription}
          disabled={!file || !rightsConfirmed || busy || transcriptionUnavailable || insufficientMcoins}
        >
          {busy
            ? 'Uploading…'
            : paymentMethod === 'allowance'
              ? 'Use 1 included translation'
              : paymentMethod === 'admin'
                ? 'Transcribe with administrator access'
                : `Transcribe for ${overageCost} Mcoins`}
        </button>
      )}

      {job && (
        <div className={`media-transcription-job ${job.status}`}>
          <div className="job-status-row">
            <div>
              <span>{polymathLabel(job.stage)}</span>
              <strong>{job.title}</strong>
              {job.status === 'processing' && (
                <small>{restoredJob ? 'Restored safely · ' : ''}{elapsedLabel(job.startedAt, clock)} · you may leave this page</small>
              )}
            </div>
            <span className="job-state-badge">{job.progress}%</span>
          </div>
          <div className="job-progress-track" aria-label={`Transcription progress ${job.progress}%`}>
            <span style={{ width: `${job.progress}%` }} />
          </div>
          {job.status === 'completed' && (
            <>
              <div className="media-result-actions">
                <button className="primary" type="button" onClick={openReadySheet} disabled={busy}>
                  Open Ready-to-Play Sheet
                </button>
                <button
                  className="ghost"
                  type="button"
                  onClick={() => downloadProtectedFile(`/api/media-transcriptions/${job.id}/download`, job.outputFilename)}
                >
                  Download JSON
                </button>
              </div>
              <div className="transcription-feedback" aria-label="Rate transcription quality">
                <span>{job.feedback ? 'Review saved' : 'After listening, was it playable?'}</span>
                <div>
                  <button
                    type="button"
                    className={job.feedback === 'accurate' ? 'selected' : ''}
                    disabled={feedbackBusy || Boolean(job.feedback)}
                    onClick={() => submitFeedback('accurate')}
                  >
                    Accurate
                  </button>
                  <button
                    type="button"
                    className={job.feedback === 'needs-work' ? 'selected' : ''}
                    disabled={feedbackBusy || Boolean(job.feedback)}
                    onClick={() => submitFeedback('needs-work')}
                  >
                    Needs work
                  </button>
                </div>
              </div>
            </>
          )}
          {job.status === 'failed' && (
            <div className="media-result-actions">
              <button className="primary" type="button" onClick={resetFailedJob}>
                {file ? 'Try this file again' : 'Choose the file again'}
              </button>
              {job.refunded && <small>Refund restored automatically.</small>}
            </div>
          )}
        </div>
      )}

      {status && <p className="form-status">{status}</p>}
      <small className="muscriptor-license-note">
        Foundation-model licence: CC BY-NC 4.0, non-commercial use only. Large models may take much longer on a CPU-only server.
      </small>
    </div>
  );
}
