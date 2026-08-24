import { useEffect, useRef, useState } from 'react';
import { apiRequest, downloadProtectedFile, fetchProtectedFile } from '../services/api.js';

const MEDIA_ACCEPT = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.mp4,.mov,.webm,.mkv,.avi';

export default function MediaTranscriptionPanel({
  user,
  setUser,
  onNavigate,
  instrument,
  onReadyFile,
}) {
  const [capability, setCapability] = useState(null);
  const [file, setFile] = useState(null);
  const [playbackMode, setPlaybackMode] = useState(instrument === 'piano' ? 'full' : 'instrumental');
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const pollTimer = useRef(null);
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

  useEffect(() => {
    let cancelled = false;
    apiRequest('/api/media-transcriptions/capabilities')
      .then((data) => {
        if (!cancelled) setCapability(data);
      })
      .catch((error) => {
        if (!cancelled) setStatus(error.message);
      });
    return () => {
      cancelled = true;
      clearPolling();
    };
  }, []);

  async function refreshJob(jobId) {
    try {
      const data = await apiRequest(`/api/media-transcriptions/${jobId}`);
      setJob(data.job);
      if (data.user && setUser) setUser(data.user);
      if (data.job.status === 'completed') {
        setStatus('Your ready-to-play sheet is ready. Opening the piano studio...');
        clearPolling();
        await openReadySheet(data.job);
        return;
      }
      if (data.job.status === 'failed') {
        setStatus(data.job.error || 'MuScriptor could not transcribe this recording.');
        clearPolling();
        return;
      }
      pollTimer.current = window.setTimeout(() => refreshJob(jobId), 4000);
    } catch (error) {
      setStatus(error.message);
      pollTimer.current = window.setTimeout(() => refreshJob(jobId), 6000);
    }
  }

  function chooseFile(event) {
    const selected = event.target.files?.[0] || null;
    setJob(null);
    setStatus('');
    setFile(selected);
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
      const form = new FormData();
      form.append('media', file, file.name);
      form.append('instrument', instrument || 'band');
      form.append('title', file.name.replace(/\.[^.]+$/, ''));
      form.append('playbackMode', playbackMode);
      form.append('rightsConfirmed', 'true');
      form.append('paymentMethod', paymentMethod);
      const data = await apiRequest('/api/media-transcriptions', {
        method: 'POST',
        body: form,
      });
      setCapability(data.capability);
      if (data.user && setUser) setUser(data.user);
      setJob(data.job);
      setStatus('MuScriptor is preparing your recording.');
      clearPolling();
      pollTimer.current = window.setTimeout(() => refreshJob(data.job.id), 1500);
    } catch (error) {
      setStatus(error.message);
      if (error.details?.capability) setCapability(error.details.capability);
    } finally {
      setBusy(false);
    }
  }

  async function openReadySheet(completedJob = null) {
    const readyJob = completedJob?.id ? completedJob : job;
    if (!readyJob?.id || !onReadyFile) return;
    setBusy(true);
    try {
      const readyFile = await fetchProtectedFile(
        `/api/media-transcriptions/${readyJob.id}/download`,
        readyJob.outputFilename || 'muscriptor-ready-to-play.json',
      );
      await onReadyFile(readyFile);
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
          <strong>MuScriptor is not enabled on this server.</strong>
          <span>{capability.reason}</span>
        </div>
      )}

      {capability?.enabled && capability.adminOnly && !user.admin && (
        <div className={'quota-warning'}>
          <strong>MuScriptor model testing is restricted.</strong>
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

      <label className="media-rights-check">
        <input
          type="checkbox"
          checked={rightsConfirmed}
          onChange={(event) => setRightsConfirmed(event.target.checked)}
        />
        <span>I have permission to transcribe this recording.</span>
      </label>

      {instrument === 'piano' && (
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

      {!job && (
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
            <div><span>{job.stage}</span><strong>{job.title}</strong></div>
            <span className="job-state-badge">{job.progress}%</span>
          </div>
          <div className="job-progress-track" aria-label={`Transcription progress ${job.progress}%`}>
            <span style={{ width: `${job.progress}%` }} />
          </div>
          {job.status === 'completed' && (
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
          )}
        </div>
      )}

      {status && <p className="form-status">{status}</p>}
      <small className="muscriptor-license-note">
        MuScriptor model weights: CC BY-NC 4.0, non-commercial use only. Large may take much longer on a CPU-only server.
      </small>
    </div>
  );
}
