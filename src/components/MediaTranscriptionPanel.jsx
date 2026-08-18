import { useEffect, useRef, useState } from 'react';
import { apiRequest, downloadProtectedFile, fetchProtectedFile } from '../services/api.js';

const MAX_MEDIA_BYTES = 100 * 1024 * 1024;
const MEDIA_ACCEPT = 'audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.mp4,.mov,.webm,.mkv,.avi';

export default function MediaTranscriptionPanel({
  user,
  onNavigate,
  instrument,
  onReadyFile,
}) {
  const [capability, setCapability] = useState(null);
  const [file, setFile] = useState(null);
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const pollTimer = useRef(null);

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
      if (data.job.status === 'completed') {
        setStatus('Your ready-to-play sheet is ready.');
        clearPolling();
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
    if (selected && selected.size > MAX_MEDIA_BYTES) {
      setFile(null);
      setStatus('Audio or video must be smaller than 100 MB.');
    } else {
      setFile(selected);
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
      const form = new FormData();
      form.append('media', file, file.name);
      form.append('instrument', instrument || 'band');
      form.append('title', file.name.replace(/\.[^.]+$/, ''));
      form.append('rightsConfirmed', 'true');
      const data = await apiRequest('/api/media-transcriptions', {
        method: 'POST',
        body: form,
      });
      setCapability(data.capability);
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

  async function openReadySheet() {
    if (!job?.id || !onReadyFile) return;
    setBusy(true);
    try {
      const readyFile = await fetchProtectedFile(
        `/api/media-transcriptions/${job.id}/download`,
        job.outputFilename || 'muscriptor-ready-to-play.json',
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

      <label className="upload-box compact">
        <input type="file" accept={MEDIA_ACCEPT} onChange={chooseFile} disabled={busy || capability?.enabled === false} />
        <span>{file ? file.name : 'Choose MP3, audio, or music video'}</span>
        <small>Up to 100 MB · first 10 minutes</small>
      </label>

      <label className="media-rights-check">
        <input
          type="checkbox"
          checked={rightsConfirmed}
          onChange={(event) => setRightsConfirmed(event.target.checked)}
        />
        <span>I have permission to transcribe this recording.</span>
      </label>

      {!job && (
        <button
          className="primary full"
          type="button"
          onClick={startTranscription}
          disabled={!file || !rightsConfirmed || busy || capability?.enabled === false}
        >
          {busy ? 'Uploading…' : 'Transcribe with MuScriptor'}
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
