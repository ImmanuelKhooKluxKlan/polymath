import { useCallback, useEffect, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

const SAVED_INVITE_KEY = 'polymath_human_lesson_invite';

function readSavedInvite(meetingId) {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(SAVED_INVITE_KEY) || 'null');
    if (!value || (meetingId && value.meetingId !== meetingId)) return null;
    return value;
  } catch {
    return null;
  }
}

function participantName(participant) {
  return String(participant?.name || 'Polymath member').trim() || 'Polymath member';
}

function initials(name) {
  return participantName({ name }).split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
}

function remainingLabel(expiresAt) {
  if (!expiresAt) return 'Waiting for host';
  const seconds = Math.max(0, Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function StreamVideo({ stream, muted, label }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.srcObject = stream || null;
    if (stream) ref.current.play().catch(() => {});
  }, [stream]);
  return <video ref={ref} autoPlay playsInline muted={muted} aria-label={label} />;
}

export default function HumanLessonRoomPage({ user, setUser, meetingId = '', onNavigate }) {
  const savedInvite = readSavedInvite(meetingId);
  const [credentials, setCredentials] = useState({
    meetingId: meetingId || savedInvite?.meetingId || '',
    accessCode: savedInvite?.accessCode || '',
  });
  const [meeting, setMeeting] = useState(null);
  const [participants, setParticipants] = useState([]);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStreams, setRemoteStreams] = useState({});
  const [status, setStatus] = useState('');
  const [joining, setJoining] = useState(false);
  const [cameraOn, setCameraOn] = useState(true);
  const [microphoneOn, setMicrophoneOn] = useState(true);
  const [clock, setClock] = useState('Waiting for host');
  const autoJoinInviteRef = useRef(Boolean(savedInvite?.autoJoin));
  const autoJoinAttemptedRef = useRef(false);
  const localStreamRef = useRef(null);
  const meetingRef = useRef(null);
  const userRef = useRef(user);
  const iceServersRef = useRef([]);
  const peersRef = useRef(new Map());
  const signalCursorRef = useRef('1970-01-01T00:00:00.000Z');
  const refreshRunningRef = useRef(false);

  useEffect(() => { userRef.current = user; }, [user]);
  useEffect(() => {
    if (!meetingId) return;
    const invite = readSavedInvite(meetingId);
    setCredentials((current) => ({
      meetingId,
      accessCode: invite?.accessCode || current.accessCode,
    }));
  }, [meetingId]);

  useEffect(() => {
    if (!user?.user_id || !credentials.meetingId || meeting || joining
        || !autoJoinInviteRef.current || autoJoinAttemptedRef.current) return;
    autoJoinAttemptedRef.current = true;
    joinLesson();
  }, [user?.user_id, credentials.meetingId, meeting, joining]);

  const sendSignal = useCallback(async (toUserId, kind, payload) => {
    const activeMeeting = meetingRef.current;
    if (!activeMeeting?.meetingId) return;
    await apiRequest(`/api/human-lessons/${encodeURIComponent(activeMeeting.meetingId)}/signals`, {
      method: 'POST',
      globalProgress: false,
      body: JSON.stringify({ toUserId, kind, payload }),
    });
  }, []);

  const closePeer = useCallback((peerId) => {
    const entry = peersRef.current.get(peerId);
    if (!entry) return;
    entry.connection.ontrack = null;
    entry.connection.onicecandidate = null;
    entry.connection.close();
    peersRef.current.delete(peerId);
    setRemoteStreams((current) => {
      const next = { ...current };
      delete next[peerId];
      return next;
    });
  }, []);

  const ensurePeer = useCallback((peerId) => {
    const known = peersRef.current.get(peerId);
    if (known) return known;
    const connection = new window.RTCPeerConnection({ iceServers: iceServersRef.current });
    const entry = { connection, pendingCandidates: [], offerSent: false };
    peersRef.current.set(peerId, entry);
    for (const track of localStreamRef.current?.getTracks?.() || []) {
      connection.addTrack(track, localStreamRef.current);
    }
    connection.ontrack = (event) => {
      const stream = event.streams?.[0] || new window.MediaStream([event.track]);
      setRemoteStreams((current) => ({ ...current, [peerId]: stream }));
    };
    connection.onicecandidate = (event) => {
      if (!event.candidate) return;
      sendSignal(peerId, 'candidate', event.candidate.toJSON()).catch((error) => setStatus(error.message));
    };
    connection.onconnectionstatechange = () => {
      if (['failed', 'closed'].includes(connection.connectionState)) closePeer(peerId);
    };
    return entry;
  }, [closePeer, sendSignal]);

  const offerPeer = useCallback(async (peerId) => {
    const entry = ensurePeer(peerId);
    if (entry.offerSent || entry.connection.signalingState !== 'stable') return;
    entry.offerSent = true;
    try {
      const offer = await entry.connection.createOffer();
      await entry.connection.setLocalDescription(offer);
      await sendSignal(peerId, 'offer', { type: offer.type, sdp: offer.sdp });
    } catch (error) {
      entry.offerSent = false;
      setStatus(`Could not connect to ${peerId.slice(0, 8)}. ${error.message}`);
    }
  }, [ensurePeer, sendSignal]);

  const acceptSignal = useCallback(async (signal) => {
    const peerId = signal.fromUserId;
    const entry = ensurePeer(peerId);
    try {
      if (signal.kind === 'offer') {
        await entry.connection.setRemoteDescription(signal.payload);
        for (const candidate of entry.pendingCandidates.splice(0)) {
          await entry.connection.addIceCandidate(candidate);
        }
        const answer = await entry.connection.createAnswer();
        await entry.connection.setLocalDescription(answer);
        await sendSignal(peerId, 'answer', { type: answer.type, sdp: answer.sdp });
      } else if (signal.kind === 'answer') {
        if (entry.connection.signalingState === 'have-local-offer') {
          await entry.connection.setRemoteDescription(signal.payload);
          for (const candidate of entry.pendingCandidates.splice(0)) {
            await entry.connection.addIceCandidate(candidate);
          }
        }
      } else if (signal.kind === 'candidate') {
        if (entry.connection.remoteDescription) await entry.connection.addIceCandidate(signal.payload);
        else entry.pendingCandidates.push(signal.payload);
      }
    } catch (error) {
      setStatus(`A participant connection was interrupted. ${error.message}`);
    }
  }, [ensurePeer, sendSignal]);

  const refreshRoom = useCallback(async () => {
    const activeMeeting = meetingRef.current;
    const activeUser = userRef.current;
    if (!activeMeeting?.meetingId || !activeUser?.user_id || refreshRunningRef.current) return;
    refreshRunningRef.current = true;
    try {
      const data = await apiRequest(
        `/api/human-lessons/${encodeURIComponent(activeMeeting.meetingId)}/room-state?after=${encodeURIComponent(signalCursorRef.current)}`,
        { globalProgress: false },
      );
      meetingRef.current = data.meeting;
      setMeeting(data.meeting);
      setParticipants(data.meeting.participants || []);
      iceServersRef.current = data.iceServers || [];
      for (const signal of data.signals || []) await acceptSignal(signal);
      signalCursorRef.current = data.cursor || signalCursorRef.current;
      const activePeerIds = new Set((data.meeting.participants || [])
        .map((participant) => participant.user_id)
        .filter((participantId) => participantId && participantId !== activeUser.user_id));
      for (const peerId of activePeerIds) {
        ensurePeer(peerId);
        if (String(activeUser.user_id).localeCompare(String(peerId)) < 0) await offerPeer(peerId);
      }
      for (const peerId of peersRef.current.keys()) {
        if (!activePeerIds.has(peerId)) closePeer(peerId);
      }
      if (data.meeting.status === 'ended' || data.meeting.status === 'cancelled') {
        setStatus('This lesson has ended.');
      }
    } catch (error) {
      setStatus(error.message);
    } finally {
      refreshRunningRef.current = false;
    }
  }, [acceptSignal, closePeer, ensurePeer, offerPeer]);

  async function requestMedia() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('This browser cannot access a camera or microphone. You can still enter the room.');
      return null;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      localStreamRef.current?.getTracks?.().forEach((track) => track.stop());
      localStreamRef.current = stream;
      setLocalStream(stream);
      setCameraOn(true);
      setMicrophoneOn(true);
      for (const [peerId, entry] of peersRef.current) {
        const existingTrackIds = new Set(entry.connection.getSenders().map((sender) => sender.track?.id));
        for (const track of stream.getTracks()) {
          if (!existingTrackIds.has(track.id)) entry.connection.addTrack(track, stream);
        }
        entry.offerSent = false;
        if (String(userRef.current?.user_id).localeCompare(String(peerId)) < 0) offerPeer(peerId);
      }
      return stream;
    } catch {
      setStatus('Camera or microphone access was not allowed. You entered without live audio/video; use “Enable camera + mic” to try again.');
      return null;
    }
  }

  async function joinLesson(event) {
    event?.preventDefault?.();
    if (!user) {
      onNavigate('account', { next: 'lesson-room' });
      return;
    }
    setJoining(true);
    setStatus('Opening your private lesson…');
    try {
      const data = await apiRequest('/api/human-lessons/join', {
        method: 'POST',
        body: JSON.stringify(credentials),
      });
      meetingRef.current = data.meeting;
      setMeeting(data.meeting);
      setParticipants(data.meeting.participants || []);
      setUser?.(data.user);
      window.sessionStorage.removeItem(SAVED_INVITE_KEY);
      await requestMedia();
      setStatus(data.waitingForHost ? 'You are in. Waiting for the teacher to start.' : 'Private lesson connected.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setJoining(false);
    }
  }

  useEffect(() => {
    if (!meeting?.meetingId) return undefined;
    refreshRoom();
    const poll = window.setInterval(refreshRoom, 1500);
    const heartbeat = window.setInterval(() => {
      apiRequest(`/api/human-lessons/${encodeURIComponent(meeting.meetingId)}/heartbeat`, {
        method: 'POST', body: '{}', globalProgress: false,
      }).catch((error) => setStatus(error.message));
    }, 10_000);
    return () => {
      window.clearInterval(poll);
      window.clearInterval(heartbeat);
    };
  }, [meeting?.meetingId, refreshRoom]);

  useEffect(() => {
    if (!meeting) return undefined;
    const update = () => setClock(remainingLabel(meeting.expiresAt));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [meeting?.expiresAt]);

  useEffect(() => () => {
    localStreamRef.current?.getTracks?.().forEach((track) => track.stop());
    for (const entry of peersRef.current.values()) entry.connection.close();
    peersRef.current.clear();
  }, []);

  function toggleMedia(kind) {
    const tracks = kind === 'video' ? localStreamRef.current?.getVideoTracks?.() : localStreamRef.current?.getAudioTracks?.();
    const enabled = !(kind === 'video' ? cameraOn : microphoneOn);
    for (const track of tracks || []) track.enabled = enabled;
    if (kind === 'video') setCameraOn(enabled);
    else setMicrophoneOn(enabled);
  }

  async function leaveLesson({ end = false } = {}) {
    if (!meeting?.meetingId) return;
    if (end && !window.confirm('End this lesson for everyone?')) return;
    try {
      await apiRequest(`/api/human-lessons/${encodeURIComponent(meeting.meetingId)}/${end ? 'end' : 'leave'}`, {
        method: 'POST', body: '{}',
      });
    } catch (error) {
      setStatus(error.message);
      return;
    }
    localStreamRef.current?.getTracks?.().forEach((track) => track.stop());
    onNavigate('find-teacher');
  }

  if (!user) {
    return (
      <section className="page-shell narrow-page empty-state-page">
        <div className="empty-state"><h1>Sign in to join a private lesson.</h1><button className="primary" type="button" onClick={() => onNavigate('account', { next: 'lesson-room' })}>Sign in</button></div>
      </section>
    );
  }

  if (!meeting) {
    return (
      <section className="page-shell human-lesson-entry-page">
        <button className="text-button" type="button" onClick={() => onNavigate('find-teacher')}>← Back to Learn</button>
        <form className="human-lesson-entry-card" onSubmit={joinLesson}>
          <div><p className="eyebrow">Private virtual lesson</p><h1>Join your teacher</h1><p>Enter the meeting ID and password sent in your private chat.</p></div>
          <label className="field">Meeting ID<input value={credentials.meetingId} onChange={(event) => setCredentials({ ...credentials, meetingId: event.target.value.toUpperCase() })} placeholder="PM-ABCD-2345" autoCapitalize="characters" required /></label>
          <label className="field">Password<input value={credentials.accessCode} onChange={(event) => setCredentials({ ...credentials, accessCode: event.target.value.toUpperCase() })} placeholder={savedInvite?.isHost ? 'Not required for host' : 'ABCD-2345'} autoCapitalize="characters" required={!savedInvite?.isHost} /></label>
          <button className="primary" type="submit" disabled={joining}>{joining ? 'Opening lesson…' : 'Join lesson'}</button>
          {status && <p className="form-status" role="status">{status}</p>}
        </form>
      </section>
    );
  }

  const remoteParticipants = participants.filter((participant) => participant.user_id !== user.user_id);
  return (
    <section className="page-shell human-lesson-room-page">
      <header className="human-lesson-room-header">
        <div><p className="eyebrow">{meeting.kind === 'breakout' ? meeting.roomName : 'Live lesson'}</p><h1>{meeting.title}</h1><p>{meeting.meetingId} · {participants.length} connected</p></div>
        <div className="lesson-room-clock"><small>{meeting.status === 'ready' ? 'Not started' : 'Time left'}</small><strong>{clock}</strong></div>
      </header>

      <div className="lesson-video-grid">
        <article className="lesson-video-tile mine">
          {localStream ? <StreamVideo stream={localStream} muted label="Your camera" /> : <div className="lesson-video-placeholder"><span>{initials(user.name)}</span><small>Camera off</small></div>}
          <strong>You {meeting.isHost ? '· Host' : ''}</strong>
        </article>
        {remoteParticipants.map((participant) => (
          <article className="lesson-video-tile" key={participant.user_id}>
            {remoteStreams[participant.user_id]
              ? <StreamVideo stream={remoteStreams[participant.user_id]} muted={false} label={`${participantName(participant)} camera`} />
              : <div className="lesson-video-placeholder"><span>{initials(participant.name)}</span><small>Connecting…</small></div>}
            <strong>{participantName(participant)} {participant.isHost ? '· Host' : ''}</strong>
          </article>
        ))}
        {!remoteParticipants.length && <div className="lesson-room-waiting"><span>♪</span><strong>{meeting.status === 'ready' ? 'Waiting for the teacher to start' : 'Waiting for another participant'}</strong></div>}
      </div>

      <div className="lesson-call-controls" aria-label="Call controls">
        {localStream ? (
          <>
            <button type="button" className={microphoneOn ? 'active' : ''} onClick={() => toggleMedia('audio')}>{microphoneOn ? 'Mute' : 'Unmute'}</button>
            <button type="button" className={cameraOn ? 'active' : ''} onClick={() => toggleMedia('video')}>{cameraOn ? 'Camera off' : 'Camera on'}</button>
          </>
        ) : <button type="button" onClick={requestMedia}>Enable camera + mic</button>}
        <button className="danger" type="button" onClick={() => leaveLesson()}>Leave</button>
        {meeting.isHost && <button className="danger-outline" type="button" onClick={() => leaveLesson({ end: true })}>End for everyone</button>}
      </div>
      {status && <p className="form-status lesson-room-status" role="status">{status}</p>}
      <p className="lesson-room-privacy">Video and audio travel directly between lesson participants. Polymath stores meeting access, billing, and connection signals—not the call recording.</p>
    </section>
  );
}

export { SAVED_INVITE_KEY };
