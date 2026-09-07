import { useEffect, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

const DEFAULT_STATE = Object.freeze({
  mode: 'wall',
  style: 'regular',
  speaking: false,
  visible: true,
  caption: '',
  version: 0,
});

function safeProjectionState(value) {
  return {
    mode: value?.mode === 'hologram' ? 'hologram' : 'wall',
    style: value?.style === 'bikini' ? 'bikini' : 'regular',
    speaking: Boolean(value?.speaking),
    visible: value?.visible !== false,
    caption: String(value?.caption || '').slice(0, 280),
    version: Number(value?.version || 0),
  };
}

function ProjectionCharacter({ state, className = '' }) {
  return (
    <div
      className={`projection-character style-${state.style} ${state.speaking ? 'is-speaking' : 'is-idle'} ${className}`}
      aria-hidden="true"
    >
      <div className="projection-character-pose projection-character-idle" />
      <div className="projection-character-pose projection-character-speaking" />
    </div>
  );
}

export default function TeacherProjectionPage({ params }) {
  const sessionId = params?.get('session') || '';
  const token = params?.get('token') || '';
  const initialMode = params?.get('mode') === 'hologram' ? 'hologram' : 'wall';
  const [projection, setProjection] = useState({ ...DEFAULT_STATE, mode: initialMode });
  const [connection, setConnection] = useState(sessionId ? 'connecting' : 'preview');
  const [controlsVisible, setControlsVisible] = useState(true);
  const latestVersionRef = useRef(0);

  useEffect(() => {
    document.body.classList.add('projection-output-active');
    return () => document.body.classList.remove('projection-output-active');
  }, []);

  useEffect(() => {
    function accept(next) {
      const state = safeProjectionState(next?.state || next);
      if (state.version && state.version < latestVersionRef.current) return;
      latestVersionRef.current = Math.max(latestVersionRef.current, state.version);
      setProjection(state);
      setConnection('connected');
    }

    let channel;
    if (sessionId && 'BroadcastChannel' in window) {
      channel = new window.BroadcastChannel(`polymath-teacher-projection:${sessionId}`);
      channel.onmessage = (event) => accept(event.data);
      channel.postMessage({ type: 'receiver-ready' });
    }

    const receiver = navigator.presentation?.receiver;
    const attachedConnections = new Set();
    function attachPresentation(presentationConnection) {
      if (!presentationConnection || attachedConnections.has(presentationConnection)) return;
      attachedConnections.add(presentationConnection);
      presentationConnection.addEventListener('message', (event) => {
        try { accept(JSON.parse(event.data)); } catch { /* Ignore malformed controller data. */ }
      });
      presentationConnection.start?.();
    }
    receiver?.connectionList?.then((connections) => connections.forEach(attachPresentation)).catch(() => {});
    const onConnection = (event) => attachPresentation(event.connection);
    receiver?.addEventListener?.('connectionavailable', onConnection);

    let cancelled = false;
    let timer = 0;
    async function poll() {
      if (!sessionId || !token || cancelled) return;
      try {
        const result = await apiRequest(`/api/teacher/projection-sessions/${encodeURIComponent(sessionId)}`, {
          headers: { 'X-Projection-Token': token },
        });
        if (!cancelled) accept(result.state);
      } catch (error) {
        if (!cancelled) setConnection(error.status === 404 ? 'expired' : 'reconnecting');
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, 1000);
      }
    }
    poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      channel?.close();
      receiver?.removeEventListener?.('connectionavailable', onConnection);
      attachedConnections.forEach((item) => item.close?.());
    };
  }, [sessionId, token]);

  async function enterFullscreen() {
    try {
      await document.documentElement.requestFullscreen?.();
      setControlsVisible(false);
    } catch {
      setControlsVisible(true);
    }
  }

  const hidden = !projection.visible;
  return (
    <main className={`teacher-projection-output mode-${projection.mode} ${hidden ? 'is-hidden' : ''}`}>
      {projection.mode === 'wall' ? (
        <div className="projection-wall-stage">
          <ProjectionCharacter state={projection} />
          {projection.caption && <p className="projection-caption">{projection.caption}</p>}
        </div>
      ) : (
        <div className="projection-hologram-stage" aria-label="Four-way reflective pyramid display">
          <ProjectionCharacter state={projection} className="projection-hologram-copy is-top" />
          <ProjectionCharacter state={projection} className="projection-hologram-copy is-right" />
          <ProjectionCharacter state={projection} className="projection-hologram-copy is-bottom" />
          <ProjectionCharacter state={projection} className="projection-hologram-copy is-left" />
        </div>
      )}

      {controlsVisible && (
        <div className="projection-display-controls">
          <strong>{projection.mode === 'hologram' ? 'Pyramid hologram illusion' : 'Wall projection'}</strong>
          <small>{connection === 'connected' ? 'Connected to teacher controls' : connection}</small>
          <button type="button" className="primary" onClick={enterFullscreen}>Enter fullscreen</button>
          <button type="button" className="ghost" onClick={() => setControlsVisible(false)}>Hide controls</button>
        </div>
      )}

      {!controlsVisible && (
        <button type="button" className="projection-show-controls" onClick={() => setControlsVisible(true)} aria-label="Show projection controls" />
      )}
    </main>
  );
}
