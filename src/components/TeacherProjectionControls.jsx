import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

function displayUrl(session, mode) {
  if (!session?.id || !session?.token) return '';
  const params = new URLSearchParams({ session: session.id, token: session.token, mode });
  return `${window.location.origin}${window.location.pathname}#teacher-projection?${params.toString()}`;
}

export default function TeacherProjectionControls({
  style = 'regular',
  speaking = false,
  caption = '',
  onProjectionStarted,
}) {
  const [expanded, setExpanded] = useState(false);
  const [session, setSession] = useState(null);
  const [mode, setMode] = useState('wall');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const broadcastRef = useRef(null);
  const presentationRef = useRef(null);

  const state = useMemo(() => ({
    mode,
    style,
    speaking,
    visible: true,
    caption: String(caption || '').slice(0, 280),
  }), [caption, mode, speaking, style]);

  useEffect(() => {
    if (!session?.id) return undefined;
    const channel = 'BroadcastChannel' in window
      ? new window.BroadcastChannel(`polymath-teacher-projection:${session.id}`)
      : null;
    broadcastRef.current = channel;
    channel?.postMessage({ state });
    channel?.addEventListener('message', (event) => {
      if (event.data?.type === 'receiver-ready') channel.postMessage({ state });
    });
    return () => {
      channel?.close();
      broadcastRef.current = null;
    };
  }, [session?.id]);

  useEffect(() => {
    if (!session?.id) return;
    broadcastRef.current?.postMessage({ state });
    if (presentationRef.current?.state === 'connected') {
      presentationRef.current.send(JSON.stringify({ state }));
    }
    const timer = window.setTimeout(() => {
      apiRequest(`/api/teacher/projection-sessions/${encodeURIComponent(session.id)}`, {
        method: 'PUT',
        body: JSON.stringify({ state }),
      }).catch((requestError) => setError(requestError.message));
    }, 90);
    return () => window.clearTimeout(timer);
  }, [session?.id, state]);

  async function prepareProjection() {
    setExpanded((current) => !current);
    if (session || busy) return;
    setBusy(true);
    setError('');
    try {
      const created = await apiRequest('/api/teacher/projection-sessions', {
        method: 'POST',
        body: JSON.stringify({ state }),
      });
      setSession(created);
    } catch (requestError) {
      setError(requestError.message);
      setExpanded(true);
    } finally {
      setBusy(false);
    }
  }

  function openDisplay(nextMode) {
    if (!session) return;
    setMode(nextMode);
    const popup = window.open(displayUrl(session, nextMode), `polymath-teacher-${session.id}`, 'popup,width=1280,height=800');
    if (!popup) {
      setError('The browser blocked the display window. Allow pop-ups for Polymath, then try again.');
      return;
    }
    popup.focus?.();
    onProjectionStarted?.();
  }

  async function presentWirelessly() {
    if (!session || !window.PresentationRequest) return;
    setError('');
    try {
      const request = new window.PresentationRequest([displayUrl(session, mode)]);
      const connection = await request.start();
      presentationRef.current = connection;
      connection.addEventListener('connect', () => connection.send(JSON.stringify({ state })));
      if (connection.state === 'connected') connection.send(JSON.stringify({ state }));
      onProjectionStarted?.();
    } catch (requestError) {
      setError(requestError.message || 'No compatible wireless presentation display was selected.');
    }
  }

  async function copyDisplayLink() {
    if (!session) return;
    try {
      await navigator.clipboard.writeText(displayUrl(session, mode));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError('The browser could not copy the link. Open the display window instead.');
    }
  }

  async function endProjection() {
    if (session?.id) {
      await apiRequest(`/api/teacher/projection-sessions/${encodeURIComponent(session.id)}`, {
        method: 'DELETE',
      }).catch(() => {});
    }
    presentationRef.current?.terminate?.();
    presentationRef.current = null;
    setSession(null);
    setExpanded(false);
    setError('');
  }

  return (
    <section className="teacher-projection-controls">
      <button type="button" className="teacher-project-trigger" onClick={prepareProjection} aria-expanded={expanded}>
        <span aria-hidden="true">▣</span>
        {busy ? 'Preparing…' : 'Project teacher'}
      </button>

      {expanded && (
        <div className="teacher-projection-menu">
          <div>
            <p className="eyebrow">External display</p>
            <h3>Put the teacher in the room</h3>
          </div>

          <div className="teacher-projection-mode-grid">
            <button type="button" className={mode === 'wall' ? 'active' : ''} onClick={() => setMode('wall')}>
              <strong>Wall</strong>
              <small>Projector, TV or second monitor</small>
            </button>
            <button type="button" className={mode === 'hologram' ? 'active' : ''} onClick={() => setMode('hologram')}>
              <strong>Pyramid</strong>
              <small>Four-way reflection illusion</small>
            </button>
          </div>

          {error && <p className="teacher-projection-error" role="alert">{error}</p>}

          <div className="teacher-projection-actions">
            <button type="button" className="primary" disabled={!session || busy} onClick={() => openDisplay(mode)}>
              Open display
            </button>
            {window.PresentationRequest && (
              <button type="button" className="ghost" disabled={!session || busy} onClick={presentWirelessly}>
                Wireless display
              </button>
            )}
            <button type="button" className="ghost" disabled={!session || busy} onClick={copyDisplayLink}>
              {copied ? 'Copied' : 'Copy display link'}
            </button>
            {session && <button type="button" className="ghost" onClick={endProjection}>End</button>}
          </div>

          <p className="teacher-projection-help">
            Connect a projector by HDMI/USB-C, or open the copied link on a projector-connected browser. The controller keeps the lesson; the external screen shows only the teacher.
          </p>
        </div>
      )}
    </section>
  );
}
