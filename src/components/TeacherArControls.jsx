import { useEffect, useMemo, useState } from 'react';
import {
  detectTeacherArCapabilities,
  teacherArRoute,
} from '../engine/teacherArEngine.js';

const CHANNEL_SESSION_KEY = 'polymath_teacher_ar_channel';

function getChannelId() {
  let value = window.sessionStorage.getItem(CHANNEL_SESSION_KEY);
  if (!value) {
    value = globalThis.crypto?.randomUUID?.() || `ar-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.sessionStorage.setItem(CHANNEL_SESSION_KEY, value);
  }
  return value;
}

export default function TeacherArControls({ style = 'regular', speaking = false, caption = '', onArStarted }) {
  const [expanded, setExpanded] = useState(false);
  const [capabilities, setCapabilities] = useState(null);
  const channelId = useMemo(getChannelId, []);
  const sharedState = useMemo(() => ({
    style,
    speaking,
    caption: String(caption || '').slice(0, 280),
  }), [caption, speaking, style]);

  useEffect(() => {
    if (!expanded || capabilities) return;
    let active = true;
    detectTeacherArCapabilities().then((result) => {
      if (active) setCapabilities(result);
    });
    return () => { active = false; };
  }, [capabilities, expanded]);

  useEffect(() => {
    window.sessionStorage.setItem('polymath_teacher_ar_state', JSON.stringify(sharedState));
    if (!('BroadcastChannel' in window)) return undefined;
    const channel = new window.BroadcastChannel(`polymath-teacher-ar:${channelId}`);
    channel.postMessage({ type: 'teacher-state', state: sharedState });
    channel.onmessage = (event) => {
      if (event.data?.type === 'ar-receiver-ready') {
        channel.postMessage({ type: 'teacher-state', state: sharedState });
      }
    };
    return () => channel.close();
  }, [channelId, sharedState]);

  function openStudio(simulate = false) {
    const route = teacherArRoute({ style, channel: channelId, simulate });
    const url = `${window.location.origin}${window.location.pathname}#${route}`;
    const popup = window.open(url, `polymath-teacher-ar-${channelId}`);
    if (!popup) window.location.hash = route;
    else popup.focus?.();
    onArStarted?.();
  }

  const modeLabel = !capabilities
    ? 'Checking this device…'
    : capabilities.spatialAr
      ? 'Spatial AR is available'
      : capabilities.cameraPreview
        ? 'Camera preview available'
        : 'AR unavailable in this browser';

  return (
    <section className="teacher-ar-controls">
      <button type="button" className="teacher-ar-trigger" onClick={() => setExpanded((current) => !current)} aria-expanded={expanded}>
        <span aria-hidden="true">AR</span>
        AR glasses
      </button>

      {expanded && (
        <div className="teacher-ar-menu">
          <div>
            <p className="eyebrow">Spatial classroom</p>
            <h3>Put the teacher beside your piano</h3>
          </div>
          <p className={`teacher-ar-capability ${capabilities?.spatialAr ? 'is-ready' : ''}`}>{modeLabel}</p>
          <div className="teacher-ar-actions">
            <button type="button" className="primary" disabled={Boolean(capabilities && !capabilities.spatialAr)} onClick={() => openStudio(false)}>
              Open spatial AR
            </button>
            <button type="button" className="ghost" disabled={Boolean(capabilities && !capabilities.cameraPreview)} onClick={() => openStudio(true)}>
              Test with camera
            </button>
          </div>
          <p className="teacher-ar-help">
            Real AR requires 6DoF glasses and a browser with WebXR immersive AR. Camera mode lets us test placement before buying hardware.
          </p>
        </div>
      )}
    </section>
  );
}
