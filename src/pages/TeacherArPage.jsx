import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import TeacherArSimulator from '../components/TeacherArSimulator.jsx';
import {
  detectTeacherArCapabilities,
  sanitizeTeacherArSettings,
  TEACHER_AR_HARDWARE_REQUIREMENTS,
} from '../engine/teacherArEngine.js';
import { TeacherWebXrRuntime } from '../engine/teacherWebXrRuntime.js';

function readSharedTeacherState(params) {
  let stored = {};
  try { stored = JSON.parse(window.sessionStorage.getItem('polymath_teacher_ar_state') || '{}'); } catch { /* Ignore malformed state. */ }
  return {
    ...stored,
    style: params?.get('style') === 'bikini' ? 'bikini' : stored.style,
  };
}

export default function TeacherArPage({ params }) {
  const channelId = params?.get('channel') || '';
  const requestedSimulator = params?.get('simulate') === '1';
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const runtimeRef = useRef(null);
  const [capabilities, setCapabilities] = useState(null);
  const [mode, setMode] = useState(requestedSimulator ? 'simulator' : 'spatial');
  const [active, setActive] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState({ phase: 'ready', message: 'Check the room, then start AR.' });
  const [settings, setSettings] = useState(() => sanitizeTeacherArSettings(readSharedTeacherState(params)));

  const setPartialSettings = useCallback((partial) => {
    setSettings((current) => sanitizeTeacherArSettings({ ...current, ...partial }));
  }, []);

  useEffect(() => {
    let live = true;
    detectTeacherArCapabilities().then((result) => {
      if (!live) return;
      setCapabilities(result);
      if (!requestedSimulator && !result.spatialAr && result.cameraPreview) setMode('simulator');
    });
    return () => { live = false; };
  }, [requestedSimulator]);

  useEffect(() => {
    if (!channelId || !('BroadcastChannel' in window)) return undefined;
    const channel = new window.BroadcastChannel(`polymath-teacher-ar:${channelId}`);
    channel.onmessage = (event) => {
      if (event.data?.type !== 'teacher-state') return;
      setPartialSettings(event.data.state || {});
    };
    channel.postMessage({ type: 'ar-receiver-ready' });
    return () => channel.close();
  }, [channelId, setPartialSettings]);

  useEffect(() => {
    runtimeRef.current?.setSettings(settings);
  }, [settings]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return undefined;
    const preventWorldSelection = (event) => {
      if (event.target?.closest?.('button, input, select, summary')) event.preventDefault();
    };
    overlay.addEventListener('beforexrselect', preventWorldSelection);
    return () => overlay.removeEventListener('beforexrselect', preventWorldSelection);
  }, []);

  useEffect(() => () => {
    runtimeRef.current?.end?.();
  }, []);

  async function startSpatialAr() {
    setError('');
    if (!capabilities?.spatialAr) {
      setError('Spatial AR is not supported by this browser/device. Use camera preview for testing.');
      return;
    }
    try {
      const runtime = new TeacherWebXrRuntime({
        canvas: canvasRef.current,
        overlayRoot: overlayRef.current,
        settings,
        onStatus: setStatus,
        onError: (runtimeError) => setError(runtimeError?.message || String(runtimeError)),
        onEnded: () => {
          runtimeRef.current = null;
          setActive(false);
          setStatus({ phase: 'ended', message: 'AR session ended.' });
        },
      });
      runtimeRef.current = runtime;
      await runtime.start();
      setActive(true);
    } catch (requestError) {
      runtimeRef.current = null;
      setActive(false);
      setError(requestError?.message || 'The AR session could not start.');
    }
  }

  function closeStudio() {
    runtimeRef.current?.end?.();
    window.close();
    window.location.hash = 'studio';
  }

  const capabilityText = useMemo(() => {
    if (!capabilities) return 'Checking AR hardware…';
    if (capabilities.spatialAr) return 'Spatial WebXR ready';
    if (capabilities.cameraPreview) return 'Camera simulator ready · spatial WebXR unavailable';
    if (!capabilities.secureContext) return 'HTTPS is required for camera and WebXR';
    return 'This browser has no compatible AR runtime';
  }, [capabilities]);

  return (
    <main className={`teacher-ar-page mode-${mode}`}>
      <canvas ref={canvasRef} className="teacher-ar-xr-canvas" aria-hidden="true" />

      {mode === 'simulator' && (
        <TeacherArSimulator settings={settings} onSettingsChange={setPartialSettings} onStatus={setStatus} />
      )}

      <div ref={overlayRef} className={`teacher-ar-overlay ${active ? 'is-xr-active' : ''}`}>
        <header className="teacher-ar-header">
          <div>
            <p className="eyebrow">Polymath spatial classroom</p>
            <h1>AR teacher studio</h1>
            <small>{capabilityText}</small>
          </div>
          <button type="button" className="ghost" onClick={closeStudio}>Close</button>
        </header>

        <section className="teacher-ar-status" aria-live="polite">
          <i aria-hidden="true" />
          <span>{status.message}</span>
        </section>

        <section className="teacher-ar-control-dock" aria-label="AR teacher controls">
          {!active && (
            <div className="teacher-ar-mode-tabs" role="tablist" aria-label="AR mode">
              <button type="button" className={mode === 'spatial' ? 'active' : ''} onClick={() => setMode('spatial')}>
                AR glasses
              </button>
              <button type="button" className={mode === 'simulator' ? 'active' : ''} onClick={() => setMode('simulator')}>
                Camera test
              </button>
            </div>
          )}

          {mode === 'spatial' && !active && (
            <button type="button" className="primary teacher-ar-start" disabled={!capabilities?.spatialAr} onClick={startSpatialAr}>
              Start spatial AR
            </button>
          )}

          {mode === 'spatial' && active && (
            <div className="teacher-ar-placement-controls">
              <button type="button" className="primary" onClick={() => runtimeRef.current?.placeCurrent()}>Place here</button>
              <button type="button" className="ghost" onClick={() => runtimeRef.current?.move({ x: -0.12 })}>Left</button>
              <button type="button" className="ghost" onClick={() => runtimeRef.current?.move({ x: 0.12 })}>Right</button>
              <button type="button" className="ghost" onClick={() => runtimeRef.current?.move({ z: -0.12 })}>Away</button>
              <button type="button" className="ghost" onClick={() => runtimeRef.current?.move({ z: 0.12 })}>Closer</button>
              <button type="button" className="ghost" onClick={() => runtimeRef.current?.resetPlacement()}>Re-scan</button>
            </div>
          )}

          <div className="teacher-ar-settings-row">
            <label>
              Teacher
              <select value={settings.style} onChange={(event) => setPartialSettings({ style: event.target.value })}>
                <option value="regular">Regular</option>
                <option value="bikini">Resort</option>
              </select>
            </label>
            <label>
              Piano side
              <select value={settings.pianoSide} onChange={(event) => setPartialSettings({ pianoSide: event.target.value })}>
                <option value="right">Right</option>
                <option value="left">Left</option>
                <option value="behind">Behind</option>
              </select>
            </label>
            <label>
              Size
              <input type="range" min="0.55" max="1.65" step="0.05" value={settings.scale} onChange={(event) => setPartialSettings({ scale: event.target.value })} />
            </label>
          </div>

          {error && <p className="teacher-ar-error" role="alert">{error}</p>}
        </section>

        {!active && mode === 'spatial' && (
          <details className="teacher-ar-buying-checklist">
            <summary>Alibaba glasses checklist</summary>
            <ul>{TEACHER_AR_HARDWARE_REQUIREMENTS.map((item) => <li key={item}>{item}</li>)}</ul>
            <strong>Ask the seller to open a WebXR immersive-AR test page before buying.</strong>
          </details>
        )}
      </div>
    </main>
  );
}
