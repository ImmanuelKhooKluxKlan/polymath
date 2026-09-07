import { useEffect, useRef, useState } from 'react';
import { pointFromStageTap, sanitizeArPoint } from '../engine/teacherArEngine.js';

export default function TeacherArSimulator({ settings, onSettingsChange, onStatus }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [point, setPoint] = useState(() => sanitizeArPoint());
  const [error, setError] = useState('');

  useEffect(() => () => {
    streamRef.current?.getTracks?.().forEach((track) => track.stop());
  }, []);

  async function startCamera() {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current?.getTracks?.().forEach((track) => track.stop());
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      onStatus?.({ phase: 'simulating', message: 'Tap the floor beside your piano to position the teacher.' });
    } catch (requestError) {
      const message = requestError?.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access and try again.'
        : requestError?.message || 'The camera could not be opened.';
      setError(message);
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks?.().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  }

  function placeTeacher(event) {
    if (event.target.closest?.('button, input, select, label')) return;
    setPoint(pointFromStageTap(event, event.currentTarget.getBoundingClientRect()));
  }

  return (
    <div className="teacher-ar-simulator" onPointerDown={placeTeacher} role="application" aria-label="Camera AR teacher simulator">
      <video ref={videoRef} muted playsInline className={cameraActive ? 'is-active' : ''} />
      {!cameraActive && (
        <div className="teacher-ar-camera-placeholder">
          <strong>Camera AR preview</strong>
          <span>Test teacher placement without glasses.</span>
          <button type="button" className="primary" onClick={startCamera}>Allow camera</button>
        </div>
      )}

      <div
        className={`teacher-ar-simulator-character style-${settings.style} ${settings.speaking ? 'is-speaking' : 'is-idle'}`}
        style={{
          left: `${point.x * 100}%`,
          top: `${point.y * 100}%`,
          '--teacher-ar-scale': settings.scale,
          opacity: settings.opacity,
        }}
        aria-hidden="true"
      >
        <i className="teacher-ar-simulator-idle" />
        <i className="teacher-ar-simulator-speaking" />
      </div>

      <div className="teacher-ar-simulator-toolbar">
        <span>{cameraActive ? 'Camera stays on this device' : 'Camera is off'}</span>
        <label>
          Size
          <input type="range" min="0.55" max="1.65" step="0.05" value={settings.scale} onChange={(event) => onSettingsChange({ scale: event.target.value })} />
        </label>
        <button type="button" className="ghost" onClick={() => setPoint(sanitizeArPoint())}>Reset position</button>
        {cameraActive && <button type="button" className="ghost" onClick={stopCamera}>Camera off</button>}
      </div>
      {error && <p className="teacher-ar-simulator-error" role="alert">{error}</p>}
    </div>
  );
}
