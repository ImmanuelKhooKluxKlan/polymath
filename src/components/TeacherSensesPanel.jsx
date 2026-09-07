import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { createTeacherHearingMonitor } from '../engine/teacherHearingEngine.js';
import {
  WHITE_KEY_OPTIONS,
  buildKeyboardGeometry,
  detectKeyMotion,
  drawTeacherVisionOverlay,
  extractPerspectiveGray,
  findKeyboardCandidate,
  regionToCorners,
} from '../engine/teacherVisionEngine.js';

const EMPTY_HEARING = Object.freeze({
  heard: false,
  note: '—',
  frequency: 0,
  cents: 0,
  dbfs: -96,
  confidence: 0,
  notes: [],
});

const INITIAL_VISION = Object.freeze({
  detected: false,
  confidence: 0,
  pressed: [],
  calibrated: false,
  fps: 0,
});

function friendlyMediaError(error) {
  if (!window.isSecureContext) return 'Camera access requires HTTPS or localhost.';
  if (error?.name === 'NotAllowedError') return 'Camera or microphone permission was blocked. Allow both in your browser settings, then try again.';
  if (error?.name === 'NotFoundError') return 'No usable camera or microphone was found on this device.';
  if (error?.name === 'NotReadableError') return 'Another application may be using the camera or microphone.';
  return error?.message || 'The teacher could not start the camera and microphone.';
}

function stabilizePressedKeys(history, pressed) {
  const seen = new Set(pressed.map((entry) => entry.midi));
  for (const [midi, existing] of history) {
    if (seen.has(midi)) continue;
    const hits = existing.hits - 1;
    if (hits <= 0) history.delete(midi);
    else history.set(midi, { ...existing, hits });
  }
  for (const entry of pressed) {
    const existing = history.get(entry.midi);
    history.set(entry.midi, { ...entry, hits: Math.min(4, (existing?.hits || 0) + 1) });
  }
  return [...history.values()].filter((entry) => entry.hits >= 2).sort((left, right) => right.score - left.score);
}

function validKeyboardCorners(points) {
  if (points.length !== 4) return false;
  const [topLeft, topRight, bottomRight, bottomLeft] = points;
  const topAboveBottom = ((topLeft.y + topRight.y) / 2) < ((bottomLeft.y + bottomRight.y) / 2);
  const rowsPointRight = topLeft.x < topRight.x && bottomLeft.x < bottomRight.x;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += (current.x * next.y) - (next.x * current.y);
  }
  return topAboveBottom && rowsPointRight && Math.abs(twiceArea / 2) >= 0.025;
}

const TeacherSensesPanel = forwardRef(function TeacherSensesPanel({
  deviceClass = 'desktop',
  performanceTier = 'balanced',
  currentTime = 0,
  onObservation,
}, forwardedRef) {
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('Start when the camera can see the keyboard from above.');
  const [facingMode, setFacingMode] = useState('environment');
  const [cameraAspect, setCameraAspect] = useState('16 / 9');
  const [cornerMode, setCornerMode] = useState(false);
  const [cornerPoints, setCornerPoints] = useState([]);
  const [firstWhiteMidi, setFirstWhiteMidi] = useState(48);
  const [whiteKeyCount, setWhiteKeyCount] = useState(21);
  const [vision, setVision] = useState(INITIAL_VISION);
  const [hearing, setHearing] = useState(EMPTY_HEARING);

  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const analysisCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const hearingMonitorRef = useRef(null);
  const animationFrameRef = useRef(0);
  const lastVisionFrameRef = useRef(0);
  const lastDetectionRef = useRef(0);
  const detectionRef = useRef({
    detected: false,
    confidence: 0,
    region: { x: 0.04, y: 0.48, width: 0.92, height: 0.3 },
  });
  const lockedRegionRef = useRef(null);
  const lockedPointsRef = useRef(null);
  const cornerModeRef = useRef(false);
  const cornerPointsRef = useRef([]);
  const latestRegionFrameRef = useRef(null);
  const baselineRef = useRef(null);
  const pressedHistoryRef = useRef(new Map());
  const geometryRef = useRef(null);
  const hearingRef = useRef(EMPTY_HEARING);
  const currentTimeRef = useRef(currentTime);
  const onObservationRef = useRef(onObservation);
  currentTimeRef.current = currentTime;
  onObservationRef.current = onObservation;

  const geometry = useMemo(
    () => buildKeyboardGeometry({ firstWhiteMidi, whiteKeyCount }),
    [firstWhiteMidi, whiteKeyCount],
  );
  geometryRef.current = geometry;

  useEffect(() => {
    pressedHistoryRef.current.clear();
  }, [geometry]);

  const releaseMedia = useCallback(() => {
    window.cancelAnimationFrame(animationFrameRef.current);
    animationFrameRef.current = 0;
    hearingMonitorRef.current?.stop?.();
    hearingMonitorRef.current = null;
    streamRef.current?.getTracks?.().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => releaseMedia(), [releaseMedia]);

  const startVisionLoop = useCallback(() => {
    const intervalMs = performanceTier === 'full' ? 105 : performanceTier === 'balanced' ? 155 : 235;
    const analysisWidth = performanceTier === 'full' ? 320 : performanceTier === 'balanced' ? 280 : 224;

    const processFrame = (timestamp) => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const analysisCanvas = analysisCanvasRef.current;
      if (!streamRef.current || !video || !overlay || !analysisCanvas) return;
      animationFrameRef.current = window.requestAnimationFrame(processFrame);
      if (video.readyState < 2 || timestamp - lastVisionFrameRef.current < intervalMs) return;

      const previousFrameAt = lastVisionFrameRef.current;
      lastVisionFrameRef.current = timestamp;
      const ratio = video.videoWidth && video.videoHeight ? video.videoHeight / video.videoWidth : 9 / 16;
      const analysisHeight = Math.max(126, Math.round(analysisWidth * ratio));
      if (analysisCanvas.width !== analysisWidth) analysisCanvas.width = analysisWidth;
      if (analysisCanvas.height !== analysisHeight) analysisCanvas.height = analysisHeight;
      const analysisContext = analysisCanvas.getContext('2d', { willReadFrequently: true });
      analysisContext.drawImage(video, 0, 0, analysisWidth, analysisHeight);
      const frame = analysisContext.getImageData(0, 0, analysisWidth, analysisHeight);

      if (timestamp - lastDetectionRef.current > 480) {
        lastDetectionRef.current = timestamp;
        detectionRef.current = findKeyboardCandidate(frame, analysisWidth, analysisHeight);
      }

      const activeRegion = lockedRegionRef.current || detectionRef.current.region;
      const activePoints = lockedPointsRef.current || regionToCorners(activeRegion);
      const regionFrame = extractPerspectiveGray(
        frame,
        analysisWidth,
        analysisHeight,
        activePoints,
        performanceTier === 'lite' ? 196 : 252,
        performanceTier === 'lite' ? 72 : 92,
      );
      latestRegionFrameRef.current = regionFrame;
      let pressed = [];
      if (
        baselineRef.current
        && baselineRef.current.width === regionFrame.width
        && baselineRef.current.height === regionFrame.height
      ) {
        const motion = detectKeyMotion({
          baseline: baselineRef.current.gray,
          current: regionFrame.gray,
          width: regionFrame.width,
          height: regionFrame.height,
          geometry: geometryRef.current,
        });
        pressed = stabilizePressedKeys(pressedHistoryRef.current, motion.pressed);
      }

      const overlayWidth = video.videoWidth || analysisWidth;
      const overlayHeight = video.videoHeight || analysisHeight;
      if (overlay.width !== overlayWidth) overlay.width = overlayWidth;
      if (overlay.height !== overlayHeight) overlay.height = overlayHeight;
      const overlayContext = overlay.getContext('2d');
      drawTeacherVisionOverlay(overlayContext, {
        width: overlay.width,
        height: overlay.height,
        region: activeRegion,
        points: activePoints,
        manualPoints: cornerModeRef.current ? cornerPointsRef.current : [],
        geometry: geometryRef.current,
        pressed,
        calibrated: Boolean(baselineRef.current),
        detected: detectionRef.current.detected,
        confidence: detectionRef.current.confidence,
      });

      setVision({
        detected: detectionRef.current.detected,
        confidence: detectionRef.current.confidence,
        pressed,
        calibrated: Boolean(baselineRef.current),
        fps: previousFrameAt ? Math.min(30, 1000 / (timestamp - previousFrameAt)) : 0,
      });
      onObservationRef.current?.({
        capturedAt: timestamp,
        songTime: currentTimeRef.current,
        vision: {
          detected: detectionRef.current.detected,
          confidence: detectionRef.current.confidence,
          calibrated: Boolean(baselineRef.current),
          pressed,
        },
        hearing: hearingRef.current,
      });
    };

    animationFrameRef.current = window.requestAnimationFrame(processFrame);
  }, [performanceTier]);

  async function startSenses(requestedFacingMode = facingMode) {
    releaseMedia();
    setExpanded(true);
    setStatus('requesting');
    setMessage('Waiting for camera and microphone permission…');
    setHearing(EMPTY_HEARING);
    hearingRef.current = EMPTY_HEARING;
    baselineRef.current = null;
    lockedRegionRef.current = null;
    lockedPointsRef.current = null;
    cornerModeRef.current = false;
    cornerPointsRef.current = [];
    setCornerMode(false);
    setCornerPoints([]);
    pressedHistoryRef.current.clear();

    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not support camera and microphone access.');
      const compactDevice = deviceClass !== 'desktop' || performanceTier === 'lite';
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: requestedFacingMode },
          width: { ideal: compactDevice ? 720 : 1280 },
          height: { ideal: compactDevice ? 540 : 720 },
          frameRate: { ideal: compactDevice ? 18 : 24, max: 30 },
        },
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setCameraAspect(`${video.videoWidth || 16} / ${video.videoHeight || 9}`);
      setStatus('active');
      setMessage('Camera and microphone are active. Next, calibrate the empty keyboard.');
      startVisionLoop();

      try {
        hearingMonitorRef.current = await createTeacherHearingMonitor(stream, (nextHearing) => {
          hearingRef.current = nextHearing;
          setHearing(nextHearing);
        }, {
          intervalMs: performanceTier === 'lite' ? 180 : 120,
          fftSize: 4096,
        });
      } catch (audioError) {
        setMessage(`Camera is active, but hearing analysis is unavailable: ${audioError.message}`);
      }
    } catch (error) {
      releaseMedia();
      setStatus('error');
      setMessage(friendlyMediaError(error));
    }
  }

  function stopSenses() {
    releaseMedia();
    baselineRef.current = null;
    lockedRegionRef.current = null;
    lockedPointsRef.current = null;
    cornerModeRef.current = false;
    cornerPointsRef.current = [];
    latestRegionFrameRef.current = null;
    pressedHistoryRef.current.clear();
    setStatus('idle');
    setCornerMode(false);
    setCornerPoints([]);
    setVision(INITIAL_VISION);
    setHearing(EMPTY_HEARING);
    hearingRef.current = EMPTY_HEARING;
    setMessage('Eyes and ears stopped. No camera or microphone data was saved.');
  }

  function calibrateKeyboard() {
    const current = latestRegionFrameRef.current;
    if (!current) {
      setMessage('Wait until the camera preview appears, then calibrate again.');
      return;
    }
    baselineRef.current = {
      gray: current.gray.slice(),
      width: current.width,
      height: current.height,
    };
    pressedHistoryRef.current.clear();
    setVision((previous) => ({ ...previous, calibrated: true, pressed: [] }));
    setMessage('Calibrated. The teacher is now watching for movement over each mapped key.');
  }

  function beginCornerCalibration() {
    if (!active) return;
    baselineRef.current = null;
    lockedRegionRef.current = null;
    lockedPointsRef.current = null;
    pressedHistoryRef.current.clear();
    cornerModeRef.current = true;
    cornerPointsRef.current = [];
    setCornerMode(true);
    setCornerPoints([]);
    setMessage('Tap four keyboard corners: top-left, top-right, bottom-right, then bottom-left.');
  }

  function recordKeyboardCorner(event) {
    if (!cornerModeRef.current) return;
    const canvas = overlayRef.current;
    const bounds = canvas.getBoundingClientRect();
    const point = {
      x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    };
    const next = [...cornerPointsRef.current, point].slice(0, 4);
    cornerPointsRef.current = next;
    setCornerPoints(next);
    if (next.length === 4) {
      if (!validKeyboardCorners(next)) {
        cornerPointsRef.current = [];
        setCornerPoints([]);
        setMessage('Those corners crossed or were too close together. Start again at the keyboard’s top-left corner.');
        return;
      }
      lockedPointsRef.current = next;
      cornerModeRef.current = false;
      setCornerMode(false);
      setMessage('Corners saved. Keep hands off the keys, then press Calibrate empty keyboard.');
    } else {
      const labels = ['top-right', 'bottom-right', 'bottom-left'];
      setMessage(`Corner ${next.length} saved. Tap ${labels[next.length - 1]} next.`);
    }
  }

  async function switchCamera() {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    await startSenses(next);
  }

  const active = status === 'active';
  const visualNotes = vision.pressed.map((entry) => entry.note).join(', ');
  const hearingConfidence = hearing.frequency ? Math.round(hearing.confidence * 100) : 0;

  useImperativeHandle(forwardedRef, () => ({
    isActive: () => Boolean(streamRef.current),
    captureSnapshot() {
      const video = videoRef.current;
      if (!streamRef.current || !video || video.readyState < 2) {
        throw new Error('Start the teacher camera before asking her to look.');
      }
      const sourceWidth = video.videoWidth || 0;
      const sourceHeight = video.videoHeight || 0;
      if (!sourceWidth || !sourceHeight) throw new Error('Wait for the camera preview, then try Look again.');
      const width = Math.min(960, sourceWidth);
      const height = Math.max(1, Math.round(sourceHeight * (width / sourceWidth)));
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      canvas.getContext('2d').drawImage(video, 0, 0, width, height);
      return canvas.toDataURL('image/jpeg', 0.78);
    },
  }), []);

  return (
    <section className={`teacher-senses-panel is-${status}`}>
      <button
        type="button"
        className="teacher-senses-heading"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <span>
          <strong>Teacher eyes &amp; ears</strong>
          <small>Camera vision + microphone pitch</small>
        </span>
        <b>{active ? 'Live' : status === 'requesting' ? 'Starting…' : 'Open'}</b>
      </button>

      <div className="teacher-senses-content" hidden={!expanded}>
        <div className="teacher-senses-controls">
          {!active && status !== 'requesting' && (
            <button type="button" className="primary" onClick={() => startSenses()}>
              Start eyes + ears
            </button>
          )}
          {active && (
            <>
              <button type="button" className="ghost" onClick={switchCamera}>Switch camera</button>
              <button type="button" className="ghost" onClick={stopSenses}>Stop camera + mic</button>
            </>
          )}
        </div>

        <p className={`teacher-senses-message ${status === 'error' ? 'error' : ''}`} role="status">
          {message}
        </p>

        <div className="teacher-senses-grid">
          <div className="teacher-camera-column">
            <div className={`teacher-camera-preview ${active ? 'is-live' : ''}`} style={{ aspectRatio: cameraAspect }}>
              <video ref={videoRef} autoPlay muted playsInline aria-label="Live camera view used by the virtual teacher" />
              <canvas
                ref={overlayRef}
                className={cornerMode ? 'is-corner-mode' : ''}
                onPointerDown={recordKeyboardCorner}
                aria-label={cornerMode ? 'Tap the four visible keyboard corners' : undefined}
                aria-hidden={cornerMode ? undefined : 'true'}
              />
              {!active && (
                <div className="teacher-camera-placeholder">
                  <span aria-hidden="true">◎</span>
                  <strong>Camera is off</strong>
                  <small>Place the device above the visible piano keys.</small>
                </div>
              )}
            </div>
            <canvas ref={analysisCanvasRef} className="teacher-analysis-canvas" aria-hidden="true" />

            <div className="teacher-key-calibration">
              <label>
                First white key visible
                <select value={firstWhiteMidi} onChange={(event) => setFirstWhiteMidi(Number(event.target.value))}>
                  {WHITE_KEY_OPTIONS.map((option) => (
                    <option key={option.midi} value={option.midi}>{option.label}</option>
                  ))}
                </select>
              </label>
              <label>
                White keys visible
                <input
                  type="number"
                  min="3"
                  max="52"
                  value={whiteKeyCount}
                  onChange={(event) => setWhiteKeyCount(Math.max(3, Math.min(52, Number(event.target.value) || 3)))}
                />
              </label>
              <button type="button" className="primary" onClick={calibrateKeyboard} disabled={!active}>
                Calibrate empty keyboard
              </button>
              <button type="button" className="ghost teacher-corner-button" onClick={beginCornerCalibration} disabled={!active}>
                {cornerMode ? `Tap corner ${cornerPoints.length + 1} of 4` : 'Set 4 corners'}
              </button>
            </div>
          </div>

          <div className="teacher-sense-readings" aria-live="polite">
            <article>
              <span>Object</span>
              <strong>{vision.detected ? 'Piano keyboard' : active ? 'Searching…' : '—'}</strong>
              <small>{active ? `${Math.round(vision.confidence * 100)}% visual confidence` : 'Camera not started'}</small>
            </article>
            <article>
              <span>Likely pressed</span>
              <strong>{vision.calibrated ? visualNotes || 'None' : 'Calibrate first'}</strong>
              <small>{vision.calibrated ? `${geometry.whiteKeyCount} white keys mapped` : 'Use an empty keyboard baseline'}</small>
            </article>
            <article>
              <span>Dominant heard</span>
              <strong>{hearing.frequency ? hearing.note : active ? 'Listening…' : '—'}</strong>
              <small>
                {hearing.frequency
                  ? `${hearing.frequency.toFixed(1)} Hz · peaks: ${hearing.notes?.map((entry) => entry.note).join(', ') || hearing.note}`
                  : 'Play one clear note near the microphone'}
              </small>
            </article>
            <article>
              <span>Audio level</span>
              <strong>{active ? `${Math.round(hearing.dbfs)} dBFS` : '—'}</strong>
              <small>{hearing.frequency ? `${hearingConfidence}% pitch confidence` : 'Waiting for stable pitch'}</small>
            </article>
            <article>
              <span>Vision rate</span>
              <strong>{active ? `${vision.fps.toFixed(1)} fps` : '—'}</strong>
              <small>{performanceTier} device profile</small>
            </article>
          </div>
        </div>

        <p className="teacher-senses-privacy">
          Live camera frames and microphone samples stay on this device. A single snapshot is uploaded only when you press Look in teacher chat; Polymath does not save it.
        </p>
      </div>
    </section>
  );
});

export default TeacherSensesPanel;
