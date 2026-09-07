export const TEACHER_AR_DEFAULTS = Object.freeze({
  style: 'regular',
  scale: 1,
  opacity: 0.94,
  pianoSide: 'right',
  speaking: false,
});

export const TEACHER_AR_HARDWARE_REQUIREMENTS = Object.freeze([
  'WebXR immersive-ar support in the glasses browser',
  '6DoF head tracking, not rotation-only 3DoF',
  'Transparent or camera-passthrough view of the real room',
  'WebGL rendering and spatial hit testing or plane detection',
  'A select input: hand tracking, controller, touch or gaze click',
]);

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function sanitizeTeacherArSettings(input = {}) {
  return {
    style: input.style === 'bikini' ? 'bikini' : 'regular',
    scale: clamp(finiteNumber(input.scale, TEACHER_AR_DEFAULTS.scale), 0.55, 1.65),
    opacity: clamp(finiteNumber(input.opacity, TEACHER_AR_DEFAULTS.opacity), 0.35, 1),
    pianoSide: ['left', 'right', 'behind'].includes(input.pianoSide) ? input.pianoSide : 'right',
    speaking: Boolean(input.speaking),
  };
}

export function sanitizeArPoint(input = {}, fallback = { x: 0.72, y: 0.68 }) {
  return {
    x: clamp(finiteNumber(input.x, fallback.x), 0.08, 0.92),
    y: clamp(finiteNumber(input.y, fallback.y), 0.14, 0.9),
  };
}

export function pointFromStageTap(event, bounds) {
  if (!bounds?.width || !bounds?.height) return sanitizeArPoint();
  return sanitizeArPoint({
    x: (finiteNumber(event?.clientX, bounds.left) - bounds.left) / bounds.width,
    y: (finiteNumber(event?.clientY, bounds.top) - bounds.top) / bounds.height,
  });
}

export function teacherArImageUrl(style) {
  return style === 'bikini'
    ? '/assets/virtual-piano-teacher-sprite-resort-v1.png'
    : '/assets/virtual-piano-teacher-sprite-transparent-v2.png';
}

export async function detectTeacherArCapabilities(options = {}) {
  const navigatorRef = options.navigatorRef || globalThis.navigator;
  const windowRef = options.windowRef || globalThis.window;
  const secureContext = options.secureContext ?? Boolean(windowRef?.isSecureContext);
  const xrApi = Boolean(navigatorRef?.xr?.isSessionSupported);
  let spatialAr = false;
  let xrError = '';

  if (secureContext && xrApi) {
    try {
      spatialAr = Boolean(await navigatorRef.xr.isSessionSupported('immersive-ar'));
    } catch (error) {
      xrError = String(error?.message || error || 'WebXR capability check failed.');
    }
  }

  const cameraPreview = Boolean(
    secureContext
    && navigatorRef?.mediaDevices?.getUserMedia,
  );

  return Object.freeze({
    secureContext,
    xrApi,
    spatialAr,
    cameraPreview,
    recommendedMode: spatialAr ? 'spatial' : cameraPreview ? 'simulator' : 'unavailable',
    xrError,
  });
}

export function teacherArRoute({ style = 'regular', channel = '', simulate = false } = {}) {
  const params = new URLSearchParams({ style: style === 'bikini' ? 'bikini' : 'regular' });
  if (channel) params.set('channel', String(channel).slice(0, 80));
  if (simulate) params.set('simulate', '1');
  return `teacher-ar?${params.toString()}`;
}
