const STORAGE_KEY = 'polymath-device-performance-v1';
const PROFILE_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

export const PERFORMANCE_TIERS = Object.freeze({
  lite: Object.freeze({ label: 'Lite', visualFps: 30, maximumNotes: 120 }),
  balanced: Object.freeze({ label: 'Balanced', visualFps: 45, maximumNotes: 320 }),
  full: Object.freeze({ label: 'Full', visualFps: 60, maximumNotes: 820 }),
});

const TIER_ORDER = ['lite', 'balanced', 'full'];

export const DEVICE_CLASSES = Object.freeze({
  phone: Object.freeze({ label: 'Phone', maximumTier: 'balanced' }),
  tablet: Object.freeze({ label: 'Tablet', maximumTier: 'balanced' }),
  desktop: Object.freeze({ label: 'Computer', maximumTier: 'full' }),
});

export function normalizePerformanceTier(value, fallback = 'balanced') {
  return Object.prototype.hasOwnProperty.call(PERFORMANCE_TIERS, value) ? value : fallback;
}

export function lowerPerformanceTier(value) {
  const index = TIER_ORDER.indexOf(normalizePerformanceTier(value));
  return TIER_ORDER[Math.max(0, index - 1)];
}

export function raisePerformanceTier(value) {
  const index = TIER_ORDER.indexOf(normalizePerformanceTier(value));
  return TIER_ORDER[Math.min(TIER_ORDER.length - 1, index + 1)];
}

export function isCoarsePointerDevice() {
  return typeof window !== 'undefined'
    && window.matchMedia('(pointer: coarse)').matches;
}

export function detectDeviceClass() {
  if (typeof window === 'undefined') return 'desktop';
  const coarsePointer = isCoarsePointerDevice();
  const shortestSide = Math.min(
    Number(window.screen?.width) || Number(window.innerWidth) || 0,
    Number(window.screen?.height) || Number(window.innerHeight) || 0,
  );
  const longestSide = Math.max(
    Number(window.screen?.width) || Number(window.innerWidth) || 0,
    Number(window.screen?.height) || Number(window.innerHeight) || 0,
  );
  if (coarsePointer && shortestSide > 0 && shortestSide <= 520) return 'phone';
  if (coarsePointer || (shortestSide > 0 && shortestSide <= 1024 && longestSide <= 1440)) {
    return 'tablet';
  }
  return 'desktop';
}

export function capTierForDevice(tier, deviceClass = detectDeviceClass()) {
  const normalized = normalizePerformanceTier(tier);
  const maximumTier = DEVICE_CLASSES[deviceClass]?.maximumTier || 'balanced';
  return TIER_ORDER.indexOf(normalized) > TIER_ORDER.indexOf(maximumTier)
    ? maximumTier
    : normalized;
}

function deviceSignature() {
  if (typeof window === 'undefined') return 'server';
  const width = Math.round(Math.min(window.screen?.width || 0, window.screen?.height || 0));
  const height = Math.round(Math.max(window.screen?.width || 0, window.screen?.height || 0));
  return [
    navigator.platform || 'unknown',
    navigator.hardwareConcurrency || 'unknown',
    width,
    height,
    Math.round((window.devicePixelRatio || 1) * 10) / 10,
  ].join(':');
}

export function readSavedPerformanceProfile() {
  if (typeof window === 'undefined') return null;
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY));
    if (!saved || saved.signature !== deviceSignature()) return null;
    if (Date.now() - Number(saved.updatedAt || 0) > PROFILE_MAX_AGE_MS) return null;
    const deviceClass = detectDeviceClass();
    return {
      ...saved,
      deviceClass,
      tier: capTierForDevice(saved.tier, deviceClass),
    };
  } catch {
    return null;
  }
}

export function getInitialPerformanceTier() {
  const saved = readSavedPerformanceProfile();
  if (saved?.tier) return saved.tier;
  const deviceClass = detectDeviceClass();
  if (deviceClass === 'phone') return 'lite';
  if (deviceClass === 'tablet') return 'balanced';
  return 'full';
}

export function savePerformanceProfile(tier, measurements = {}) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      tier: capTierForDevice(tier),
      deviceClass: detectDeviceClass(),
      signature: deviceSignature(),
      updatedAt: Date.now(),
      measurements,
    }));
  } catch {
    // Private browsing and full storage can reject localStorage writes.
  }
}

function percentile(values, amount) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * amount))];
}

function measureCpuWork() {
  const startedAt = performance.now();
  let state = 0x12345678;
  for (let index = 0; index < 220000; index += 1) {
    state = Math.imul(state ^ index, 2654435761) >>> 0;
    state = ((state << 7) | (state >>> 25)) >>> 0;
  }
  return { milliseconds: performance.now() - startedAt, checksum: state };
}

export async function calibrateDevice({ durationMs = 3000, onProgress } = {}) {
  if (typeof window === 'undefined' || document.visibilityState === 'hidden') {
    const deviceClass = detectDeviceClass();
    return {
      tier: getInitialPerformanceTier(),
      deviceClass,
      deviceLabel: DEVICE_CLASSES[deviceClass].label,
      skipped: true,
      reason: 'page-hidden',
    };
  }

  measureCpuWork();
  const cpu = measureCpuWork();
  const frameDeltas = [];
  const startedAt = performance.now();

  await new Promise((resolve) => {
    let previousFrame = 0;
    function frame(now) {
      if (previousFrame) frameDeltas.push(now - previousFrame);
      previousFrame = now;
      if (typeof onProgress === 'function') {
        onProgress(Math.min(1, Math.max(0, (now - startedAt) / durationMs)));
      }
      if (now - startedAt >= durationMs) {
        if (typeof onProgress === 'function') onProgress(1);
        resolve();
        return;
      }
      window.requestAnimationFrame(frame);
    }
    window.requestAnimationFrame(frame);
  });

  const medianFrameMs = percentile(frameDeltas, 0.5);
  const p95FrameMs = percentile(frameDeltas, 0.95);
  const averageFrameMs = frameDeltas.length
    ? frameDeltas.reduce((sum, value) => sum + value, 0) / frameDeltas.length
    : 100;
  const averageFps = Math.min(120, 1000 / Math.max(averageFrameMs, 1));
  const memoryGb = Number(navigator.deviceMemory) || null;
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const effectiveConnection = connection?.effectiveType || null;
  const coarsePointer = isCoarsePointerDevice();
  const deviceClass = detectDeviceClass();

  let tier = deviceClass === 'phone'
    ? 'lite'
    : deviceClass === 'tablet'
      ? 'balanced'
      : 'full';
  const clearlyStruggling = averageFps < 38
    || p95FrameMs > 42
    || cpu.milliseconds > 22
    || (memoryGb !== null && memoryGb <= 2);
  const clearlyStrong = averageFps >= 55
    && p95FrameMs <= 25
    && cpu.milliseconds <= 5
    && (memoryGb === null || memoryGb >= 6);

  if (clearlyStruggling) tier = 'lite';
  else if (clearlyStrong) tier = deviceClass === 'desktop' ? 'full' : 'balanced';
  if (effectiveConnection === 'slow-2g' || effectiveConnection === '2g') {
    tier = lowerPerformanceTier(tier);
  }
  tier = capTierForDevice(tier, deviceClass);

  return {
    tier,
    skipped: false,
    averageFps: Math.round(averageFps * 10) / 10,
    medianFrameMs: Math.round(medianFrameMs * 10) / 10,
    p95FrameMs: Math.round(p95FrameMs * 10) / 10,
    cpuWorkMs: Math.round(cpu.milliseconds * 10) / 10,
    logicalProcessors: Number(navigator.hardwareConcurrency) || null,
    memoryGb,
    effectiveConnection,
    coarsePointer,
    deviceClass,
    deviceLabel: DEVICE_CLASSES[deviceClass].label,
  };
}

export function refineTierFromLoading(tier, loading = {}) {
  const current = normalizePerformanceTier(tier);
  const decodedSamples = Number(loading.decodedSamples) || 0;
  const averageDecodeMs = decodedSamples
    ? Number(loading.decodeMs || 0) / decodedSamples
    : 0;
  const averageLoadMs = decodedSamples
    ? Number(loading.wallMs || 0) / decodedSamples
    : 0;

  if (averageDecodeMs > 35 || averageLoadMs > 900 || Number(loading.wallMs) > 30000) {
    return lowerPerformanceTier(current);
  }
  return current;
}

export function visualFrameInterval(tier) {
  return 1000 / PERFORMANCE_TIERS[normalizePerformanceTier(tier)].visualFps;
}
