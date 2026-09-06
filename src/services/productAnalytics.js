import { API_BASE, getAuthToken } from './api.js';

const ANONYMOUS_KEY = 'polymath-product-anonymous-v1';
const SESSION_KEY = 'polymath-product-session-v1';
const QUEUE_KEY = 'polymath-product-event-queue-v1';
const DISABLED_KEY = 'polymath-product-analytics-disabled';
const SAFE_PROPERTIES = new Set([
  'audience', 'deviceClass', 'durationMs', 'durationSeconds', 'execution', 'feedback',
  'freePreview', 'hand', 'inputMode', 'instrument', 'interval', 'level', 'noteCount',
  'outcome', 'page', 'performanceTier', 'plan', 'playbackMode', 'productId', 'qualityScore',
  'refunded', 'restored', 'score', 'signedIn', 'sizeBucket', 'sourceKind', 'tier',
]);
const MAX_QUEUE = 100;
let queue = [];
let flushTimer = null;
let installed = false;

function browserAvailable() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

function randomId(prefix) {
  const value = globalThis.crypto?.randomUUID?.()
    || `${Date.now()}_${Math.random().toString(36).slice(2)}`;
  return `${prefix}_${value}`;
}

function storedId(storage, key, prefix) {
  try {
    const existing = storage.getItem(key);
    if (existing) return existing;
    const created = randomId(prefix);
    storage.setItem(key, created);
    return created;
  } catch {
    return randomId(prefix);
  }
}

function analyticsDisabled() {
  if (!browserAvailable()) return true;
  if (window.navigator?.globalPrivacyControl === true) return true;
  if (String(window.navigator?.doNotTrack || '') === '1') return true;
  try {
    return window.localStorage.getItem(DISABLED_KEY) === 'true';
  } catch {
    return false;
  }
}

export function sanitizeProductProperties(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const output = {};
  for (const [key, candidate] of Object.entries(value)) {
    if (!SAFE_PROPERTIES.has(key)) continue;
    if (typeof candidate === 'boolean') output[key] = candidate;
    else if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      output[key] = Math.max(-1_000_000_000, Math.min(1_000_000_000, candidate));
    } else if (typeof candidate === 'string') {
      output[key] = [...candidate.trim()]
        .filter((character) => {
          const code = character.charCodeAt(0);
          return code >= 32 && code !== 127;
        })
        .join('')
        .slice(0, 80);
    }
  }
  return output;
}

export function uploadSizeBucket(bytes) {
  const size = Math.max(0, Number(bytes) || 0);
  if (size < 1024 * 1024) return '<1MB';
  if (size < 10 * 1024 * 1024) return '1-10MB';
  if (size < 100 * 1024 * 1024) return '10-100MB';
  if (size < 500 * 1024 * 1024) return '100-500MB';
  return '500MB+';
}

function currentPath() {
  if (!browserAvailable()) return '';
  return String(window.location.hash || window.location.pathname || '')
    .replace(/^#/, '')
    .split('?')[0]
    .slice(0, 100);
}

function releaseName() {
  return String(import.meta.env?.VITE_BUILD_SHA || import.meta.env?.VITE_COMMIT_SHA || 'web').slice(0, 80);
}

function persistQueue() {
  if (!browserAvailable()) return;
  try {
    window.localStorage.setItem(QUEUE_KEY, JSON.stringify(queue.slice(-MAX_QUEUE)));
  } catch {
    // Delivery remains best-effort for this browser session.
  }
}

function restoreQueue() {
  if (!browserAvailable() || queue.length) return;
  try {
    const stored = JSON.parse(window.localStorage.getItem(QUEUE_KEY) || '[]');
    if (Array.isArray(stored)) queue = stored.slice(-MAX_QUEUE);
  } catch {
    queue = [];
  }
}

function scheduleFlush(delay = 1200) {
  if (!browserAvailable() || flushTimer) return;
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    void flushProductEvents();
  }, delay);
}

export function trackProductEvent(eventName, properties = {}) {
  if (analyticsDisabled()) return false;
  restoreQueue();
  const anonymousId = storedId(window.localStorage, ANONYMOUS_KEY, 'anon');
  const sessionId = storedId(window.sessionStorage, SESSION_KEY, 'session');
  queue.push({
    eventId: randomId('event'),
    eventName: String(eventName || '').trim().toLowerCase(),
    occurredAt: new Date().toISOString(),
    anonymousId,
    sessionId,
    path: currentPath(),
    release: releaseName(),
    properties: sanitizeProductProperties(properties),
  });
  if (queue.length > MAX_QUEUE) queue = queue.slice(-MAX_QUEUE);
  persistQueue();
  scheduleFlush();
  return true;
}

export async function flushProductEvents() {
  if (analyticsDisabled()) return false;
  restoreQueue();
  if (!queue.length) return true;
  const batch = queue.splice(0, 20);
  persistQueue();
  const token = getAuthToken();
  try {
    const response = await fetch(`${API_BASE}/api/product-events`, {
      method: 'POST',
      keepalive: true,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ events: batch }),
    });
    if (!response.ok) throw new Error(`Product event delivery failed (${response.status}).`);
    if (queue.length) scheduleFlush(250);
    return true;
  } catch {
    queue = [...batch, ...queue].slice(0, MAX_QUEUE);
    persistQueue();
    scheduleFlush(10000);
    return false;
  }
}

export function installProductAnalytics() {
  if (!browserAvailable() || installed || analyticsDisabled()) return () => {};
  installed = true;
  restoreQueue();
  const flush = () => { void flushProductEvents(); };
  window.addEventListener('online', flush);
  window.addEventListener('pagehide', flush);
  if (queue.length) scheduleFlush(250);
  return () => {
    window.removeEventListener('online', flush);
    window.removeEventListener('pagehide', flush);
    installed = false;
  };
}
