const API_BASE = import.meta.env.VITE_API_BASE_URL
  || (import.meta.env.PROD ? '' : 'http://localhost:3000');
const TOKEN_KEY = 'polymath_musician_auth_token';
const LEGACY_TOKEN_KEY = 'polymath_muscian_auth_token';

function retryDelay(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      window.clearTimeout(timer);
      reject(new window.DOMException('Aborted', 'AbortError'));
    }, { once: true });
  });
}

function friendlyNetworkError(error) {
  if (error?.name === 'AbortError') return error;
  const networkError = new Error('The connection was interrupted. Reconnecting may take a moment.');
  networkError.name = 'NetworkError';
  networkError.code = 'NETWORK_INTERRUPTED';
  networkError.cause = error;
  return networkError;
}

async function resilientFetch(url, options = {}, retries = 0) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await fetch(url, options);
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      lastError = error;
      if (attempt < retries) await retryDelay(350 * (attempt + 1), options.signal);
    }
  }
  throw friendlyNetworkError(lastError);
}

export function getAuthToken() {
  const current = window.localStorage.getItem(TOKEN_KEY) || '';
  if (current) return current;
  const legacy = window.localStorage.getItem(LEGACY_TOKEN_KEY) || '';
  if (legacy) {
    window.localStorage.setItem(TOKEN_KEY, legacy);
    window.localStorage.removeItem(LEGACY_TOKEN_KEY);
  }
  return legacy;
}

export function setAuthToken(token) {
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(LEGACY_TOKEN_KEY);
}

export function apiAssetUrl(path) {
  const value = String(path || '').trim();
  if (!value || /^(blob:|data:|https?:\/\/)/i.test(value)) return value;
  return `${API_BASE.replace(/\/$/, '')}/${value.replace(/^\//, '')}`;
}

export async function apiRequest(path, options = {}) {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const { networkRetries, ...fetchOptions } = options;
  const method = String(fetchOptions.method || 'GET').toUpperCase();
  const retries = Number.isFinite(Number(networkRetries))
    ? Math.max(0, Math.min(4, Number(networkRetries)))
    : (method === 'GET' || method === 'HEAD' ? 2 : 0);
  const response = await resilientFetch(`${API_BASE}${path}`, { ...fetchOptions, headers }, retries);
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const error = new Error(typeof data === 'object' && data?.error ? data.error : `Request failed (${response.status})`);
    error.status = response.status;
    error.details = data;
    throw error;
  }
  return data;
}

export async function fetchProtectedBlob(path, options = {}) {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const { networkRetries = 1, ...fetchOptions } = options;
  const response = await resilientFetch(
    `${API_BASE}${path}`,
    { ...fetchOptions, headers },
    Math.max(0, Math.min(3, Number(networkRetries) || 0)),
  );
  if (!response.ok) {
    const contentType = response.headers.get('content-type') || '';
    const data = contentType.includes('application/json')
      ? await response.json()
      : await response.text();
    const error = new Error(typeof data === 'object' && data?.error
      ? data.error
      : `Request failed (${response.status})`);
    error.status = response.status;
    error.details = data;
    throw error;
  }
  return {
    blob: await response.blob(),
    headers: response.headers,
  };
}

export async function fetchProtectedFile(path, fallbackName = 'download') {
  const token = getAuthToken();
  const response = await resilientFetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }, 2);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'File could not be loaded.');
  }
  const disposition = response.headers.get('content-disposition') || '';
  const match = disposition.match(/filename="?([^";]+)"?/i);
  return new window.File([await response.blob()], match?.[1] || fallbackName);
}

export async function downloadProtectedFile(path, fallbackName = 'download') {
  const file = await fetchProtectedFile(path, fallbackName);
  const url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = url;
  link.download = file.name;
  link.click();
  URL.revokeObjectURL(url);
}

export async function uploadProtectedArtifact(file, purpose, { onProgress } = {}) {
  const intent = await apiRequest('/api/artifact-upload-intents', {
    method: 'POST',
    body: JSON.stringify({
      purpose,
      filename: file.name,
      contentType: file.type || 'application/octet-stream',
      size: file.size,
    }),
  });
  if (!intent.direct) return null;

  await new Promise((resolve, reject) => {
    const request = new window.XMLHttpRequest();
    request.open('PUT', intent.uploadUrl, true);
    request.setRequestHeader('Content-Type', intent.contentType);
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress?.(Math.max(0, Math.min(100, Math.round((event.loaded / event.total) * 100))));
    };
    request.onerror = () => reject(new Error('The direct file upload could not reach secure storage. Try again.'));
    request.onabort = () => reject(new Error('The direct file upload was cancelled.'));
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) resolve();
      else reject(new Error(`Secure storage rejected the upload (${request.status}). Try again.`));
    };
    request.send(file);
  });

  onProgress?.(100);
  return intent;
}

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || '').split(',')[1] || '');
    reader.onerror = () => reject(reader.error || new Error('Could not read file.'));
    reader.readAsDataURL(file);
  });
}

export { API_BASE };
