const API_BASE = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:3000';
const TOKEN_KEY = 'polymath_musician_auth_token';
const LEGACY_TOKEN_KEY = 'polymath_muscian_auth_token';

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

export async function apiRequest(path, options = {}) {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
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

export async function fetchProtectedFile(path, fallbackName = 'download') {
  const token = getAuthToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
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

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || '').split(',')[1] || '');
    reader.onerror = () => reject(reader.error || new Error('Could not read file.'));
    reader.readAsDataURL(file);
  });
}

export { API_BASE };
