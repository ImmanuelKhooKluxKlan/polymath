const INTERNAL_DETAIL = /(?:openai|runpod|backend|server|worker|endpoint|provider|response api|json blueprint|schema|parse error|database|inference|secure storage|econn|http status|request failed \(\d+\)|\b(?:model|gpu|api|configuration|configured)\b)/i;

export function userFacingError(error, fallback = 'Something went wrong. Please try again.') {
  const message = String(error?.message || error || '').trim();
  const status = Number(error?.status);
  if (!message || INTERNAL_DETAIL.test(message) || status >= 500) return fallback;
  return message;
}
