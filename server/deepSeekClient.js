'use strict';

const DEFAULT_BASE_URL = 'https://api.deepseek.com';
const DEFAULT_MODEL = 'deepseek-v4-flash';
const DEFAULT_TIMEOUT_MS = 45 * 1000;

function clean(value) {
  return String(value || '').trim();
}

function safeParameters(parameters = {}) {
  const allowed = [
    'temperature',
    'top_p',
    'max_tokens',
    'presence_penalty',
    'frequency_penalty',
    'stop',
    'response_format',
    'tools',
    'tool_choice',
  ];
  return Object.fromEntries(
    allowed
      .filter((key) => parameters[key] !== undefined)
      .map((key) => [key, parameters[key]]),
  );
}

function createDeepSeekClient(options = {}) {
  const apiKey = clean(options.apiKey || process.env.DEEPSEEK_API_KEY);
  const model = clean(options.model || process.env.DEEPSEEK_MODEL || DEFAULT_MODEL);
  const baseUrl = clean(options.baseUrl || process.env.DEEPSEEK_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, '');
  const timeoutMs = Number(options.timeoutMs || process.env.DEEPSEEK_TIMEOUT_MS || DEFAULT_TIMEOUT_MS);
  const requestFetch = options.fetch || globalThis.fetch;

  if (!apiKey) throw new Error('DEEPSEEK_API_KEY is required.');
  if (!model) throw new Error('DEEPSEEK_MODEL is required.');
  if (!/^https:\/\//i.test(baseUrl)) throw new Error('DEEPSEEK_BASE_URL must use HTTPS.');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('DEEPSEEK_TIMEOUT_MS must be positive.');
  if (typeof requestFetch !== 'function') throw new Error('A fetch implementation is required.');

  async function chat(messages, parameters = {}) {
    if (!Array.isArray(messages) || !messages.length) {
      throw new Error('messages must be a non-empty array.');
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(timeoutMs, DEFAULT_TIMEOUT_MS));
    try {
      const response = await requestFetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages,
          stream: false,
          thinking: { type: 'disabled' },
          ...safeParameters(parameters),
        }),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.error?.message || body?.error || body?.message || response.statusText;
        throw new Error(`DeepSeek request failed (${response.status}): ${detail}`);
      }
      return body;
    } finally {
      clearTimeout(timer);
    }
  }

  return Object.freeze({ provider: 'deepseek', baseUrl, model, chat });
}

module.exports = {
  createDeepSeekClient,
  safeParameters,
};
