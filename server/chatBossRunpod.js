'use strict';

const DEFAULT_MODEL = 'polymath-chat-boss';
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;

function clean(value) {
  return String(value || '').trim();
}

function createChatBossRunpodClient(options = {}) {
  const endpointId = clean(options.endpointId || process.env.RUNPOD_CHAT_BOSS_ENDPOINT_ID);
  const apiKey = clean(options.apiKey || process.env.RUNPOD_API_KEY);
  const model = clean(options.model || process.env.RUNPOD_CHAT_BOSS_MODEL || DEFAULT_MODEL);
  const requestFetch = options.fetch || globalThis.fetch;
  const timeoutMs = Number(
    options.timeoutMs || process.env.RUNPOD_CHAT_BOSS_TIMEOUT_MS || DEFAULT_TIMEOUT_MS,
  );

  if (!endpointId) throw new Error('RUNPOD_CHAT_BOSS_ENDPOINT_ID is required.');
  if (!apiKey) throw new Error('RUNPOD_API_KEY is required.');
  if (typeof requestFetch !== 'function') throw new Error('A fetch implementation is required.');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error('RUNPOD_CHAT_BOSS_TIMEOUT_MS must be a positive number.');
  }

  const baseUrl = `https://api.runpod.ai/v2/${encodeURIComponent(endpointId)}/openai/v1`;

  async function chat(messages, parameters = {}) {
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error('messages must be a non-empty array.');
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
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
          temperature: 0.55,
          top_p: 0.85,
          max_tokens: 640,
          chat_template_kwargs: { enable_thinking: false },
          ...parameters,
        }),
        signal: controller.signal,
      });

      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.error?.message || body?.error || body?.message || response.statusText;
        throw new Error(`RunPod ChatBoss request failed (${response.status}): ${detail}`);
      }
      return body;
    } finally {
      clearTimeout(timeout);
    }
  }

  return Object.freeze({ baseUrl, model, chat });
}

module.exports = { createChatBossRunpodClient };
