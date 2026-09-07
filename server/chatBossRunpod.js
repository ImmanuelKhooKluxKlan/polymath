'use strict';

const DEFAULT_MODEL = 'polymath-chat-boss';
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const EDGE_SAFE_TIMEOUT_MS = 45 * 1000;
const CONTROL_REQUEST_TIMEOUT_MS = 20 * 1000;

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
  const nativeBaseUrl = `https://api.runpod.ai/v2/${encodeURIComponent(endpointId)}`;

  async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(timeoutMs, CONTROL_REQUEST_TIMEOUT_MS));
    try {
      const response = await requestFetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          ...(options.body ? { 'Content-Type': 'application/json' } : {}),
          ...(options.headers || {}),
        },
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = body?.error?.message || body?.error || body?.message || response.statusText;
        throw new Error(`RunPod Chat Boss request failed (${response.status}): ${detail}`);
      }
      return body;
    } finally {
      clearTimeout(timer);
    }
  }

  async function chat(messages, parameters = {}) {
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error('messages must be a non-empty array.');
    }

    const controller = new AbortController();
    // Synchronous callers must receive a JSON error before the 60-second ALB
    // idle timeout. Long-running teacher replies use submit/status instead.
    const timeout = setTimeout(() => controller.abort(), Math.min(timeoutMs, EDGE_SAFE_TIMEOUT_MS));
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

  async function submit(messages, samplingParams = {}) {
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error('messages must be a non-empty array.');
    }
    const body = await requestJson(`${nativeBaseUrl}/run`, {
      method: 'POST',
      body: JSON.stringify({
        input: {
          // Use RunPod's OpenAI passthrough shape here. The legacy
          // messages/sampling_params shorthand discards OpenAI-only fields
          // such as chat_template_kwargs, which leaves Qwen thinking mode on.
          openai_route: '/v1/chat/completions',
          openai_input: {
            model,
            messages,
            temperature: 0.7,
            top_p: 0.8,
            max_tokens: 1024,
            ...samplingParams,
            stream: false,
            chat_template_kwargs: {
              ...(samplingParams.chat_template_kwargs || {}),
              enable_thinking: false,
            },
          },
        },
        policy: {
          executionTimeout: Math.min(timeoutMs, 15 * 60 * 1000),
          ttl: Math.max(timeoutMs + (15 * 60 * 1000), 30 * 60 * 1000),
        },
      }),
    });
    if (!body?.id) throw new Error('RunPod accepted the request without returning a job ID.');
    return { id: body.id, status: body.status || 'IN_QUEUE' };
  }

  function validateJobId(jobId) {
    const id = clean(jobId);
    if (!/^[A-Za-z0-9_-]{8,160}$/.test(id)) throw new Error('Invalid RunPod job ID.');
    return id;
  }

  async function status(jobId) {
    const id = validateJobId(jobId);
    return requestJson(`${nativeBaseUrl}/status/${encodeURIComponent(id)}`, { method: 'GET' });
  }

  async function cancel(jobId) {
    const id = validateJobId(jobId);
    return requestJson(`${nativeBaseUrl}/cancel/${encodeURIComponent(id)}`, { method: 'POST' });
  }

  return Object.freeze({ baseUrl, nativeBaseUrl, model, chat, submit, status, cancel });
}

module.exports = {
  createChatBossRunpodClient,
};
