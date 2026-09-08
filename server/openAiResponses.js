'use strict';

const DEFAULT_CHAT_MODEL = 'gpt-5.6-terra';
const DEFAULT_MUSIC_MODEL = 'gpt-6-astra';
const DEFAULT_TIMEOUT_MS = 50 * 1000;
const CONTROL_TIMEOUT_MS = 20 * 1000;
const ACTIVE_STATUSES = new Set(['queued', 'in_progress']);

function clean(value, maximum = 1000) {
  return String(value || '').trim().slice(0, maximum);
}

function positiveInteger(value, fallback, minimum = 16, maximum = 100000) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(maximum, Math.floor(number)));
}

function normalizeReasoningEffort(value, fallback = 'low') {
  const effort = clean(value, 20).toLowerCase();
  return ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'].includes(effort)
    ? effort
    : fallback;
}

function normalizeContent(content, role) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return clean(content, 20000);
  return content.map((part) => {
    if (typeof part === 'string') {
      return { type: role === 'assistant' ? 'output_text' : 'input_text', text: part };
    }
    if (!part || typeof part !== 'object') return null;
    if (part.type === 'image_url') {
      const imageUrl = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
      if (!imageUrl) return null;
      return {
        type: 'input_image',
        image_url: imageUrl,
        ...(part.image_url?.detail || part.detail ? { detail: part.image_url?.detail || part.detail } : {}),
      };
    }
    if (part.type === 'text') {
      return { type: role === 'assistant' ? 'output_text' : 'input_text', text: clean(part.text, 20000) };
    }
    if (['input_text', 'output_text'].includes(part.type) && typeof part.text === 'string') {
      return { type: part.type, text: part.text };
    }
    if (part.type === 'input_image' && typeof part.image_url === 'string') {
      return {
        type: 'input_image',
        image_url: part.image_url,
        ...(part.detail ? { detail: part.detail } : {}),
      };
    }
    return null;
  }).filter(Boolean);
}

function prepareInput(messages) {
  if (!Array.isArray(messages) || !messages.length) {
    throw new Error('messages must be a non-empty array.');
  }
  const instructions = [];
  const input = [];
  for (const message of messages) {
    const role = ['assistant', 'system', 'developer'].includes(message?.role)
      ? message.role
      : 'user';
    if (role === 'system' || role === 'developer') {
      const instruction = clean(message?.content, 50000);
      if (instruction) instructions.push(instruction);
      continue;
    }
    const content = normalizeContent(message?.content, role);
    if ((typeof content === 'string' && content.trim()) || (Array.isArray(content) && content.length)) {
      input.push({ role, content });
    }
  }
  if (!input.length) throw new Error('messages must include user or assistant content.');
  return { instructions: instructions.join('\n\n'), input };
}

function extractOutputText(body) {
  if (typeof body === 'string') return body.trim();
  if (Array.isArray(body)) return body.map(extractOutputText).filter(Boolean).join('').trim();
  if (!body || typeof body !== 'object') return '';
  if (typeof body.output_text === 'string') return body.output_text.trim();
  if (typeof body.text === 'string') return body.text.trim();
  if (Array.isArray(body.text)) return body.text.map((part) => String(part || '')).join('').trim();
  if (Array.isArray(body.output)) {
    const pieces = [];
    for (const item of body.output) {
      if (item?.type === 'message' && Array.isArray(item.content)) {
        for (const part of item.content) {
          if (part?.type === 'output_text' && typeof part.text === 'string') pieces.push(part.text);
          if (part?.type === 'refusal' && typeof part.refusal === 'string') pieces.push(part.refusal);
        }
      }
    }
    if (pieces.length) return pieces.join('').trim();
  }
  if (body.output !== undefined) return extractOutputText(body.output);
  const chatCompletionText = body?.choices?.[0]?.message?.content;
  if (typeof chatCompletionText === 'string') return chatCompletionText.trim();
  return '';
}

function normalizedStatus(value) {
  const status = clean(value, 30).toLowerCase();
  if (status === 'queued') return 'IN_QUEUE';
  if (status === 'in_progress') return 'IN_PROGRESS';
  if (status === 'completed') return 'COMPLETED';
  if (status === 'cancelled') return 'CANCELLED';
  if (status === 'failed' || status === 'incomplete') return 'FAILED';
  return status ? status.toUpperCase() : 'UNKNOWN';
}

function providerError(body) {
  if (!body || typeof body !== 'object') return '';
  return clean(
    body.error?.message
      || body.error
      || body.incomplete_details?.reason
      || body.message,
    600,
  );
}

function normalizeResponse(body) {
  const status = normalizedStatus(body?.status);
  const text = extractOutputText(body);
  return {
    id: clean(body?.id, 220),
    status,
    output: { text: text ? [text] : [] },
    error: ['FAILED', 'CANCELLED'].includes(status) ? providerError(body) : '',
    usage: body?.usage && typeof body.usage === 'object' ? {
      inputTokens: Number(body.usage.input_tokens) || 0,
      outputTokens: Number(body.usage.output_tokens) || 0,
      totalTokens: Number(body.usage.total_tokens) || 0,
    } : null,
  };
}

function createOpenAiResponsesClient(options = {}) {
  const apiKey = clean(options.apiKey || process.env.OPENAI_API_KEY, 1000);
  const model = clean(options.model || process.env.OPENAI_CHAT_MODEL || DEFAULT_CHAT_MODEL, 160);
  const baseUrl = clean(options.baseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1', 500)
    .replace(/\/+$/, '');
  const organization = clean(options.organization || process.env.OPENAI_ORGANIZATION, 200);
  const project = clean(options.project || process.env.OPENAI_PROJECT, 200);
  const timeoutMs = positiveInteger(
    options.timeoutMs || process.env.OPENAI_TIMEOUT_MS,
    DEFAULT_TIMEOUT_MS,
    1000,
    5 * 60 * 1000,
  );
  const defaultReasoningEffort = normalizeReasoningEffort(
    options.reasoningEffort || process.env.OPENAI_REASONING_EFFORT,
    'low',
  );
  const requestFetch = options.fetch || globalThis.fetch;

  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');
  if (!model) throw new Error('An OpenAI model is required.');
  if (typeof requestFetch !== 'function') throw new Error('A fetch implementation is required.');

  async function requestJson(pathname, requestOptions = {}, maximumMs = CONTROL_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(timeoutMs, maximumMs));
    try {
      const response = await requestFetch(`${baseUrl}${pathname}`, {
        ...requestOptions,
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          ...(organization ? { 'OpenAI-Organization': organization } : {}),
          ...(project ? { 'OpenAI-Project': project } : {}),
          ...(requestOptions.headers || {}),
        },
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(`OpenAI request failed (${response.status}): ${providerError(body) || response.statusText || 'Unknown error'}`);
        error.code = response.status === 429 ? 'OPENAI_RATE_LIMITED' : 'OPENAI_REQUEST_FAILED';
        error.status = response.status;
        throw error;
      }
      return body;
    } catch (error) {
      if (error?.name === 'AbortError') {
        const timeoutError = new Error('OpenAI did not respond before the server timeout.');
        timeoutError.code = 'OPENAI_TIMEOUT';
        throw timeoutError;
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function responsePayload(messages, parameters = {}, background = false) {
    const prepared = prepareInput(messages);
    const requestedEffort = parameters.reasoning?.effort
      || parameters.reasoning_effort
      || parameters.reasoningEffort;
    const payload = {
      model,
      input: prepared.input,
      background,
      store: background,
      reasoning: {
        effort: normalizeReasoningEffort(requestedEffort, defaultReasoningEffort),
      },
      max_output_tokens: positiveInteger(
        parameters.max_output_tokens || parameters.max_tokens,
        800,
        16,
        100000,
      ),
    };
    if (prepared.instructions) payload.instructions = prepared.instructions;
    if (parameters.text && typeof parameters.text === 'object') payload.text = parameters.text;
    if (parameters.metadata && typeof parameters.metadata === 'object') payload.metadata = parameters.metadata;
    if (clean(parameters.prompt_cache_key, 100)) payload.prompt_cache_key = clean(parameters.prompt_cache_key, 100);
    return payload;
  }

  async function chat(messages, parameters = {}) {
    const body = await requestJson('/responses', {
      method: 'POST',
      body: JSON.stringify(responsePayload(messages, parameters, false)),
    }, timeoutMs);
    const normalized = normalizeResponse(body);
    if (normalized.status !== 'COMPLETED') {
      const error = new Error(normalized.error || `OpenAI response ended with status ${normalized.status}.`);
      error.code = 'OPENAI_INCOMPLETE';
      throw error;
    }
    return normalized;
  }

  async function submit(messages, parameters = {}) {
    const body = await requestJson('/responses', {
      method: 'POST',
      body: JSON.stringify(responsePayload(messages, parameters, true)),
    });
    if (!body?.id) throw new Error('OpenAI accepted the request without returning a response ID.');
    return { id: body.id, status: normalizedStatus(body.status) };
  }

  function validateResponseId(responseId) {
    const id = clean(responseId, 220);
    if (!/^resp_[A-Za-z0-9_-]{8,210}$/.test(id)) throw new Error('Invalid OpenAI response ID.');
    return id;
  }

  async function status(responseId) {
    const id = validateResponseId(responseId);
    const body = await requestJson(`/responses/${encodeURIComponent(id)}`, { method: 'GET' });
    return normalizeResponse(body);
  }

  async function cancel(responseId) {
    const id = validateResponseId(responseId);
    const body = await requestJson(`/responses/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
    return normalizeResponse(body);
  }

  return Object.freeze({
    baseUrl,
    model,
    chat,
    submit,
    status,
    cancel,
    capabilities: Object.freeze({ background: true, responsesApi: true }),
  });
}

module.exports = {
  ACTIVE_STATUSES,
  DEFAULT_CHAT_MODEL,
  DEFAULT_MUSIC_MODEL,
  createOpenAiResponsesClient,
  extractOutputText,
  normalizeResponse,
  normalizedStatus,
  prepareInput,
};
