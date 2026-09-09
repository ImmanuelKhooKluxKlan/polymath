'use strict';

const {
  DEFAULT_CHAT_MODEL,
  createOpenAiResponsesClient,
} = require('./openAiResponses');
const {
  OWNER_CONTEXT,
  PROMPT_VERSIONS,
  SYSTEMS,
} = require('./assistantBehavior');

const MAX_MESSAGES = 24;
const MAX_MESSAGE_CHARS = 6000;
const ACTIVE_STATUSES = new Set(['IN_QUEUE', 'IN_PROGRESS']);
const FINISHED_STATUSES = new Set(['COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED']);

function clean(value) {
  return String(value || '').trim();
}

function sanitizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .slice(-MAX_MESSAGES)
    .map((message) => ({
      role: message?.role === 'assistant' ? 'assistant' : 'user',
      content: clean(message?.content).slice(0, MAX_MESSAGE_CHARS),
    }))
    .filter((message) => message.content);
}

function textFromOutput(output) {
  if (typeof output === 'string') return output.trim();
  if (Array.isArray(output)) {
    return output.map(textFromOutput).filter(Boolean).join('').trim();
  }
  if (!output || typeof output !== 'object') return '';

  const direct = output.text;
  if (typeof direct === 'string') return direct.trim();
  if (Array.isArray(direct)) return direct.map((part) => String(part || '')).join('').trim();

  const messageContent = output?.choices?.[0]?.message?.content;
  if (typeof messageContent === 'string') return messageContent.trim();

  if (output.output !== undefined) return textFromOutput(output.output);
  return '';
}

function createInvalidRequest(message) {
  const error = new Error(message);
  error.code = 'INVALID_CHAT_BOSS_REQUEST';
  return error;
}

function createChatBossAssistant(env = process.env, options = {}) {
  const configured = Boolean(options.client || clean(env.OPENAI_API_KEY));
  const client = options.client || (configured ? createOpenAiResponsesClient({
    apiKey: env.OPENAI_API_KEY,
    baseUrl: env.OPENAI_BASE_URL,
    organization: env.OPENAI_ORGANIZATION,
    project: env.OPENAI_PROJECT,
    model: env.OPENAI_CHAT_MODEL || DEFAULT_CHAT_MODEL,
    reasoningEffort: env.OPENAI_CHAT_REASONING_EFFORT || 'low',
    timeoutMs: env.OPENAI_TIMEOUT_MS,
    fetch: options.fetch,
  }) : null);

  function capabilities() {
    return {
      configured,
      provider: 'OpenAI Responses API',
      model: clean(env.OPENAI_CHAT_MODEL) || client?.model || DEFAULT_CHAT_MODEL,
      servedModel: clean(env.OPENAI_CHAT_MODEL) || client?.model || DEFAULT_CHAT_MODEL,
      fineTuned: false,
      historyStorage: 'browser',
      promptVersion: PROMPT_VERSIONS.chatboss,
      contextPolicy: 'server-owned-design-context-no-live-system-access',
    };
  }

  async function submit(messages) {
    if (!client) {
      const error = new Error('Chat Boss is not connected to OpenAI yet.');
      error.code = 'CHAT_BOSS_UNAVAILABLE';
      throw error;
    }
    const safeMessages = sanitizeMessages(messages);
    if (!safeMessages.length || safeMessages.at(-1)?.role !== 'user') {
      throw createInvalidRequest('Send a user message to Chat Boss.');
    }
    return client.submit([
      {
        role: 'system',
        content: [
          SYSTEMS.chatboss,
          'Known Polymath design context (not live health evidence):',
          OWNER_CONTEXT,
        ].join('\n'),
      },
      ...safeMessages,
    ], {
      reasoning_effort: clean(env.OPENAI_CHAT_REASONING_EFFORT) || 'low',
      max_output_tokens: 1200,
      prompt_cache_key: PROMPT_VERSIONS.chatboss,
      metadata: { workload: 'chat-boss', prompt_version: PROMPT_VERSIONS.chatboss },
    });
  }

  async function status(jobId) {
    if (!client) {
      const error = new Error('Chat Boss is not connected to OpenAI yet.');
      error.code = 'CHAT_BOSS_UNAVAILABLE';
      throw error;
    }
    const body = await client.status(jobId);
    const jobStatus = clean(body?.status).toUpperCase() || 'UNKNOWN';
    return {
      id: clean(body?.id) || clean(jobId),
      status: jobStatus,
      active: ACTIVE_STATUSES.has(jobStatus),
      finished: FINISHED_STATUSES.has(jobStatus),
      reply: jobStatus === 'COMPLETED' ? textFromOutput(body?.output) : '',
      error: jobStatus === 'FAILED' || jobStatus === 'TIMED_OUT'
        ? clean(body?.error || body?.output?.error) || 'OpenAI could not complete this reply.'
        : '',
      delayTime: Number(body?.delayTime) || 0,
      executionTime: Number(body?.executionTime) || 0,
    };
  }

  async function cancel(jobId) {
    if (!client) {
      const error = new Error('Chat Boss is not connected to OpenAI yet.');
      error.code = 'CHAT_BOSS_UNAVAILABLE';
      throw error;
    }
    return client.cancel(jobId);
  }

  return Object.freeze({ capabilities, submit, status, cancel });
}

module.exports = {
  createChatBossAssistant,
  sanitizeMessages,
  textFromOutput,
};
