'use strict';

const path = require('path');

const DEFAULT_FINE_TUNE_MODEL = 'gpt-4.1-mini-2025-04-14';
const MINIMUM_EXAMPLES = 10;
const MAXIMUM_MESSAGE_CHARS = 30000;
const SECRET_PATTERN = /(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|\bAKIA[A-Z0-9]{16}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----)/;

function clean(value, maximum = 1000) {
  return String(value || '').trim().slice(0, maximum);
}

function fineTuningDataError(message) {
  const error = new Error(message);
  error.code = 'INVALID_FINE_TUNING_DATA';
  return error;
}

function parseJsonl(text, source = 'training data') {
  const rows = [];
  String(text || '').split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      throw fineTuningDataError(`${source} line ${index + 1} is not valid JSON: ${error.message}`);
    }
  });
  return rows;
}

function normalizeExample(example, index) {
  if (!example || typeof example !== 'object' || Array.isArray(example)) {
    throw fineTuningDataError(`Example ${index + 1} must be a JSON object.`);
  }
  if (!Array.isArray(example.messages) || example.messages.length < 2) {
    throw fineTuningDataError(`Example ${index + 1} must contain at least two messages.`);
  }
  const messages = example.messages.map((message, messageIndex) => {
    const role = clean(message?.role, 20).toLowerCase();
    if (!['system', 'user', 'assistant'].includes(role)) {
      throw fineTuningDataError(`Example ${index + 1}, message ${messageIndex + 1} has an invalid role.`);
    }
    const content = clean(message?.content, MAXIMUM_MESSAGE_CHARS);
    if (!content) {
      throw fineTuningDataError(`Example ${index + 1}, message ${messageIndex + 1} is empty.`);
    }
    if (SECRET_PATTERN.test(content)) {
      throw fineTuningDataError(`Example ${index + 1} appears to contain a credential or private key.`);
    }
    return { role, content };
  });
  if (!messages.some((message) => message.role === 'user')) {
    throw fineTuningDataError(`Example ${index + 1} has no user message.`);
  }
  if (messages.at(-1)?.role !== 'assistant') {
    throw fineTuningDataError(`Example ${index + 1} must end with the desired assistant answer.`);
  }
  return { messages };
}

function validateTrainingExamples(examples, options = {}) {
  if (!Array.isArray(examples)) throw fineTuningDataError('Training examples must be an array.');
  const minimum = Math.max(1, Number(options.minimumExamples) || MINIMUM_EXAMPLES);
  if (examples.length < minimum) {
    throw fineTuningDataError(`At least ${minimum} examples are required; found ${examples.length}.`);
  }
  const normalized = examples.map(normalizeExample);
  const userPrompts = new Set();
  let totalCharacters = 0;
  normalized.forEach((example, index) => {
    totalCharacters += example.messages.reduce((sum, message) => sum + message.content.length, 0);
    const latestUser = [...example.messages].reverse().find((message) => message.role === 'user');
    const fingerprint = latestUser.content.toLowerCase().replace(/\s+/g, ' ').trim();
    if (userPrompts.has(fingerprint)) {
      throw fineTuningDataError(`Example ${index + 1} duplicates another user prompt.`);
    }
    userPrompts.add(fingerprint);
  });
  return {
    examples: normalized,
    summary: {
      exampleCount: normalized.length,
      totalMessages: normalized.reduce((sum, example) => sum + example.messages.length, 0),
      totalCharacters,
      uniqueUserPrompts: userPrompts.size,
    },
  };
}

function formatJsonl(examples, options = {}) {
  const validated = validateTrainingExamples(examples, options);
  return {
    ...validated,
    jsonl: `${validated.examples.map((example) => JSON.stringify(example)).join('\n')}\n`,
  };
}

function providerError(body) {
  return clean(body?.error?.message || body?.error || body?.message, 600);
}

function createOpenAiFineTuningClient(options = {}) {
  const apiKey = clean(options.apiKey || process.env.OPENAI_API_KEY, 1000);
  const baseUrl = clean(options.baseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1', 500)
    .replace(/\/+$/, '');
  const organization = clean(options.organization || process.env.OPENAI_ORGANIZATION, 200);
  const project = clean(options.project || process.env.OPENAI_PROJECT, 200);
  const requestFetch = options.fetch || globalThis.fetch;
  const timeoutMs = Math.max(5000, Math.min(120000, Number(options.timeoutMs) || 60000));

  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');
  if (typeof requestFetch !== 'function') throw new Error('A fetch implementation is required.');

  async function requestJson(pathname, requestOptions = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const hasMultipartBody = typeof FormData !== 'undefined' && requestOptions.body instanceof FormData;
      const response = await requestFetch(`${baseUrl}${pathname}`, {
        ...requestOptions,
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          ...(!hasMultipartBody && requestOptions.body ? { 'Content-Type': 'application/json' } : {}),
          ...(organization ? { 'OpenAI-Organization': organization } : {}),
          ...(project ? { 'OpenAI-Project': project } : {}),
          ...(requestOptions.headers || {}),
        },
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(`OpenAI fine-tuning request failed (${response.status}): ${providerError(body) || response.statusText || 'Unknown error'}`);
        error.code = response.status === 403 ? 'OPENAI_FINE_TUNING_ACCESS_DENIED' : 'OPENAI_FINE_TUNING_REQUEST_FAILED';
        error.status = response.status;
        throw error;
      }
      return body;
    } catch (error) {
      if (error?.name === 'AbortError') {
        const timeoutError = new Error('OpenAI fine-tuning request timed out.');
        timeoutError.code = 'OPENAI_FINE_TUNING_TIMEOUT';
        throw timeoutError;
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  async function uploadTrainingFile({ filename, jsonl }) {
    const safeFilename = path.basename(clean(filename, 160) || 'polymath-training.jsonl');
    if (!safeFilename.toLowerCase().endsWith('.jsonl')) {
      throw fineTuningDataError('Fine-tuning files must use the .jsonl extension.');
    }
    const parsed = parseJsonl(jsonl, safeFilename);
    validateTrainingExamples(parsed);
    const form = new FormData();
    form.append('purpose', 'fine-tune');
    form.append('file', new Blob([jsonl], { type: 'application/jsonl' }), safeFilename);
    const body = await requestJson('/files', { method: 'POST', body: form });
    if (!clean(body?.id)) throw new Error('OpenAI uploaded the dataset without returning a file ID.');
    return body;
  }

  async function createJob({
    trainingFileId,
    validationFileId,
    model = DEFAULT_FINE_TUNE_MODEL,
    suffix = 'polymath-v001',
    epochs = 'auto',
    metadata = {},
  }) {
    const trainingFile = clean(trainingFileId, 200);
    const validationFile = clean(validationFileId, 200);
    if (!/^file-[A-Za-z0-9_-]+$/.test(trainingFile)) throw new Error('A valid OpenAI training file ID is required.');
    if (validationFile && !/^file-[A-Za-z0-9_-]+$/.test(validationFile)) throw new Error('Invalid OpenAI validation file ID.');
    const numberOfEpochs = epochs === 'auto'
      ? 'auto'
      : Math.max(1, Math.min(20, Math.floor(Number(epochs) || 1)));
    return requestJson('/fine_tuning/jobs', {
      method: 'POST',
      body: JSON.stringify({
        training_file: trainingFile,
        ...(validationFile ? { validation_file: validationFile } : {}),
        model: clean(model, 200) || DEFAULT_FINE_TUNE_MODEL,
        suffix: clean(suffix, 40).replace(/[^A-Za-z0-9_-]+/g, '-'),
        method: {
          type: 'supervised',
          supervised: { hyperparameters: { n_epochs: numberOfEpochs } },
        },
        metadata: Object.fromEntries(Object.entries(metadata).slice(0, 16).map(([key, value]) => [
          clean(key, 64), clean(value, 512),
        ]).filter(([key]) => key)),
      }),
    });
  }

  function validatedId(value, prefix, label) {
    const id = clean(value, 240);
    if (!new RegExp(`^${prefix}[A-Za-z0-9_-]+$`).test(id)) throw new Error(`Invalid OpenAI ${label} ID.`);
    return id;
  }

  async function retrieveJob(jobId) {
    const id = validatedId(jobId, 'ftjob-', 'fine-tuning job');
    return requestJson(`/fine_tuning/jobs/${encodeURIComponent(id)}`, { method: 'GET' });
  }

  async function cancelJob(jobId) {
    const id = validatedId(jobId, 'ftjob-', 'fine-tuning job');
    return requestJson(`/fine_tuning/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
  }

  async function listEvents(jobId, after = '') {
    const id = validatedId(jobId, 'ftjob-', 'fine-tuning job');
    const query = new URLSearchParams({ limit: '100' });
    if (clean(after, 240)) query.set('after', clean(after, 240));
    return requestJson(`/fine_tuning/jobs/${encodeURIComponent(id)}/events?${query}`, { method: 'GET' });
  }

  async function deleteFile(fileId) {
    const id = validatedId(fileId, 'file-', 'file');
    return requestJson(`/files/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  return Object.freeze({
    uploadTrainingFile,
    createJob,
    retrieveJob,
    cancelJob,
    listEvents,
    deleteFile,
  });
}

module.exports = {
  DEFAULT_FINE_TUNE_MODEL,
  MINIMUM_EXAMPLES,
  createOpenAiFineTuningClient,
  formatJsonl,
  parseJsonl,
  validateTrainingExamples,
};
