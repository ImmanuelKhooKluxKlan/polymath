'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createChatBossAssistant, sanitizeMessages, textFromOutput } = require('./chatBossAssistant');

test('sanitizeMessages keeps recent user and assistant text within limits', () => {
  const messages = sanitizeMessages([
    { role: 'system', content: 'browser supplied system text' },
    { role: 'assistant', content: 'answer' },
    { role: 'user', content: 'question' },
  ]);
  assert.deepEqual(messages, [
    { role: 'user', content: 'browser supplied system text' },
    { role: 'assistant', content: 'answer' },
    { role: 'user', content: 'question' },
  ]);
});

test('textFromOutput reads the native vLLM result shape', () => {
  assert.equal(textFromOutput({ text: ['Hello', ' world'] }), 'Hello world');
  assert.equal(textFromOutput({ output: { text: ['Nested reply'] } }), 'Nested reply');
});

test('assistant submits original chat history without a fine-tuning prompt', async () => {
  let submitted;
  const client = {
    model: 'gpt-5.6-terra',
    async submit(messages, sampling) {
      submitted = { messages, sampling };
      return { id: 'job-12345678', status: 'IN_QUEUE' };
    },
  };
  const assistant = createChatBossAssistant({}, { client });
  const result = await assistant.submit([{ role: 'user', content: 'Who are you?' }]);
  assert.equal(result.id, 'job-12345678');
  assert.deepEqual(submitted.messages, [{ role: 'user', content: 'Who are you?' }]);
  assert.equal(submitted.sampling.max_output_tokens, 1200);
  assert.equal(submitted.sampling.reasoning_effort, 'low');
  assert.equal(assistant.capabilities().fineTuned, false);
  assert.equal(assistant.capabilities().provider, 'OpenAI Responses API');
});

test('assistant returns a completed reply without exposing raw provider data', async () => {
  const assistant = createChatBossAssistant({}, {
    client: {
      model: 'gpt-5.6-terra',
      async status() {
        return {
          id: 'job-12345678',
          status: 'COMPLETED',
          delayTime: 25,
          executionTime: 50,
          output: { text: ['Direct OpenAI reply'] },
          workerId: 'private-worker-id',
        };
      },
    },
  });
  assert.deepEqual(await assistant.status('job-12345678'), {
    id: 'job-12345678',
    status: 'COMPLETED',
    active: false,
    finished: true,
    reply: 'Direct OpenAI reply',
    error: '',
    delayTime: 25,
    executionTime: 50,
  });
});
