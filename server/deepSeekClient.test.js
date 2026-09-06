'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { createDeepSeekClient, safeParameters } = require('./deepSeekClient');

test('DeepSeek client sends non-thinking chat completions without RunPod-only fields', async () => {
  const calls = [];
  const client = createDeepSeekClient({
    apiKey: 'test-key',
    model: 'deepseek-v4-flash',
    fetch: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        async json() { return { choices: [{ message: { content: 'Try C4 again.' } }] }; },
      };
    },
  });

  const response = await client.chat([{ role: 'user', content: 'Help me.' }], {
    temperature: 0.4,
    max_tokens: 220,
    top_k: 20,
    repetition_penalty: 1.06,
  });
  const body = JSON.parse(calls[0].options.body);
  assert.equal(calls[0].url, 'https://api.deepseek.com/chat/completions');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer test-key');
  assert.equal(body.model, 'deepseek-v4-flash');
  assert.deepEqual(body.thinking, { type: 'disabled' });
  assert.equal(body.temperature, 0.4);
  assert.equal(body.max_tokens, 220);
  assert.equal(body.top_k, undefined);
  assert.equal(body.repetition_penalty, undefined);
  assert.equal(response.choices[0].message.content, 'Try C4 again.');
});

test('DeepSeek parameter filtering keeps only supported chat fields', () => {
  assert.deepEqual(safeParameters({ top_p: 0.8, top_k: 20, max_tokens: 100 }), {
    top_p: 0.8,
    max_tokens: 100,
  });
});
