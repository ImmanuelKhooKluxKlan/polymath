'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  createOpenAiResponsesClient,
  extractOutputText,
  normalizeResponse,
  modelAcceptsReasoning,
  prepareInput,
} = require('./openAiResponses');

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

test('Responses client keeps system instructions server-side and creates a private synchronous response', async () => {
  let request;
  const client = createOpenAiResponsesClient({
    apiKey: 'sk-test-only',
    model: 'gpt-test-chat',
    async fetch(url, options) {
      request = { url, options };
      return jsonResponse({
        id: 'resp_sync_12345678',
        status: 'completed',
        output: [{
          type: 'message',
          content: [{ type: 'output_text', text: 'A clear answer.' }],
        }],
        usage: { input_tokens: 12, output_tokens: 4, total_tokens: 16 },
      });
    },
  });

  const result = await client.chat([
    { role: 'system', content: 'Be concise.' },
    { role: 'user', content: 'Help me.' },
  ], { max_tokens: 333, temperature: 0.9 });

  assert.equal(request.url, 'https://api.openai.com/v1/responses');
  assert.equal(request.options.headers.Authorization, 'Bearer sk-test-only');
  const payload = JSON.parse(request.options.body);
  assert.equal(payload.model, 'gpt-test-chat');
  assert.equal(payload.instructions, 'Be concise.');
  assert.deepEqual(payload.input, [{ role: 'user', content: 'Help me.' }]);
  assert.equal(payload.background, false);
  assert.equal(payload.store, false);
  assert.equal(payload.max_output_tokens, 333);
  assert.equal(payload.reasoning.effort, 'low');
  assert.equal(Object.hasOwn(payload, 'temperature'), false);
  assert.equal(result.output.text[0], 'A clear answer.');
  assert.equal(result.usage.totalTokens, 16);
});

test('Responses client submits, retrieves, and cancels durable background responses', async () => {
  const requests = [];
  const responseId = 'resp_background_12345678';
  const client = createOpenAiResponsesClient({
    apiKey: 'sk-test-only',
    model: 'gpt-test-background',
    async fetch(url, options) {
      requests.push({ url, options });
      if (url.endsWith('/responses') && options.method === 'POST') {
        return jsonResponse({ id: responseId, status: 'queued' });
      }
      if (url.endsWith(`/responses/${responseId}`)) {
        return jsonResponse({
          id: responseId,
          status: 'completed',
          output: [{ type: 'message', content: [{ type: 'output_text', text: 'Finished.' }] }],
        });
      }
      return jsonResponse({ id: responseId, status: 'cancelled' });
    },
  });

  const submitted = await client.submit([{ role: 'user', content: 'Work in the background.' }]);
  assert.deepEqual(submitted, { id: responseId, status: 'IN_QUEUE' });
  const submitPayload = JSON.parse(requests[0].options.body);
  assert.equal(submitPayload.background, true);
  assert.equal(submitPayload.store, true);

  const completed = await client.status(responseId);
  assert.equal(completed.status, 'COMPLETED');
  assert.equal(completed.output.text[0], 'Finished.');

  const cancelled = await client.cancel(responseId);
  assert.equal(cancelled.status, 'CANCELLED');
  assert.match(requests[2].url, /\/cancel$/);
  await assert.rejects(() => client.status('../not-a-response'), /Invalid OpenAI response ID/);
});

test('Responses helpers parse multimodal input and normalize incomplete responses as failures', () => {
  const prepared = prepareInput([{
    role: 'user',
    content: [
      { type: 'input_text', text: 'What is visible?' },
      { type: 'input_image', image_url: 'data:image/png;base64,AQID', detail: 'low' },
    ],
  }]);
  assert.equal(prepared.input[0].content[1].type, 'input_image');
  assert.equal(extractOutputText({ output_text: 'Shortcut text' }), 'Shortcut text');
  assert.equal(modelAcceptsReasoning('gpt-5.6-terra'), true);
  assert.equal(modelAcceptsReasoning('gpt-4.1-mini-2025-04-14'), false);
  assert.equal(modelAcceptsReasoning('ft:gpt-4.1-mini-2025-04-14:org:polymath'), false);
  assert.deepEqual(normalizeResponse({
    id: 'resp_incomplete_12345678',
    status: 'incomplete',
    incomplete_details: { reason: 'max_output_tokens' },
  }), {
    id: 'resp_incomplete_12345678',
    status: 'FAILED',
    output: { text: [] },
    error: 'max_output_tokens',
    usage: null,
  });
});
