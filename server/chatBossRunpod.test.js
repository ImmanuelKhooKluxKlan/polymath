'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createChatBossRunpodClient } = require('./chatBossRunpod');

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    async json() { return body; },
  };
}

test('native RunPod chat submits an asynchronous vLLM job', async () => {
  let request;
  const client = createChatBossRunpodClient({
    endpointId: 'endpoint-123',
    apiKey: 'secret-test-key',
    timeoutMs: 600000,
    fetch: async (url, options) => {
      request = { url, options };
      return jsonResponse({ id: 'job-12345678', status: 'IN_QUEUE' });
    },
  });

  const result = await client.submit([{ role: 'user', content: 'Hello' }]);
  assert.deepEqual(result, { id: 'job-12345678', status: 'IN_QUEUE' });
  assert.equal(request.url, 'https://api.runpod.ai/v2/endpoint-123/run');
  assert.equal(request.options.headers.Authorization, 'Bearer secret-test-key');
  const payload = JSON.parse(request.options.body);
  assert.deepEqual(payload.input.messages, [{ role: 'user', content: 'Hello' }]);
  assert.equal(payload.input.sampling_params.max_tokens, 1024);
});

test('native RunPod status and cancellation encode validated job IDs', async () => {
  const urls = [];
  const client = createChatBossRunpodClient({
    endpointId: 'endpoint-123',
    apiKey: 'secret-test-key',
    fetch: async (url) => {
      urls.push(url);
      return jsonResponse({ id: 'job_12345678', status: 'CANCELLED' });
    },
  });
  await client.status('job_12345678');
  await client.cancel('job_12345678');
  assert.deepEqual(urls, [
    'https://api.runpod.ai/v2/endpoint-123/status/job_12345678',
    'https://api.runpod.ai/v2/endpoint-123/cancel/job_12345678',
  ]);
  await assert.rejects(() => client.status('../bad'), /Invalid RunPod job ID/);
});
