'use strict';

const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-create-music-route-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'creator-admin@example.test';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.OPENAI_API_KEY = 'sk-test-openai-key';
process.env.OPENAI_MUSIC_MODEL = 'gpt-test-music';

const nativeFetch = globalThis.fetch;
let openAiCalls = 0;
globalThis.fetch = (url, options) => {
  if (String(url).startsWith('https://api.openai.com/v1/responses')) {
    openAiCalls += 1;
    return Promise.resolve(new Response(JSON.stringify({
      id: 'resp_freshdraft1234',
      status: 'queued',
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
  }
  return nativeFetch(url, options);
};

const { app, readDb, writeDb } = require('./server');

test('legacy music jobs end cleanly and never keep AI drafting locked', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    globalThis.fetch = nativeFetch;
    fs.rmSync(testDataDir, { recursive: true, force: true });
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  async function api(pathname, { method = 'GET', token = '', body } = {}) {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { status: response.status, data: await response.json() };
  }

  const challenge = await api('/api/auth/register/otp', {
    method: 'POST',
    body: { channel: 'email', email: 'creator-admin@example.test' },
  });
  const registration = await api('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Creator Admin',
      email: 'creator-admin@example.test',
      password: 'CreatorPassword123',
      challengeId: challenge.data.challengeId,
      verificationCode: '123456',
    },
  });
  assert.equal(registration.status, 201);
  const userId = registration.data.user.user_id;

  const db = await readDb();
  db.musicGenerationJobs.push({
    id: 'job_legacy_saved_draft',
    providerJobId: 'job_legacy_runpod_id',
    userId,
    status: 'IN_QUEUE',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
  await writeDb(db);

  const recovered = await api('/api/music-creation/jobs/job_legacy_saved_draft', {
    token: registration.data.token,
  });
  assert.equal(recovered.status, 200);
  assert.equal(recovered.data.finished, true);
  assert.equal(recovered.data.status, 'FAILED');
  assert.match(recovered.data.error, /older AI connection/i);
  assert.equal(openAiCalls, 0);

  const nextDb = await readDb();
  nextDb.musicGenerationJobs.push({
    id: 'job_legacy_recent_draft',
    providerJobId: 'legacy_provider_job',
    userId,
    status: 'IN_PROGRESS',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
  await writeDb(nextDb);

  const started = await api('/api/music-creation/jobs', {
    method: 'POST',
    token: registration.data.token,
    body: { kind: 'draft', brief: { idea: 'A hopeful song about starting again.' } },
  });
  assert.equal(started.status, 202);
  assert.equal(started.data.id, 'resp_freshdraft1234');
  assert.equal(started.data.reused, undefined);
  assert.equal(openAiCalls, 1);

  const finalDb = await readDb();
  assert.equal(
    finalDb.musicGenerationJobs.find((job) => job.id === 'job_legacy_recent_draft').status,
    'FAILED',
  );
});

