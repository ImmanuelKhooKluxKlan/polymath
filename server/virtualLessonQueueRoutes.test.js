'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-lesson-queue-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.POLYMATH_ASSISTANT_PROVIDER = 'deepseek';
process.env.DEEPSEEK_API_KEY = 'direct-teacher-test-key';
process.env.DEEPSEEK_MODEL = 'deepseek-v4-flash';

const nativeFetch = globalThis.fetch;
let submittedPayload = null;

globalThis.fetch = async (url, options = {}) => {
  const target = String(url);
  if (!target.startsWith('https://api.deepseek.com/')) return nativeFetch(url, options);
  if (target.endsWith('/chat/completions')) {
    submittedPayload = JSON.parse(options.body);
    return new Response(JSON.stringify({
      choices: [{ message: { content: 'Start slowly with five relaxed C-major scales.' } }],
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  return new Response(JSON.stringify({ error: 'unexpected DeepSeek test route' }), { status: 404 });
};

const { app, readDb, writeDb } = require('./server');

test('virtual teacher returns a direct DeepSeek reply without a RunPod queue', async (context) => {
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
    const response = await nativeFetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { status: response.status, data: await response.json() };
  }

  const email = 'queued-student@example.test';
  const challenge = await api('/api/auth/register/otp', {
    method: 'POST', body: { channel: 'email', email },
  });
  const registration = await api('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Queued Student',
      email,
      password: 'QueuedTeacherPassword123',
      challengeId: challenge.data.challengeId,
      verificationCode: '123456',
    },
  });
  const db = await readDb();
  db.users.find((user) => user.id === registration.data.user.user_id).mcoins = 10;
  await writeDb(db);

  const started = await api('/api/virtual-lessons', {
    method: 'POST',
    token: registration.data.token,
    body: {
      durationMinutes: 30,
      clientRequestId: 'queue_checkout_123456',
      teacher: { id: 'aria', name: 'Aria', title: 'Piano teacher', style: 'Patient' },
    },
  });
  assert.equal(started.status, 201);

  const submitted = await api(`/api/virtual-lessons/${started.data.session.id}/messages`, {
    method: 'POST',
    token: registration.data.token,
    body: {
      message: 'Give me a short piano warm-up.',
      lessonContext: { title: 'Warm-up study' },
      observations: {},
    },
  });
  assert.equal(submitted.status, 200);
  assert.equal(submitted.data.pending, undefined);
  assert.equal(submitted.data.session.pendingReply, null);
  assert.equal(submitted.data.session.remainingSeconds <= 1800, true);
  assert.equal(submittedPayload.model, 'deepseek-v4-flash');
  assert.equal(submittedPayload.messages[0].role, 'system');
  assert.match(submittedPayload.messages[0].content, /live paid session/i);
  assert.equal(submittedPayload.max_tokens, 220);
  assert.deepEqual(submittedPayload.thinking, { type: 'disabled' });
  assert.match(submitted.data.reply, /five relaxed C-major scales/i);
  assert.equal(Boolean(submitted.data.speechMessageId), true);
  assert.equal(submitted.data.session.messages.length, 2);
  assert.equal(submitted.data.session.memory.lastSong, 'Warm-up study');
});
