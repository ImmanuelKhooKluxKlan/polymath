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
process.env.RUNPOD_CHAT_BOSS_ENDPOINT_ID = 'queued-teacher-endpoint';
process.env.RUNPOD_API_KEY = 'queued-teacher-test-key';

const nativeFetch = globalThis.fetch;
let statusChecks = 0;
let submittedPayload = null;

globalThis.fetch = async (url, options = {}) => {
  const target = String(url);
  if (!target.startsWith('https://api.runpod.ai/')) return nativeFetch(url, options);
  if (target.endsWith('/run')) {
    submittedPayload = JSON.parse(options.body);
    return new Response(JSON.stringify({ id: 'teacher_job_12345678', status: 'IN_QUEUE' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  if (target.includes('/status/teacher_job_12345678')) {
    statusChecks += 1;
    const body = statusChecks === 1
      ? { id: 'teacher_job_12345678', status: 'IN_PROGRESS' }
      : {
        id: 'teacher_job_12345678',
        status: 'COMPLETED',
        output: { text: ['Start slowly with five relaxed C-major scales.'] },
      };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  return new Response(JSON.stringify({ id: 'teacher_job_12345678', status: 'CANCELLED' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
};

const { app, readDb, writeDb } = require('./server');

test('virtual teacher survives a long GPU cold start through a persistent queued reply', async (context) => {
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
  assert.equal(submitted.status, 202);
  assert.equal(submitted.data.pending, true);
  assert.match(submitted.data.requestId, /^teacher_reply_/);
  assert.equal(submitted.data.session.pendingReply.id, submitted.data.requestId);
  assert.equal(submitted.data.session.remainingSeconds <= 1800, true);
  assert.equal(submittedPayload.input.messages[0].role, 'system');
  assert.match(submittedPayload.input.messages[0].content, /live paid session/i);
  assert.equal(submittedPayload.input.sampling_params.max_tokens, 640);

  const checking = await api(
    `/api/virtual-lessons/${started.data.session.id}/replies/${submitted.data.requestId}`,
    { token: registration.data.token },
  );
  assert.equal(checking.status, 202);
  assert.equal(checking.data.status, 'IN_PROGRESS');

  const completed = await api(
    `/api/virtual-lessons/${started.data.session.id}/replies/${submitted.data.requestId}`,
    { token: registration.data.token },
  );
  assert.equal(completed.status, 200);
  assert.equal(completed.data.pending, false);
  assert.match(completed.data.reply, /five relaxed C-major scales/i);
  assert.equal(Boolean(completed.data.speechMessageId), true);
  assert.equal(completed.data.session.pendingReply, null);
  assert.equal(completed.data.session.messages.length, 2);
  assert.equal(completed.data.session.memory.lastSong, 'Warm-up study');

  const repeatedPoll = await api(
    `/api/virtual-lessons/${started.data.session.id}/replies/${submitted.data.requestId}`,
    { token: registration.data.token },
  );
  assert.equal(repeatedPoll.status, 200);
  assert.equal(repeatedPoll.data.reply, completed.data.reply);
});
