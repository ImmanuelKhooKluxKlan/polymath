'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-support-assistant-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'admin@example.test';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.POLYMATH_ASSISTANT_PROVIDER = 'deepseek';
process.env.DEEPSEEK_API_KEY = 'test-deepseek-key';
process.env.DEEPSEEK_MODEL = 'deepseek-v4-flash';
process.env.SUPPORT_REQUEST_INTERVAL_MS = '0';

const nativeFetch = globalThis.fetch;
globalThis.fetch = (url, options) => {
  if (String(url).startsWith('https://api.deepseek.com/')) {
    if (String(options?.body || '').includes('force-outage')) {
      return Promise.resolve(new Response(JSON.stringify({ error: 'simulated outage' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }));
    }
    return Promise.resolve(new Response(JSON.stringify({
      choices: [{ message: { content: 'Here is a concise support answer.' } }],
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
  }
  return nativeFetch(url, options);
};

const { app, readDb } = require('./server');

test('Help requires sign-in, allows seven daily questions, then returns admin helpline details', async (context) => {
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

  async function register(name, email) {
    const challenge = await api('/api/auth/register/otp', { method: 'POST', body: { channel: 'email', email } });
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: {
        name,
        email,
        password: 'SupportPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const signedOut = await api('/api/assistant/capabilities');
  assert.equal(signedOut.status, 401);

  const admin = await register('Support Admin', 'admin@example.test');
  const policy = await api('/api/admin/policies', {
    method: 'PUT',
    token: admin.token,
    body: { supportEmail: 'help@polymath.test', supportPhone: '+65 6123 4567' },
  });
  assert.equal(policy.status, 200);
  assert.equal(policy.data.policies.supportPhone, '+65 6123 4567');

  const student = await register('Help Student', 'student@example.test');
  const initial = await api('/api/assistant/capabilities', { token: student.token });
  assert.equal(initial.status, 200);
  assert.equal(initial.data.support.dailyLimit, 7);
  assert.equal(initial.data.support.remainingQuestions, 7);
  assert.deepEqual(initial.data.support.contact, {
    email: 'help@polymath.test',
    phone: '+65 6123 4567',
  });

  const failed = await api('/api/assistant/support', {
    method: 'POST',
    token: student.token,
    body: { messages: [{ role: 'user', content: 'force-outage' }] },
  });
  assert.equal(failed.status, 502);
  assert.equal(failed.data.support.remainingQuestions, 7);

  for (let index = 0; index < 7; index += 1) {
    const answer = await api('/api/assistant/support', {
      method: 'POST',
      token: student.token,
      body: { messages: [{ role: 'user', content: `Help question ${index + 1}` }] },
    });
    assert.equal(answer.status, 200);
    assert.equal(answer.data.support.remainingQuestions, 6 - index);
  }

  const exhausted = await api('/api/assistant/support', {
    method: 'POST',
    token: student.token,
    body: { messages: [{ role: 'user', content: 'Question eight' }] },
  });
  assert.equal(exhausted.status, 429);
  assert.equal(exhausted.data.code, 'SUPPORT_DAILY_LIMIT_REACHED');
  assert.equal(exhausted.data.support.remainingQuestions, 0);
  assert.equal(exhausted.data.support.contact.phone, '+65 6123 4567');

  for (let index = 0; index < 8; index += 1) {
    const adminAnswer = await api('/api/assistant/support', {
      method: 'POST',
      token: admin.token,
      body: { messages: [{ role: 'user', content: `Admin help ${index + 1}` }] },
    });
    assert.equal(adminAnswer.status, 200);
    assert.equal(adminAnswer.data.support.unlimited, true);
  }

  const db = await readDb();
  const studentRecord = db.users.find((user) => user.id === student.user.user_id);
  const adminRecord = db.users.find((user) => user.id === admin.user.user_id);
  assert.equal(studentRecord.supportQuestionUsage.usedQuestions, 7);
  assert.equal(adminRecord.supportQuestionUsage, undefined);
});
