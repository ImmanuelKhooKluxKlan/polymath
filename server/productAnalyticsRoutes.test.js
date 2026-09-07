'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-product-analytics-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.ADMIN_EMAILS = 'growth-admin@example.test';

const { app, readDb, writeDb } = require('./server');

test('product events accept guests while aggregate evidence remains admin-only', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
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

  const accepted = await api('/api/product-events', {
    method: 'POST',
    body: {
      events: [{
        eventId: 'event_route_123456',
        eventName: 'app_opened',
        occurredAt: new Date().toISOString(),
        anonymousId: 'anon_route_123456',
        sessionId: 'session_route_123456',
        path: 'studio',
        properties: { signedIn: false, email: 'must-not-store@example.test' },
      }],
    },
  });
  assert.equal(accepted.status, 202);
  assert.equal(accepted.data.accepted, 1);

  const spoofedPayment = await api('/api/product-events', {
    method: 'POST',
    body: {
      events: [{
        eventId: 'event_fake_payment_123',
        eventName: 'subscription_activated',
        anonymousId: 'anon_fake_payment_123',
        sessionId: 'session_fake_payment_123',
      }],
    },
  });
  assert.equal(spoofedPayment.status, 400);

  const privateSummary = await api('/api/admin/product-analytics');
  assert.equal(privateSummary.status, 401);

  const challenge = await api('/api/auth/register/otp', {
    method: 'POST', body: { channel: 'email', email: 'growth-admin@example.test' },
  });
  const registration = await api('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Growth Admin',
      email: 'growth-admin@example.test',
      password: 'GrowthEvidencePassword123',
      challengeId: challenge.data.challengeId,
      verificationCode: '123456',
    },
  });
  assert.equal(registration.status, 201);
  assert.equal(registration.data.user.admin, true);

  const db = await readDb();
  db.mediaTranscriptionJobs.push({
    id: 'media_tx_feedback_test',
    userId: registration.data.user.user_id,
    filename: 'private-name.mp3',
    title: 'Private title',
    instrument: 'piano',
    playbackMode: 'full',
    status: 'completed',
    stage: 'Ready to play',
    progress: 100,
    startedAt: new Date(Date.now() - 30000).toISOString(),
    completedAt: new Date().toISOString(),
  });
  await writeDb(db);
  const feedback = await api('/api/media-transcriptions/media_tx_feedback_test/feedback', {
    method: 'POST',
    token: registration.data.token,
    body: { feedback: 'accurate' },
  });
  assert.equal(feedback.status, 200);
  assert.equal(feedback.data.job.feedback, 'accurate');
  const changedFeedback = await api('/api/media-transcriptions/media_tx_feedback_test/feedback', {
    method: 'POST',
    token: registration.data.token,
    body: { feedback: 'needs-work' },
  });
  assert.equal(changedFeedback.status, 409);

  const summary = await api('/api/admin/product-analytics?days=7', { token: registration.data.token });
  assert.equal(summary.status, 200);
  assert.equal(summary.data.windowDays, 7);
  assert.equal(summary.data.stages[0].actors, 1);
  assert.equal(summary.data.transcription.playablePercent, 100);
  assert.match(summary.data.privacy, /No source audio/);
});
