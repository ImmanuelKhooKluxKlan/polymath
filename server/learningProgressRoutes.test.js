'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-learning-progress-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';

const { app, readDb, writeDb } = require('./server');

test('learning evidence is private, idempotent, bounded, and available only with Learn access', async (context) => {
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

  async function register(name, email) {
    const challenge = await api('/api/auth/register/otp', {
      method: 'POST', body: { channel: 'email', email },
    });
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: {
        name,
        email,
        password: 'LearningProgressPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const learner = await register('Learning Student', 'learning-student@example.test');
  const blocked = await api('/api/learning/progress', { token: learner.token });
  assert.equal(blocked.status, 403);
  assert.equal(blocked.data.code, 'LEARN_ACCESS_REQUIRED');

  const other = await register('Other Student', 'other-learning-student@example.test');
  const db = await readDb();
  [learner.user.user_id, other.user.user_id].forEach((userId) => {
    const user = db.users.find((candidate) => candidate.id === userId);
    user.subscriptionTier = 'musician';
    user.proStatus = 'ACTIVE';
    user.pro = true;
  });
  await writeDb(db);

  const attempt = {
    attemptId: 'attempt_1234567890',
    songId: 'library:mean',
    songTitle: 'Mean',
    report: {
      createdAt: '2026-09-06T04:00:00.000Z',
      levelId: 'first-keys',
      range: { start: 10, end: 22 },
      score: 999,
      speedPercent: 65,
      handMode: 'right',
      practiceFocusId: 'rhythm',
      practicePlanSource: 'measured',
      practiceTargetScore: 82,
      focus: 'Notes',
      strongest: 'Rhythm',
      expectedCount: 12,
      matchedCount: 9,
      missedCount: 3,
      metrics: {
        notes: { score: 75, available: true, label: '<script>wrong</script>' },
        rhythm: { score: 84, available: true },
        dynamics: { score: null, available: false },
      },
    },
  };
  const saved = await api('/api/learning/attempts', { method: 'POST', token: learner.token, body: attempt });
  assert.equal(saved.status, 201);
  assert.equal(saved.data.attempts.length, 1);
  assert.equal(saved.data.attempts[0].score, 100);
  assert.equal(saved.data.attempts[0].metrics.notes.label, 'Notes');
  assert.equal(saved.data.attempts[0].metrics.dynamics.available, false);
  assert.equal(saved.data.attempts[0].metrics.dynamics.score, null);
  assert.equal(saved.data.attempts[0].speedPercent, 65);
  assert.equal(saved.data.attempts[0].handMode, 'right');
  assert.equal(saved.data.attempts[0].practiceFocusId, 'rhythm');
  assert.equal(saved.data.attempts[0].practiceTargetScore, 82);
  assert.equal(saved.data.attempts[0].levelId, 'first-keys');
  assert.match(saved.data.attempts[0].sectionKey, /^first-keys:/);

  const duplicate = await api('/api/learning/attempts', { method: 'POST', token: learner.token, body: attempt });
  assert.equal(duplicate.status, 200);
  assert.equal(duplicate.data.duplicate, true);
  assert.equal(duplicate.data.attempts.length, 1);

  const synchronized = await api('/api/learning/progress/sync', {
    method: 'POST',
    token: learner.token,
    body: {
      attempts: [
        attempt,
        {
          ...attempt,
          attemptId: 'attempt_offline_123456',
          report: {
            ...attempt.report,
            createdAt: '2026-09-06T04:05:00.000Z',
            levelId: 'piano-player',
            score: 81,
          },
        },
        {
          ...attempt,
          attemptId: 'attempt_legacy_1234567',
          report: {
            ...attempt.report,
            createdAt: '2026-09-06T04:10:00.000Z',
            levelId: 'original',
            score: 88,
          },
        },
      ],
    },
  });
  assert.equal(synchronized.status, 200);
  assert.equal(synchronized.data.savedCount, 2);
  assert.equal(synchronized.data.attempts.length, 3);
  assert.equal(synchronized.data.attempts[1].levelId, 'piano-player');
  assert.equal(synchronized.data.attempts[2].levelId, 'piano-king');

  const ownProgress = await api('/api/learning/progress', { token: learner.token });
  assert.equal(ownProgress.status, 200);
  assert.equal(ownProgress.data.attempts[0].id, attempt.attemptId);
  assert.equal(ownProgress.data.attempts.length, 3);
  const isolatedProgress = await api('/api/learning/progress', { token: other.token });
  assert.equal(isolatedProgress.status, 200);
  assert.deepEqual(isolatedProgress.data.attempts, []);

  const invalid = await api('/api/learning/attempts', {
    method: 'POST', token: learner.token, body: { ...attempt, attemptId: 'bad id' },
  });
  assert.equal(invalid.status, 400);
  assert.equal(invalid.data.code, 'INVALID_LEARNING_ATTEMPT_ID');
});
