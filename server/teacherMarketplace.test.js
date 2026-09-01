const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-teacher-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.ADMIN_EMAILS = '';

const { app } = require('./server');

test('teachers publish profiles, connected students review, and chats remain private', async (context) => {
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
      method: 'POST',
      body: { channel: 'email', email },
    });
    assert.equal(challenge.status, 202);
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: {
        name,
        email,
        password: 'TeacherTest123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const teacher = await register('Piano Teacher', 'teacher@example.test');
  const student = await register('Music Student', 'student@example.test');
  const outsider = await register('Private Outsider', 'outsider@example.test');

  const profileResult = await api('/api/teachers/me', {
    method: 'PUT',
    token: teacher.token,
    body: {
      headline: 'Patient piano teacher for new players',
      bio: 'I teach practical piano foundations through songs students already enjoy.',
      instruments: ['piano'],
      levels: ['beginner', 'intermediate'],
      lessonModes: ['online'],
      languages: ['English'],
      availability: 'Saturday mornings',
      hourlyRateMcoins: 35,
      published: true,
    },
  });
  assert.equal(profileResult.status, 201);
  const profile = profileResult.data.teacher;
  assert.equal(profile.name, 'Piano Teacher');

  const directory = await api('/api/teachers?instrument=piano&level=beginner');
  assert.equal(directory.status, 200);
  assert.equal(directory.data.teachers.length, 1);
  assert.equal(directory.data.teachers[0].hourlyRateMcoins, 35);

  const prematureReview = await api(`/api/teachers/${profile.id}/reviews`, {
    method: 'POST',
    token: student.token,
    body: { rating: 5, comment: 'Not connected yet.' },
  });
  assert.equal(prematureReview.status, 403);

  const sent = await api('/api/messages', {
    method: 'POST',
    token: student.token,
    body: { toUserId: teacher.user.user_id, text: 'Could we discuss beginner lessons?' },
  });
  assert.equal(sent.status, 201);

  const review = await api(`/api/teachers/${profile.id}/reviews`, {
    method: 'POST',
    token: student.token,
    body: { rating: 5, comment: 'Patient, clear, and encouraging.' },
  });
  assert.equal(review.status, 201);
  assert.equal(review.data.review.connectedStudent, true);
  assert.equal(review.data.summary.averageRating, 5);

  const rankedDirectory = await api('/api/teachers?instrument=piano');
  assert.equal(rankedDirectory.status, 200);
  assert.equal(rankedDirectory.data.teachers[0].studentCount, 1);
  assert.deepEqual(rankedDirectory.data.teachers[0].ranking, {
    ratingPoints: 10,
    audiencePoints: 1,
    totalPoints: 11,
    maximumPoints: 50,
  });

  const ownReview = await api(`/api/teachers/${profile.id}/reviews`, {
    method: 'POST',
    token: teacher.token,
    body: { rating: 5, comment: 'Self review.' },
  });
  assert.equal(ownReview.status, 403);

  const studentConversation = await api(`/api/messages/${teacher.user.user_id}`, { token: student.token });
  assert.equal(studentConversation.status, 200);
  assert.equal(studentConversation.data.messages.length, 1);
  const outsiderConversation = await api(`/api/messages/${teacher.user.user_id}`, { token: outsider.token });
  assert.equal(outsiderConversation.status, 200);
  assert.equal(outsiderConversation.data.messages.length, 0);
});
