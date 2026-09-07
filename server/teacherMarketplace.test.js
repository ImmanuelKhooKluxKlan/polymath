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
process.env.ADMIN_EMAILS = 'policy-admin@example.test';

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
  const policyAdmin = await register('Policy Admin', 'policy-admin@example.test');

  const teacherPolicies = await api('/api/admin/policies', {
    method: 'PUT',
    token: policyAdmin.token,
    body: {
      teacherDirectoryEnabled: true,
      teacherApplicationsEnabled: true,
      teacherReviewsEnabled: true,
      minimumTeacherHourlyRateMcoins: 10,
      maximumTeacherHourlyRateMcoins: 80,
      teacherMarketplaceFeePercent: 17.5,
      teacherMarketplaceNotice: 'Teachers set their own availability.',
    },
  });
  assert.equal(teacherPolicies.status, 200);
  assert.equal(teacherPolicies.data.policies.teacherMarketplaceFeePercent, 17.5);

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
  assert.deepEqual(directory.data.marketplace, {
    directoryEnabled: true,
    applicationsEnabled: true,
    reviewsEnabled: true,
    minimumHourlyRateMcoins: 10,
    maximumHourlyRateMcoins: 80,
    platformFeePercent: 17.5,
    teacherKeepsPercent: 82.5,
    withdrawalFeePercent: 25,
    notice: 'Teachers set their own availability.',
    checkoutAvailable: false,
  });

  const invalidRate = await api('/api/teachers/me', {
    method: 'PUT',
    token: teacher.token,
    body: {
      headline: 'Patient piano teacher for new players',
      bio: 'I teach practical piano foundations through songs students already enjoy.',
      instruments: ['piano'],
      levels: ['beginner'],
      lessonModes: ['online'],
      hourlyRateMcoins: 9,
    },
  });
  assert.equal(invalidRate.status, 400);
  assert.match(invalidRate.data.error, /from 10 Mcoins to 80/);

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

  const pausedPolicies = await api('/api/admin/policies', {
    method: 'PUT',
    token: policyAdmin.token,
    body: {
      ...teacherPolicies.data.policies,
      teacherApplicationsEnabled: false,
      teacherReviewsEnabled: false,
    },
  });
  assert.equal(pausedPolicies.status, 200);

  const blockedApplication = await api('/api/teachers/me', {
    method: 'PUT',
    token: outsider.token,
    body: {
      headline: 'New guitar teacher',
      bio: 'Friendly guitar lessons for complete beginners.',
      instruments: ['guitar'],
      levels: ['beginner'],
      lessonModes: ['online'],
      hourlyRateMcoins: 30,
    },
  });
  assert.equal(blockedApplication.status, 403);

  const blockedReview = await api(`/api/teachers/${profile.id}/reviews`, {
    method: 'POST',
    token: student.token,
    body: { rating: 4, comment: 'Updated review while reviews are paused.' },
  });
  assert.equal(blockedReview.status, 403);

  const existingTeacherCanStillEdit = await api('/api/teachers/me', {
    method: 'PUT',
    token: teacher.token,
    body: {
      headline: 'Updated patient piano teacher',
      bio: 'I still teach practical piano foundations through familiar songs.',
      instruments: ['piano'],
      levels: ['beginner'],
      lessonModes: ['online'],
      hourlyRateMcoins: 40,
    },
  });
  assert.equal(existingTeacherCanStillEdit.status, 200);

  const hiddenPolicies = await api('/api/admin/policies', {
    method: 'PUT',
    token: policyAdmin.token,
    body: { ...pausedPolicies.data.policies, teacherDirectoryEnabled: false },
  });
  assert.equal(hiddenPolicies.status, 200);
  const hiddenDirectory = await api('/api/teachers');
  assert.equal(hiddenDirectory.status, 200);
  assert.equal(hiddenDirectory.data.marketplace.directoryEnabled, false);
  assert.deepEqual(hiddenDirectory.data.teachers, []);

  const invalidRange = await api('/api/admin/policies', {
    method: 'PUT',
    token: policyAdmin.token,
    body: {
      ...hiddenPolicies.data.policies,
      minimumTeacherHourlyRateMcoins: 100,
      maximumTeacherHourlyRateMcoins: 50,
    },
  });
  assert.equal(invalidRange.status, 400);
});
