const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-human-lessons-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.REGISTRATION_OTP_SECRET = 'human-lesson-test-secret-value-that-persists';
process.env.MUSCRIPTOR_ENABLED = 'false';

const { app, readDb, writeDb } = require('./server');

test('teachers host billed calls, invite privately, create breakouts, signal peers, and receive Mcoins', async (context) => {
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
        password: 'HumanLessonPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const teacher = await register('Host Teacher', 'host@example.test');
  const student = await register('Lesson Student', 'student@example.test');
  const outsider = await register('Private Outsider', 'outsider@example.test');
  const db = await readDb();
  db.users.find((user) => user.id === teacher.user.user_id).mcoins = 10;
  db.users.find((user) => user.id === student.user.user_id).mcoins = 100;
  await writeDb(db);

  const profile = await api('/api/teachers/me', {
    method: 'PUT', token: teacher.token, body: {
      headline: 'Patient online piano teacher',
      bio: 'Clear one-to-one lessons for students learning their favourite songs.',
      instruments: ['piano'], levels: ['beginner'], lessonModes: ['online'],
      languages: ['English'], hourlyRateMcoins: 20, published: true,
    },
  });
  assert.equal(profile.status, 201);

  const invalidDuration = await api('/api/human-lessons', {
    method: 'POST', token: teacher.token, body: { title: 'Invalid', durationMinutes: 15 },
  });
  assert.equal(invalidDuration.status, 400);
  assert.equal(invalidDuration.data.code, 'INVALID_LESSON_DURATION');

  const created = await api('/api/human-lessons', {
    method: 'POST', token: teacher.token, body: { title: 'Mean — piano lesson', durationMinutes: 60 },
  });
  assert.equal(created.status, 201);
  assert.equal(created.data.meeting.quotedCallCostMcoins, 1.2);
  assert.match(created.data.meeting.meetingId, /^PM-/);
  assert.match(created.data.meeting.accessCode, /^[A-Z2-9]{4}-[A-Z2-9]{4}$/);
  assert.equal(created.data.user.mcoins, 10, 'scheduling must not charge the teacher');
  const mainMeetingId = created.data.meeting.meetingId;
  const mainAccessCode = created.data.meeting.accessCode;

  const privateBeforeInvite = await api('/api/human-lessons', { token: outsider.token });
  assert.equal(privateBeforeInvite.data.meetings.length, 0);
  const wrongPassword = await api('/api/human-lessons/join', {
    method: 'POST', token: student.token, body: { meetingId: mainMeetingId, accessCode: 'AAAA-AAAA' },
  });
  assert.equal(wrongPassword.status, 403);

  const invited = await api(`/api/human-lessons/${mainMeetingId}/invitations`, {
    method: 'POST', token: teacher.token, body: { toUserId: student.user.user_id },
  });
  assert.equal(invited.status, 201);
  assert.equal(invited.data.message.lessonInvite.accessCode, mainAccessCode);
  const outsiderConversation = await api(`/api/messages/${teacher.user.user_id}`, { token: outsider.token });
  assert.equal(outsiderConversation.data.messages.length, 0, 'meeting credentials must stay private');
  const studentConversation = await api(`/api/messages/${teacher.user.user_id}`, { token: student.token });
  assert.equal(studentConversation.data.messages[0].kind, 'lesson-invite');

  const waiting = await api('/api/human-lessons/join', {
    method: 'POST', token: student.token, body: { meetingId: mainMeetingId, accessCode: mainAccessCode },
  });
  assert.equal(waiting.status, 200);
  assert.equal(waiting.data.waitingForHost, true);

  const started = await api('/api/human-lessons/join', {
    method: 'POST', token: teacher.token, body: { meetingId: mainMeetingId },
  });
  assert.equal(started.status, 200);
  assert.equal(started.data.chargedMcoins, 1.2);
  assert.equal(started.data.user.mcoins, 8.8);
  assert.equal(started.data.meeting.status, 'active');
  const rejoined = await api('/api/human-lessons/join', {
    method: 'POST', token: teacher.token, body: { meetingId: mainMeetingId },
  });
  assert.equal(rejoined.data.chargedMcoins, 0, 'host rejoin must never charge twice');
  assert.equal(rejoined.data.user.mcoins, 8.8);

  const roomState = await api(`/api/human-lessons/${mainMeetingId}/room-state`, { token: teacher.token });
  assert.equal(roomState.status, 200);
  assert.equal(roomState.data.meeting.participants.length, 2);
  assert.ok(Array.isArray(roomState.data.iceServers));
  const signalled = await api(`/api/human-lessons/${mainMeetingId}/signals`, {
    method: 'POST', token: teacher.token, body: {
      toUserId: student.user.user_id,
      kind: 'offer',
      payload: { type: 'offer', sdp: 'test-session-description' },
    },
  });
  assert.equal(signalled.status, 201);
  const studentState = await api(`/api/human-lessons/${mainMeetingId}/room-state?after=1970-01-01T00:00:00.000Z`, { token: student.token });
  assert.equal(studentState.data.signals.length, 1);
  assert.equal(studentState.data.signals[0].kind, 'offer');

  const breakout = await api(`/api/human-lessons/${mainMeetingId}/breakouts`, {
    method: 'POST', token: teacher.token, body: { name: 'Piano pair A' },
  });
  assert.equal(breakout.status, 201);
  assert.equal(breakout.data.chargedMcoins, 0.5);
  assert.equal(breakout.data.user.mcoins, 8.3);
  assert.equal(breakout.data.breakout.kind, 'breakout');
  assert.ok(breakout.data.breakout.accessCode);
  const breakoutInvite = await api(`/api/human-lessons/${breakout.data.breakout.meetingId}/invitations`, {
    method: 'POST', token: teacher.token, body: { toUserId: student.user.user_id },
  });
  assert.equal(breakoutInvite.status, 201);
  const joinedBreakout = await api('/api/human-lessons/join', {
    method: 'POST', token: student.token, body: {
      meetingId: breakout.data.breakout.meetingId,
      accessCode: breakout.data.breakout.accessCode,
    },
  });
  assert.equal(joinedBreakout.status, 200);

  const tooSmall = await api('/api/messages/transfers', {
    method: 'POST', token: student.token, body: {
      toUserId: teacher.user.user_id, amountMcoins: 29.99, clientRequestId: 'small-transfer',
    },
  });
  assert.equal(tooSmall.status, 400);
  const transferred = await api('/api/messages/transfers', {
    method: 'POST', token: student.token, body: {
      toUserId: teacher.user.user_id, amountMcoins: 30, clientRequestId: 'lesson-payment-1',
    },
  });
  assert.equal(transferred.status, 201);
  assert.equal(transferred.data.user.mcoins, 70);
  const duplicateTransfer = await api('/api/messages/transfers', {
    method: 'POST', token: student.token, body: {
      toUserId: teacher.user.user_id, amountMcoins: 30, clientRequestId: 'lesson-payment-1',
    },
  });
  assert.equal(duplicateTransfer.status, 200);
  assert.equal(duplicateTransfer.data.duplicate, true);
  assert.equal(duplicateTransfer.data.user.mcoins, 70);

  const ended = await api(`/api/human-lessons/${mainMeetingId}/end`, {
    method: 'POST', token: teacher.token, body: {},
  });
  assert.equal(ended.status, 200);
  assert.equal(ended.data.meeting.status, 'ended');

  const finalDb = await readDb();
  const finalTeacher = finalDb.users.find((user) => user.id === teacher.user.user_id);
  assert.equal(finalTeacher.mcoins, 38.3);
  assert.equal(finalTeacher.withdrawableMcoins, 30);
  assert.equal(finalDb.mcoinTransfers.length, 1);
  assert.equal(finalDb.ledger.filter((entry) => entry.type === 'human_lesson_call').length, 1);
  assert.equal(finalDb.ledger.filter((entry) => entry.type === 'human_lesson_breakout').length, 1);
});
