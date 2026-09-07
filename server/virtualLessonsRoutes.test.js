'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-virtual-lessons-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'admin@example.test';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.RUNPOD_CHAT_BOSS_ENDPOINT_ID = 'test-chat-endpoint';
process.env.RUNPOD_API_KEY = 'test-runpod-key';

const nativeFetch = globalThis.fetch;
globalThis.fetch = (url, options) => {
  if (String(url).startsWith('https://api.runpod.ai/')) {
    return Promise.resolve(new Response(JSON.stringify({ error: 'simulated teacher outage' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    }));
  }
  return nativeFetch(url, options);
};

const { app, readDb, writeDb } = require('./server');

test('virtual lessons charge once, remember the session, demonstrate exact ranges, and erase chat when ended', async (context) => {
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
        password: 'VirtualLessonPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const student = await register('Maya Student', 'maya@example.test');
  const db = await readDb();
  db.users.find((user) => user.id === student.user.user_id).mcoins = 20;
  await writeDb(db);

  const catalog = await api('/api/virtual-lessons', { token: student.token });
  assert.equal(catalog.status, 200);
  assert.equal(catalog.data.catalog.rateMcoinsPerHour, 10);
  assert.equal(catalog.data.session, null);

  const checkout = {
    durationMinutes: 60,
    clientRequestId: 'checkout_1234567890',
    teacher: { id: 'aria', name: 'Aria', title: 'Piano teacher', style: 'Patient' },
  };
  const started = await api('/api/virtual-lessons', { method: 'POST', token: student.token, body: checkout });
  assert.equal(started.status, 201);
  assert.equal(started.data.chargedMcoins, 10);
  assert.equal(started.data.user.mcoins, 10);
  assert.equal(started.data.session.remainingSeconds > 3500, true);
  assert.match(started.data.greeting, /Hi, Maya\. I'm Aria/i);

  const unavailableSpeech = await api(`/api/virtual-lessons/${started.data.session.id}/speech`, {
    method: 'POST', token: student.token, body: { kind: 'greeting' },
  });
  assert.equal(unavailableSpeech.status, 503);
  assert.match(unavailableSpeech.data.error, /not configured/i);

  const duplicate = await api('/api/virtual-lessons', { method: 'POST', token: student.token, body: checkout });
  assert.equal(duplicate.status, 200);
  assert.equal(duplicate.data.duplicate, true);
  assert.equal(duplicate.data.user.mcoins, 10);

  const overlapping = await api('/api/virtual-lessons', {
    method: 'POST',
    token: student.token,
    body: {
      ...checkout,
      teacher: { id: 'nova', name: 'Padme' },
      clientRequestId: 'checkout_abcdefghij',
    },
  });
  assert.equal(overlapping.status, 409);
  assert.equal(overlapping.data.code, 'VIRTUAL_LESSON_TEACHER_LOCKED');
  assert.equal(overlapping.data.lockedTeacherId, 'aria');
  assert.equal(overlapping.data.session.teacher.id, 'aria');
  assert.equal(overlapping.data.session.teacherSelectionLocked, true);

  const taught = await api(`/api/virtual-lessons/${started.data.session.id}/messages`, {
    method: 'POST',
    token: student.token,
    body: {
      message: 'Show me how your hands move for the first 5 seconds',
      lessonContext: { title: 'Mean', duration: 220, currentTime: 70 },
      observations: {},
    },
  });
  assert.equal(taught.status, 200);
  assert.equal(taught.data.provider, 'polymath-demonstration-engine');
  assert.deepEqual(taught.data.action, {
    type: 'demonstrate_range', startSeconds: 0, endSeconds: 5, hand: 'both', speed: null,
  });
  assert.equal(taught.data.session.messages.length, 2);
  assert.equal(taught.data.session.memory.lastSong, 'Mean');

  // The paid clock must not consume the learner's time when the language
  // model cannot return a reply. The unacknowledged message is also removed
  // so temporary memory remains a truthful record of completed exchanges.
  await new Promise((resolve) => setTimeout(resolve, 1450));
  const failedTeacher = await api(`/api/virtual-lessons/${started.data.session.id}/messages`, {
    method: 'POST',
    token: student.token,
    body: {
      message: 'My goal is to master arpeggios.',
      lessonContext: { title: 'Another song', duration: 220, currentTime: 5 },
      observations: {},
    },
  });
  assert.equal(failedTeacher.status, 502);
  assert.equal(failedTeacher.data.recoveredSeconds >= 15, true);
  assert.equal(failedTeacher.data.session.messages.length, 2);
  assert.equal(failedTeacher.data.session.memory.lastSong, 'Mean');
  assert.equal(failedTeacher.data.session.memory.goal, '');

  const ended = await api(`/api/virtual-lessons/${started.data.session.id}/end`, {
    method: 'POST', token: student.token, body: {},
  });
  assert.equal(ended.status, 200);
  assert.equal(ended.data.session.status, 'ended');
  assert.deepEqual(ended.data.session.messages, []);

  const finalDb = await readDb();
  const charges = finalDb.ledger.filter((entry) => entry.userId === student.user.user_id && entry.type === 'virtual_lesson');
  assert.equal(charges.length, 1);
  assert.equal(charges[0].amount, -10);
  assert.equal(finalDb.virtualLessonSessions[0].memory, null);

  const freeStudent = await register('No Balance', 'empty@example.test');
  const insufficient = await api('/api/virtual-lessons', {
    method: 'POST',
    token: freeStudent.token,
    body: { ...checkout, clientRequestId: 'checkout_no_balance_1', durationMinutes: 30 },
  });
  assert.equal(insufficient.status, 402);
  assert.equal(insufficient.data.requiredMcoins, 5);

  const administrator = await register('Site Admin', 'admin@example.test');
  const policyUpdate = await api('/api/admin/policies', {
    method: 'PUT',
    token: administrator.token,
    body: {
      virtualLessonPricePer30MinutesMcoins: 3.99,
    },
  });
  assert.equal(policyUpdate.status, 200);
  assert.equal(policyUpdate.data.policies.virtualLessonPricePer30MinutesMcoins, 3.99);
  const pricedCatalog = await api('/api/virtual-lessons', { token: administrator.token });
  assert.equal(pricedCatalog.data.catalog.pricePer30MinutesMcoins, 3.99);
  assert.equal(pricedCatalog.data.catalog.durationStepMinutes, 30);

  const pricedStudent = await register('Priced Student', 'priced@example.test');
  const pricedDb = await readDb();
  pricedDb.users.find((user) => user.id === pricedStudent.user.user_id).mcoins = 10;
  await writeDb(pricedDb);
  const pricedLesson = await api('/api/virtual-lessons', {
    method: 'POST',
    token: pricedStudent.token,
    body: {
      ...checkout,
      durationMinutes: 44,
      clientRequestId: 'checkout_custom_price_1',
    },
  });
  assert.equal(pricedLesson.status, 201);
  assert.equal(pricedLesson.data.chargedMcoins, 3.99);
  assert.equal(pricedLesson.data.session.durationMinutes, 30);

  const characterPricing = await api('/api/admin/virtual-teachers/aria', {
    method: 'PATCH',
    token: administrator.token,
    body: {
      minimumAge: 16,
      pricePer30MinutesMcoins: 1.25,
    },
  });
  assert.equal(characterPricing.status, 200);
  assert.equal(characterPricing.data.character.minimumAge, 16);
  assert.equal(characterPricing.data.character.pricePer30MinutesMcoins, 1.25);
  const ageGatedStudent = await register('Age Gated Student', 'age-gated@example.test');
  const ageGatedDb = await readDb();
  ageGatedDb.users.find((user) => user.id === ageGatedStudent.user.user_id).mcoins = 10;
  await writeDb(ageGatedDb);
  const ageGatedCheckout = {
    ...checkout,
    durationMinutes: 30,
    clientRequestId: 'checkout_age_gate_1',
  };
  const ageRejected = await api('/api/virtual-lessons', {
    method: 'POST', token: ageGatedStudent.token, body: ageGatedCheckout,
  });
  assert.equal(ageRejected.status, 403);
  assert.equal(ageRejected.data.minimumAge, 16);
  const ageConfirmed = await api('/api/virtual-lessons', {
    method: 'POST',
    token: ageGatedStudent.token,
    body: { ...ageGatedCheckout, clientRequestId: 'checkout_age_gate_2', confirmedAge: 16 },
  });
  assert.equal(ageConfirmed.status, 201);
  assert.equal(ageConfirmed.data.chargedMcoins, 1.25);
  assert.equal(ageConfirmed.data.user.mcoins, 8.75);

  const companionStudent = await register('Adult Companion Student', 'companion@example.test');
  const companionDb = await readDb();
  companionDb.users.find((user) => user.id === companionStudent.user.user_id).mcoins = 10;
  await writeDb(companionDb);
  const companionCheckout = {
    durationMinutes: 30,
    clientRequestId: 'checkout_companion_1',
    teacher: { id: 'nova', name: 'Spoofed name', requiresAdultConfirmation: false },
    conversationMode: 'adult-companion',
    conversationPreferences: { companionStyle: 'playful' },
  };
  const unconfirmedCompanion = await api('/api/virtual-lessons', {
    method: 'POST', token: companionStudent.token, body: companionCheckout,
  });
  assert.equal(unconfirmedCompanion.status, 403);
  const confirmedCompanion = await api('/api/virtual-lessons', {
    method: 'POST',
    token: companionStudent.token,
    body: { ...companionCheckout, adultConfirmed: true, companionConsent: true },
  });
  assert.equal(confirmedCompanion.status, 201);
  assert.equal(confirmedCompanion.data.session.teacher.name, 'Padme');
  assert.equal(confirmedCompanion.data.session.conversationMode, 'adult-companion');
  assert.equal(confirmedCompanion.data.session.adultCompanionConfirmed, true);
  assert.equal(confirmedCompanion.data.greeting, "Oh, hi, sweetheart. I'm Padme. Come sit with me. What kind of mood are you in today?");

  const adminLesson = await api('/api/virtual-lessons', {
    method: 'POST',
    token: administrator.token,
    body: { ...checkout, clientRequestId: 'checkout_admin_12345', durationMinutes: 120 },
  });
  assert.equal(adminLesson.status, 201);
  assert.equal(adminLesson.data.chargedMcoins, 0);
  assert.equal(adminLesson.data.user.unlimitedMcoins, true);
});
