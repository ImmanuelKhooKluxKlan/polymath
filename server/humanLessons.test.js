const test = require('node:test');
const assert = require('node:assert/strict');

const {
  BREAKOUT_ROOM_COST_MCOINS,
  MINIMUM_CHAT_TRANSFER_MCOINS,
  accessCodeMatches,
  createAccessCode,
  createMeetingCode,
  hashAccessCode,
  lessonCallCost,
  normalizeLessonDuration,
  normalizeTransferAmount,
  openAccessCode,
  sealAccessCode,
} = require('./humanLessons');

test('human lesson pricing uses exact 10-minute blocks', () => {
  assert.equal(normalizeLessonDuration(10), 10);
  assert.equal(normalizeLessonDuration(60), 60);
  assert.equal(lessonCallCost(10), 0.2);
  assert.equal(lessonCallCost(60), 1.2);
  assert.equal(BREAKOUT_ROOM_COST_MCOINS, 0.5);
  assert.throws(() => normalizeLessonDuration(15), /10-minute intervals/);
  assert.throws(() => normalizeLessonDuration(0), /10-minute intervals/);
});

test('meeting IDs and access codes are readable and collision-aware', () => {
  const first = createMeetingCode();
  const second = createMeetingCode(new Set([first]));
  const accessCode = createAccessCode();
  assert.match(first, /^PM-[A-Z2-9]{4}-[A-Z2-9]{4}$/);
  assert.notEqual(first, second);
  assert.match(accessCode, /^[A-Z2-9]{4}-[A-Z2-9]{4}$/);
});

test('lesson access codes are hashed for verification and encrypted for host retrieval', () => {
  const secret = 'test-secret-at-least-long-enough-for-a-key';
  const accessCode = createAccessCode();
  const credential = hashAccessCode(accessCode);
  const sealed = sealAccessCode(accessCode, secret);
  assert.equal(accessCodeMatches(accessCode.toLowerCase(), credential.salt, credential.hash), true);
  assert.equal(accessCodeMatches('AAAA-AAAA', credential.salt, credential.hash), false);
  assert.equal(openAccessCode(sealed, secret), accessCode);
  assert.doesNotMatch(sealed, new RegExp(accessCode));
  assert.throws(() => openAccessCode(sealed, 'wrong-secret'), /can no longer be opened/);
});

test('chat transfers enforce the 30-Mcoin minimum without adding a fee', () => {
  assert.equal(MINIMUM_CHAT_TRANSFER_MCOINS, 30);
  assert.equal(normalizeTransferAmount(30), 30);
  assert.equal(normalizeTransferAmount(30.129), 30.13);
  assert.throws(() => normalizeTransferAmount(29.99), /30 to/);
});
