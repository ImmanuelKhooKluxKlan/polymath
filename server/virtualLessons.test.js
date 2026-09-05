'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  activeVirtualLesson,
  appendSessionMessage,
  createVirtualLesson,
  endVirtualLesson,
  expireVirtualLessons,
  lessonCatalog,
  lessonQuote,
  normalizeConversationMode,
  parseTeacherDemonstration,
  publicVirtualLesson,
  updateSessionMemory,
} = require('./virtualLessons');

function lesson(overrides = {}) {
  return createVirtualLesson({
    id: 'lesson_1',
    userId: 'user_1',
    clientRequestId: 'request_1234567890',
    durationMinutes: 60,
    priceMcoins: 10,
    teacher: { id: 'aria', name: 'Aria' },
    studentName: 'Maya',
    now: new Date('2026-09-04T10:00:00.000Z'),
    ...overrides,
  });
}

test('quotes 30-minute blocks and rounds manual durations to the nearest block', () => {
  const catalog = lessonCatalog();
  assert.equal(catalog.durationStepMinutes, 30);
  assert.equal(catalog.minimumDurationMinutes, 30);
  assert.equal(catalog.maximumDurationMinutes, 720);
  assert.equal(catalog.pricePer30MinutesMcoins, 5);
  assert.equal(lessonQuote(30).priceMcoins, 5);
  assert.equal(lessonQuote(60).priceMcoins, 10);
  assert.equal(lessonQuote(44).durationMinutes, 30);
  assert.equal(lessonQuote(44).rounded, true);
  assert.equal(lessonQuote(46).durationMinutes, 60);
  assert.equal(lessonQuote(46, 7.99).priceMcoins, 15.98);
  assert.equal(lessonQuote(900).durationMinutes, 720);
  assert.equal(lessonQuote('not-a-number'), null);
});

test('requires explicit adult confirmation and consent for companion mode', () => {
  assert.equal(normalizeConversationMode('anything'), 'music-coach');
  assert.throws(() => lesson({
    conversationMode: 'adult-companion',
    teacher: { id: 'nova', name: 'Padme', requiresAdultConfirmation: true },
  }), /18\+ confirmation/);
  const companion = lesson({
    conversationMode: 'adult-companion',
    conversationPreferences: { preferredName: 'May', companionStyle: 'confident' },
    teacher: { id: 'nova', name: 'Padme', requiresAdultConfirmation: true },
    adultConfirmed: true,
    companionConsent: true,
  });
  const publicSession = publicVirtualLesson(companion, new Date('2026-09-04T10:00:01.000Z'));
  assert.equal(publicSession.conversationMode, 'adult-companion');
  assert.equal(publicSession.adultCompanionConfirmed, true);
  assert.equal(publicSession.memory.preferredName, 'May');
  assert.equal(publicSession.conversationPreferences.companionStyle, 'confident');
});

test('creates a server-timed session and returns only safe current memory', () => {
  const session = lesson();
  assert.equal(session.expiresAt, '2026-09-04T11:00:00.000Z');
  appendSessionMessage(session, { id: 'message_1', role: 'user', text: 'Help me with timing.' }, new Date('2026-09-04T10:00:05.000Z'));
  updateSessionMemory(session, {
    studentName: 'Maya',
    userMessage: 'My goal is to play this smoothly',
    lessonContext: { title: 'Mean' },
    practiceReport: { focus: 'Rhythm' },
  });
  const publicSession = publicVirtualLesson(session, new Date('2026-09-04T10:30:00.000Z'));
  assert.equal(publicSession.remainingSeconds, 1800);
  assert.equal(publicSession.teacherSelectionLocked, true);
  assert.equal(publicSession.lockedTeacherId, 'aria');
  assert.equal(publicSession.messages[0].text, 'Help me with timing.');
  assert.equal(publicSession.memory.goal, 'to play this smoothly');
  assert.equal(publicSession.memory.lastSong, 'Mean');
  updateSessionMemory(session, {
    userMessage: 'Call me May. I play guitar. My favorite genre is jazz. Outside music I like astronomy.',
  });
  const personalized = publicVirtualLesson(session, new Date('2026-09-04T10:31:00.000Z'));
  assert.equal(personalized.memory.preferredName, 'May');
  assert.deepEqual(personalized.memory.instruments, ['guitar']);
  assert.deepEqual(personalized.memory.genres, ['jazz']);
  assert.deepEqual(personalized.memory.interests, ['astronomy']);
});

test('expired and manually ended sessions erase conversation memory', () => {
  const expired = lesson();
  appendSessionMessage(expired, { id: 'message_1', role: 'user', text: 'Private lesson message' }, new Date('2026-09-04T10:00:05.000Z'));
  const db = { virtualLessonSessions: [expired] };
  assert.equal(expireVirtualLessons(db, new Date('2026-09-04T11:00:01.000Z')), true);
  assert.equal(expired.status, 'expired');
  assert.deepEqual(expired.messages, []);
  assert.equal(expired.memory, null);
  assert.equal(activeVirtualLesson(db, 'user_1', new Date('2026-09-04T11:00:01.000Z')), null);
  assert.equal(publicVirtualLesson(expired, new Date('2026-09-04T11:00:01.000Z')).teacherSelectionLocked, false);

  const ended = lesson({ id: 'lesson_2' });
  endVirtualLesson(ended, new Date('2026-09-04T10:05:00.000Z'));
  assert.deepEqual(ended.messages, []);
  assert.equal(ended.memory, null);
});

test('public lesson history never exposes a previously stored model scratchpad', () => {
  const session = lesson();
  appendSessionMessage(session, {
    id: 'message_user',
    role: 'user',
    text: 'Can you hear me?',
  }, new Date('2026-09-04T10:00:05.000Z'));
  appendSessionMessage(session, {
    id: 'message_reasoning_only',
    role: 'assistant',
    text: 'Thinking Process: **Analyze the Request:** connection check **Constraint:** be concise',
  }, new Date('2026-09-04T10:00:06.000Z'));
  appendSessionMessage(session, {
    id: 'message_with_final',
    role: 'assistant',
    text: 'Thinking Process: **Analyze the Request:** connection check **Constraint:** flirt **Final Answer:** Yes, sweetheart—I hear you.',
  }, new Date('2026-09-04T10:00:07.000Z'));

  const visible = publicVirtualLesson(session, new Date('2026-09-04T10:00:08.000Z')).messages;
  assert.equal(visible.some((message) => /Thinking Process|Analyze the Request/.test(message.text)), false);
  assert.deepEqual(visible.map((message) => message.text), [
    'Can you hear me?',
    'Yes, sweetheart—I hear you.',
  ]);
});

test('parses exact teacher demonstrations and safely bounds long requests', () => {
  assert.deepEqual(
    parseTeacherDemonstration('Show me how your hands move for the first 5 seconds', { duration: 200 }),
    { type: 'demonstrate_range', startSeconds: 0, endSeconds: 5, hand: 'both', speed: null },
  );
  assert.deepEqual(
    parseTeacherDemonstration('Play the left hand slowly from 0:37 to 0:45', { duration: 200 }),
    { type: 'demonstrate_range', startSeconds: 37, endSeconds: 45, hand: 'left', speed: 0.55 },
  );
  assert.equal(parseTeacherDemonstration('What does legato mean?', { duration: 200 }), null);
  assert.equal(
    parseTeacherDemonstration('Demonstrate from 10 seconds to 100 seconds', { duration: 200 }).endSeconds,
    40,
  );
});

test('repeat and slower commands reuse only the current session demonstration', () => {
  const previous = { type: 'demonstrate_range', startSeconds: 12, endSeconds: 18, hand: 'right', speed: null };
  assert.deepEqual(
    parseTeacherDemonstration('Show that again, slower', { duration: 100 }, previous),
    { type: 'demonstrate_range', startSeconds: 12, endSeconds: 18, hand: 'both', speed: 0.55 },
  );
});
