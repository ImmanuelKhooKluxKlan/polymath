const assert = require('node:assert/strict');
const test = require('node:test');
const {
  PUBLIC_PRODUCT_EVENT_NAMES,
  sanitizeProductEventBatch,
  summarizeProductEvents,
} = require('./productAnalytics');

test('product analytics accepts only the bounded privacy-safe event schema', () => {
  const [event] = sanitizeProductEventBatch([{
    eventId: 'event_12345678',
    eventName: 'learning_attempt_completed',
    occurredAt: '2026-09-06T10:00:00.000Z',
    anonymousId: 'anon_12345678',
    sessionId: 'session_12345678',
    path: 'studio?token=secret',
    release: 'commit-123',
    properties: {
      score: 88,
      freePreview: true,
      songTitle: 'Private song title',
      email: 'person@example.com',
    },
  }], { userId: 'user_123', now: Date.parse('2026-09-06T10:01:00.000Z') });

  assert.equal(event.eventName, 'learning_attempt_completed');
  assert.equal(event.userId, 'user_123');
  assert.equal(event.path, 'studio');
  assert.deepEqual(event.properties, { score: 88, freePreview: true });
  assert.equal('email' in event.properties, false);
  assert.equal('songTitle' in event.properties, false);
  assert.deepEqual(sanitizeProductEventBatch([{ eventName: 'invented_event' }]), []);
});

test('public events cannot manufacture payments or transcription success', () => {
  const events = sanitizeProductEventBatch([{
    eventId: 'event_fake_payment_123',
    eventName: 'subscription_activated',
    anonymousId: 'anon_fake_payment_123',
    sessionId: 'session_fake_payment_123',
  }], { allowedEventNames: PUBLIC_PRODUCT_EVENT_NAMES });
  assert.deepEqual(events, []);
});

test('product summary reports activation, sharing, payment, and returning users', () => {
  const base = {
    anonymousId: 'anon_12345678',
    sessionId: 'session_12345678',
    path: 'studio',
    release: 'test',
    properties: {},
  };
  const events = [
    { ...base, eventId: 'event_00000001', eventName: 'app_opened', userId: 'user_1', occurredAt: '2026-09-01T10:00:00Z' },
    { ...base, eventId: 'event_00000002', eventName: 'learning_attempt_started', userId: 'user_1', occurredAt: '2026-09-01T10:01:00Z' },
    { ...base, eventId: 'event_00000003', eventName: 'learning_attempt_completed', userId: 'user_1', occurredAt: '2026-09-01T10:02:00Z', properties: { score: 84 } },
    { ...base, eventId: 'event_00000004', eventName: 'learning_win_shared', userId: 'user_1', occurredAt: '2026-09-01T10:03:00Z' },
    { ...base, eventId: 'event_00000005', eventName: 'checkout_started', userId: 'user_1', occurredAt: '2026-09-01T10:04:00Z' },
    { ...base, eventId: 'event_00000006', eventName: 'subscription_activated', userId: 'user_1', occurredAt: '2026-09-01T10:05:00Z' },
    { ...base, eventId: 'event_00000007', eventName: 'app_opened', userId: 'user_1', occurredAt: '2026-09-02T10:00:00Z' },
  ];
  const summary = summarizeProductEvents(events, 30);
  assert.equal(summary.stages.find((stage) => stage.id === 'paid').actors, 1);
  assert.equal(summary.learning.averageScore, 84);
  assert.equal(summary.returnSignal.returningActors, 1);
  assert.equal(summary.returnSignal.returningPercent, 100);
});
