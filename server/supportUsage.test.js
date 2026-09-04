'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  SUPPORT_DAILY_QUESTION_LIMIT,
  refundSupportQuestion,
  reserveSupportQuestion,
  supportQuestionAllowance,
} = require('./supportUsage');

test('allows seven daily support questions and resets on the next UTC date', () => {
  const user = {};
  const now = new Date('2026-09-04T23:59:00.000Z');
  for (let index = 0; index < SUPPORT_DAILY_QUESTION_LIMIT; index += 1) {
    const reservation = reserveSupportQuestion(user, { now });
    assert.equal(reservation.allowed, true);
    assert.equal(reservation.allowance.remainingQuestions, SUPPORT_DAILY_QUESTION_LIMIT - index - 1);
  }
  assert.equal(reserveSupportQuestion(user, { now }).allowed, false);
  const reset = supportQuestionAllowance(user, { now: new Date('2026-09-05T00:00:00.000Z') });
  assert.equal(reset.usedQuestions, 0);
  assert.equal(reset.remainingQuestions, SUPPORT_DAILY_QUESTION_LIMIT);
});

test('refunds a failed request without touching a new UTC day', () => {
  const user = {};
  const first = reserveSupportQuestion(user, { now: new Date('2026-09-04T10:00:00.000Z') });
  assert.equal(refundSupportQuestion(user, first), true);
  assert.equal(user.supportQuestionUsage.usedQuestions, 0);
  const oldReservation = reserveSupportQuestion(user, { now: new Date('2026-09-04T23:59:59.000Z') });
  user.supportQuestionUsage = { utcDate: '2026-09-05', usedQuestions: 1 };
  assert.equal(refundSupportQuestion(user, oldReservation), false);
  assert.equal(user.supportQuestionUsage.usedQuestions, 1);
});

test('administrator allowance is unlimited and never mutates usage', () => {
  const user = {};
  const reservation = reserveSupportQuestion(user, {
    now: new Date('2026-09-04T10:00:00.000Z'),
    unlimited: true,
  });
  assert.equal(reservation.allowed, true);
  assert.equal(reservation.reserved, false);
  assert.equal(reservation.allowance.unlimited, true);
  assert.equal(reservation.allowance.remainingQuestions, null);
  assert.equal(user.supportQuestionUsage, undefined);
});
