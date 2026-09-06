import assert from 'node:assert/strict';
import test from 'node:test';
import { buildLearningMomentum } from '../../src/engine/learningMomentum.js';

function progress(createdAtValues) {
  return {
    history: createdAtValues.map((createdAt, index) => ({
      id: `attempt_${index}`,
      createdAt,
      elapsedSeconds: 20,
    })),
  };
}

test('today’s first measured attempt completes the small daily win', () => {
  const momentum = buildLearningMomentum(progress([
    '2026-09-06T05:00:00.000Z',
  ]), {
    now: new Date('2026-09-06T12:00:00.000Z'),
    timeZone: 'Asia/Singapore',
  });

  assert.equal(momentum.todayComplete, true);
  assert.equal(momentum.todayAttempts, 1);
  assert.equal(momentum.streakDays, 1);
  assert.equal(momentum.activeDaysThisWeek, 1);
});

test('an unfinished day preserves yesterday’s streak without pretending today is complete', () => {
  const momentum = buildLearningMomentum(progress([
    '2026-09-03T14:00:00.000Z',
    '2026-09-04T14:00:00.000Z',
    '2026-09-05T14:00:00.000Z',
  ]), {
    now: new Date('2026-09-06T04:00:00.000Z'),
    timeZone: 'Asia/Singapore',
  });

  assert.equal(momentum.todayComplete, false);
  assert.equal(momentum.streakAtRisk, true);
  assert.equal(momentum.streakDays, 3);
  assert.equal(momentum.nextMilestone, 7);
});

test('the calendar uses the learner’s time zone around UTC midnight', () => {
  const momentum = buildLearningMomentum(progress([
    '2026-09-05T16:15:00.000Z',
  ]), {
    now: new Date('2026-09-05T17:00:00.000Z'),
    timeZone: 'Asia/Singapore',
  });

  assert.equal(momentum.todayKey, '2026-09-06');
  assert.equal(momentum.todayComplete, true);
  assert.equal(momentum.week.at(-1).today, true);
});

test('missing history produces an encouraging zero state', () => {
  const momentum = buildLearningMomentum(null, {
    now: new Date('2026-09-06T12:00:00.000Z'),
    timeZone: 'UTC',
  });

  assert.equal(momentum.todayAttempts, 0);
  assert.equal(momentum.streakDays, 0);
  assert.equal(momentum.streakAtRisk, false);
  assert.equal(momentum.week.length, 7);
});
