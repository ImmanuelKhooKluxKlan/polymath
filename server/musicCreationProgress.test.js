'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  describeMusicCreationProgress,
  durationWindow,
} = require('./musicCreationProgress');

const NOW = Date.parse('2026-09-09T10:01:00.000Z');

test('reports bounded queue and composing progress with an honest time range', () => {
  const queued = describeMusicCreationProgress({
    kind: 'draft',
    status: 'IN_QUEUE',
    createdAt: '2026-09-09T10:00:50.000Z',
  }, [], NOW);
  assert.equal(queued.elapsedSeconds, 10);
  assert.ok(queued.percent >= 5 && queued.percent <= 19);
  assert.ok(queued.remainingHighSeconds > queued.remainingLowSeconds);

  const composing = describeMusicCreationProgress({
    kind: 'draft',
    status: 'IN_PROGRESS',
    createdAt: '2026-09-09T10:00:20.000Z',
  }, [], NOW);
  assert.equal(composing.elapsedSeconds, 40);
  assert.ok(composing.percent >= 24 && composing.percent <= 94);
});

test('marks long jobs as active beyond the estimate instead of pretending they finished', () => {
  const progress = describeMusicCreationProgress({
    kind: 'draft',
    status: 'IN_PROGRESS',
    createdAt: '2026-09-09T09:58:00.000Z',
  }, [], NOW);
  assert.equal(progress.overEstimate, true);
  assert.equal(progress.percent, 94);
  assert.equal(progress.remainingHighSeconds, 0);
});

test('learns an estimate window from successful jobs', () => {
  const jobs = [40, 50, 60, 70].map((seconds, index) => ({
    id: `completed-${index}`,
    kind: 'draft',
    status: 'COMPLETED',
    blueprint: { title: 'Ready' },
    createdAt: '2026-09-09T10:00:00.000Z',
    completedAt: new Date(Date.parse('2026-09-09T10:00:00.000Z') + (seconds * 1000)).toISOString(),
  }));
  const window = durationWindow({ kind: 'draft' }, jobs);
  assert.equal(window.learned, true);
  assert.equal(window.sampleCount, 4);
  assert.ok(window.low >= 15);
  assert.ok(window.high > window.low);
});

test('terminal jobs show a complete bar', () => {
  const progress = describeMusicCreationProgress({
    kind: 'draft',
    status: 'COMPLETED',
    createdAt: '2026-09-09T10:00:00.000Z',
  }, [], NOW);
  assert.equal(progress.percent, 100);
});
