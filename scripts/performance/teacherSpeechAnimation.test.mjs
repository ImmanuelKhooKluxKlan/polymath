import assert from 'node:assert/strict';
import test from 'node:test';
import {
  restingSpeechFrame,
  speechAnimationFrame,
  spokenTokenLength,
  teacherVisemeAt,
} from '../../src/engine/teacherSpeechAnimation.js';

test('maps common English mouth families to stable visemes', () => {
  assert.equal(teacherVisemeAt('music', 0), 'closed');
  assert.equal(teacherVisemeAt('open', 0), 'round');
  assert.equal(teacherVisemeAt('apple', 0), 'open');
  assert.equal(teacherVisemeAt('feel', 0), 'teeth');
  assert.equal(teacherVisemeAt('cheer', 0), 'narrow');
  assert.equal(teacherVisemeAt('easy', 0), 'wide');
});

test('uses real speech boundaries as bounded local animation anchors', () => {
  const text = 'Play slowly now.';
  assert.equal(spokenTokenLength(text, 0), 4);
  const start = speechAnimationFrame({ text, boundaryIndex: 0, boundaryLength: 4, boundaryAtMs: 1000, nowMs: 1000, rate: 1 });
  assert.equal(start.active, true);
  assert.equal(start.charIndex, 0);
  assert.equal(start.viseme, 'closed');
  const afterWord = speechAnimationFrame({ text, boundaryIndex: 0, boundaryLength: 4, boundaryAtMs: 1000, nowMs: 1600, rate: 1 });
  assert.equal(afterWord.active, false);
  const nextBoundary = speechAnimationFrame({ text, boundaryIndex: 5, boundaryLength: 6, boundaryAtMs: 1600, nowMs: 1600, rate: 1 });
  assert.equal(nextBoundary.charIndex, 5);
  assert.equal(nextBoundary.active, true);
});

test('rests when speech is stopped', () => {
  assert.deepEqual(speechAnimationFrame({ text: 'Hello', speaking: false }), restingSpeechFrame());
});
