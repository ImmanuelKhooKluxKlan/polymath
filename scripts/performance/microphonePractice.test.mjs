import test from 'node:test';
import assert from 'node:assert/strict';
import {
  advancePracticeReadingStability,
  buildPracticeOnsetGroups,
  classifyPracticeReading,
  createPracticeReadingStability,
  findPracticeOnsetGroup,
} from '../../src/engine/microphonePractice.js';
import { analyzePracticeAttempt } from '../../src/engine/learningCoach.js';

function reading(midi, overrides = {}) {
  return {
    heard: true,
    midi,
    note: midi === 60 ? 'C4' : midi === 62 ? 'D4' : 'E4',
    cents: 2,
    confidence: 0.94,
    rms: 0.12,
    dbfs: -18,
    notes: [{ midi, confidence: 0.94, dbfs: -18 }],
    ...overrides,
  };
}

test('practice notes at the same onset become one correction target', () => {
  const groups = buildPracticeOnsetGroups([
    { id: 'c', note: 'C4', time: 1, duration: 0.4 },
    { id: 'e', note: 'E4', time: 1.035, duration: 0.4 },
    { id: 'd', note: 'D4', time: 1.4, duration: 0.4 },
  ], { start: 0.5, end: 2 });
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[0].midis, [60, 64]);
  assert.deepEqual(groups[0].eventIds, ['c', 'e']);
});

test('nearest unresolved onset is selected inside the timing window', () => {
  const groups = buildPracticeOnsetGroups([
    { id: 'c', note: 'C4', time: 1 },
    { id: 'd', note: 'D4', time: 1.5 },
  ]);
  assert.equal(findPracticeOnsetGroup(groups, 1.39)?.id, groups[1].id);
  assert.equal(findPracticeOnsetGroup(groups, 1.39, { resolvedIds: new Set([groups[1].id]) })?.id, groups[0].id);
  assert.equal(findPracticeOnsetGroup(groups, 3), null);
});

test('only confident stable pitches can become wrong-note corrections', () => {
  assert.equal(classifyPracticeReading(reading(60), [60]).type, 'correct');
  assert.equal(classifyPracticeReading(reading(60), [60, 64]).type, 'correct');
  assert.equal(classifyPracticeReading(reading(62), [60]).type, 'wrong');
  assert.equal(classifyPracticeReading(reading(48, {
    notes: [
      { midi: 48, confidence: 1, dbfs: -18 },
      { midi: 60, confidence: 0.55, dbfs: -25 },
    ],
  }), [60]).type, 'wrong', 'a strong octave harmonic must not pass as the played key');
  assert.equal(classifyPracticeReading(reading(62, { confidence: 0.34 }), [60]).type, 'uncertain');
  assert.equal(classifyPracticeReading({ heard: false, rms: 0, notes: [] }, [60]).type, 'silence');
});

test('stability filter emits one attack, waits for release, then permits a repeated note', () => {
  let state = createPracticeReadingStability();
  let result = advancePracticeReadingStability(state, reading(60));
  state = result.state;
  assert.equal(result.attack, null);
  result = advancePracticeReadingStability(state, reading(60));
  state = result.state;
  assert.equal(result.attack?.candidates[0].midi, 60);
  result = advancePracticeReadingStability(state, reading(60));
  state = result.state;
  assert.equal(result.attack, null, 'a held key must not retrigger');
  result = advancePracticeReadingStability(state, { heard: false, rms: 0, notes: [] });
  state = result.state;
  assert.equal(result.attack, null);
  result = advancePracticeReadingStability(state, reading(60));
  state = result.state;
  assert.equal(result.attack, null);
  result = advancePracticeReadingStability(state, reading(60));
  assert.equal(result.attack?.candidates[0].midi, 60);
});

test('microphone-only attempts do not invent hold or dynamics scores', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: [{ id: 'expected-c', note: 'C4', time: 1, duration: 1, velocity: 0.7 }],
    playedNotes: [{
      note: 'C4',
      start: 1.01,
      duration: 0.12,
      velocity: 0.9,
      inputType: 'microphone',
      durationCapable: false,
      dynamicCapable: false,
    }],
    range: { start: 0, end: 2 },
  });
  assert.equal(report.metrics.notes.score, 100);
  assert.equal(report.metrics.duration.available, false);
  assert.equal(report.metrics.dynamics.available, false);
});
