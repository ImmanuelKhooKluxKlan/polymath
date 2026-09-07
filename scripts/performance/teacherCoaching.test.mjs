import assert from 'node:assert/strict';
import test from 'node:test';
import {
  evaluatePerformanceEvent,
  expectedNotesNear,
  selectObservedNotes,
} from '../../src/engine/teacherCoachingEngine.js';

const song = {
  notes: [
    { note: 'C4', time: 2, duration: 1, velocity: 0.8 },
    { note: 'E4', time: 3.2, duration: 0.4, velocity: 0.7 },
  ],
};

test('finds expected notes around the learner position', () => {
  assert.equal(expectedNotesNear(song, 2.1)[0].note, 'C4');
});
test('measures an early release instead of inventing a correction', () => {
  const result = evaluatePerformanceEvent(song, {
    midi: 60,
    startedSongTime: 2.03,
    duration: 0.61,
    confidence: 0.9,
    source: 'camera-key-motion',
  });
  assert.equal(result.type, 'hold-short');
  assert.equal(result.earlyBySeconds, 0.39);
  assert.match(result.message, /released 0\.39 seconds early/);
});

test('reports a nearby wrong note with the expected pitch', () => {
  const result = evaluatePerformanceEvent(song, { midi: 61, startedSongTime: 2, duration: 0.3 });
  assert.equal(result.type, 'wrong-note');
  assert.deepEqual(result.expectedNotes, ['C4']);
});

test('prefers calibrated visual keys and otherwise uses heard peaks', () => {
  assert.deepEqual(selectObservedNotes({
    vision: { calibrated: true, pressed: [{ midi: 60, score: 0.9 }] },
    hearing: { notes: [{ note: 'D4', confidence: 0.8 }] },
  })[0].source, 'camera-key-motion');
  assert.equal(selectObservedNotes({
    vision: { calibrated: false, pressed: [] },
    hearing: { notes: [{ note: 'D4', confidence: 0.8 }] },
  })[0].midi, 62);
});
