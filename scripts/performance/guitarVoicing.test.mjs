import test from 'node:test';
import assert from 'node:assert/strict';
import { assignNotesToStrings, guitarCandidatesForMidi } from '../../src/engine/guitarVoicing.js';

test('guitar candidate positions stay inside six strings and 24 frets', () => {
  assert.deepEqual(guitarCandidatesForMidi(40), [{ stringIndex: 0, fret: 0 }]);
  assert.ok(guitarCandidatesForMidi(64).some(({ stringIndex, fret }) => stringIndex === 5 && fret === 0));
  assert.deepEqual(guitarCandidatesForMidi(100), []);
});

test('global fingering keeps a six-note MIDI chord without reusing a string', () => {
  const notes = [40, 47, 52, 55, 59, 64].map((midi) => ({ midi, duration: 1, velocity: 0.8 }));
  const assignments = assignNotesToStrings(notes);
  assert.equal(assignments.length, 6);
  assert.equal(new Set(assignments.map(({ stringIndex }) => stringIndex)).size, 6);
  assert.ok(assignments.some(({ midi }) => midi === 40));
  assert.ok(assignments.some(({ midi }) => midi === 64));
});

test('oversized score chord preserves melody and bass while selecting six playable notes', () => {
  const assignments = assignNotesToStrings([40, 45, 48, 52, 55, 59, 64, 67].map((midi) => ({
    midi, duration: 0.8, velocity: 0.75,
  })));
  assert.equal(assignments.length, 6);
  assert.ok(assignments.some(({ midi }) => midi === 40));
  assert.ok(assignments.some(({ midi }) => midi === 67));
});
