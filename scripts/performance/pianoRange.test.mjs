import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildAdaptivePianoLayout,
  foldMidiIntoPianellaRange,
  shouldUseTwoStoreys,
} from '../../src/engine/grandPianoLayout.js';
import { normalizeSong } from '../../src/engine/scheduler.js';

test('compact piano range lifts only outliers and keeps ordinary notes unchanged', () => {
  const song = normalizeSong({
    title: 'Compact register fixture',
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C4', time: 1, duration: 0.5 },
      { note: 'C8', time: 2, duration: 0.5 },
    ],
  });

  assert.deepEqual(song.notes.map((note) => note.note), ['A2', 'C4', 'C7']);
  assert.deepEqual(song.notes.map((note) => note.octaveShiftSemitones), [24, 0, -12]);
  assert.equal(song.pianoRangeNormalization.shiftedNotes, 2);
  assert.equal(foldMidiIntoPianellaRange(60), 60);
});

test('default piano is one A1-C7 row even when source JSON had grand-piano outliers', () => {
  const song = normalizeSong({
    title: 'Single row fixture',
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C8', time: 1, duration: 0.5 },
    ],
  });
  const layout = buildAdaptivePianoLayout(song);

  assert.equal(shouldUseTwoStoreys(song), false);
  assert.equal(layout.isTwoStorey, false);
  assert.equal(layout.rows.length, 1);
  assert.equal(layout.rangeLabel, 'A1-C7');
  assert.ok(song.notes.every((note) => layout.getPosition(note.midi)));
});

test('an explicitly preserved grand score remains one full single row', () => {
  const song = normalizeSong({
    title: 'Full grand fixture',
    performance: { preserveFullGrandRange: true },
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C8', time: 1, duration: 0.5 },
    ],
  });
  const layout = buildAdaptivePianoLayout(song);

  assert.deepEqual(song.notes.map((note) => note.note), ['A0', 'C8']);
  assert.equal(layout.isTwoStorey, false);
  assert.equal(layout.rangeLabel, 'A0-C8');
});
