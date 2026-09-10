import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildAdaptivePianoLayout,
  foldMidiIntoPianellaRange,
  planPianellaRangeShift,
  shouldUseTwoStoreys,
} from '../../src/engine/grandPianoLayout.js';
import { normalizeSong } from '../../src/engine/scheduler.js';
import { parseSongText } from '../../src/utils/songParser.js';

test('compact piano range shifts the whole score right by two octaves when there is room', () => {
  const song = normalizeSong({
    title: 'Compact register fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C2', time: 1, duration: 0.5 },
      { note: 'C4', time: 2, duration: 0.5 },
    ],
  });

  assert.deepEqual(song.notes.map((note) => note.note), ['A2', 'C4', 'C6']);
  assert.deepEqual(song.notes.map((note) => note.octaveShiftSemitones), [24, 24, 24]);
  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 24);
  assert.equal(song.pianoRangeNormalization.shiftedNotes, 3);
  assert.equal(song.pianoRangeNormalization.edgeFoldedNotes, 0);
  assert.equal(foldMidiIntoPianellaRange(60), 60);
});

test('range detection reduces the whole-score shift when C7 has no room', () => {
  const source = {
    notes: [
      { note: 'C3', time: 0, duration: 0.5 },
      { note: 'C6', time: 1, duration: 0.5 },
    ],
  };
  assert.equal(planPianellaRangeShift(source).globalShiftSemitones, 12);
  const song = normalizeSong({
    title: 'Upper register fixture',
    performance: { autoShiftPianoRegister: true },
    ...source,
  });

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 12);
  assert.deepEqual(song.notes.map((note) => note.note), ['C4', 'C7']);
});

test('unavoidable grand-piano edges octave-fold instead of disappearing', () => {
  const song = normalizeSong({
    title: 'Wide score fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C4', time: 1, duration: 0.5 },
      { note: 'C8', time: 2, duration: 0.5 },
    ],
  });

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 24);
  assert.equal(song.pianoRangeNormalization.edgeFoldedNotes, 1);
  assert.deepEqual(song.notes.map((note) => note.note), ['A2', 'C6', 'C7']);
  assert.ok(song.notes.every((note) => note.midi >= 33 && note.midi <= 96));
});

test('normalizing an uploaded score twice never applies the global shift twice', () => {
  const once = normalizeSong({
    title: 'Idempotent fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [{ note: 'C4', time: 0, duration: 0.5 }],
  });
  const twice = normalizeSong(once);

  assert.equal(once.notes[0].note, 'C6');
  assert.equal(twice.notes[0].note, 'C6');
  assert.equal(twice.notes[0].originalMidi, 60);
  assert.equal(twice.pianoRangeNormalization.globalShiftSemitones, 24);
});

test('an edge fold that lands on an occupied key becomes one playable strike', () => {
  const repeatedC5 = Array.from({ length: 100 }, (_, index) => ({
    note: 'C5',
    time: index * 0.5,
    duration: 0.35,
    velocity: 0.7,
  }));
  const song = normalizeSong({
    title: 'Collision fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [
      ...repeatedC5,
      { note: 'C8', time: 0, duration: 0.8, velocity: 0.9 },
    ],
  });

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 24);
  assert.equal(song.pianoRangeNormalization.rangeFoldCollisionsRemoved, 1);
  assert.equal(song.notes.length, 100);
  assert.equal(song.notes[0].note, 'C7');
  assert.equal(song.notes[0].velocity, 0.9);
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

test('built-in songs keep their written middle register unless marked as an upload', () => {
  const song = normalizeSong({
    title: 'Built-in fixture',
    notes: [{ note: 'C4', time: 0, duration: 0.5 }],
  });

  assert.equal(song.notes[0].note, 'C4');
  assert.equal(song.pianoRangeNormalization.mode, 'edge-fold-only');
});

test('uploaded JSON opts into whole-score range detection automatically', () => {
  const song = parseSongText(JSON.stringify({
    title: 'Uploaded JSON fixture',
    notes: [{ note: 'C4', time: 0, duration: 0.5 }],
  }), 'upload.json');

  assert.equal(song.performance.autoShiftPianoRegister, true);
  assert.equal(song.notes[0].note, 'C6');
  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 24);
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
