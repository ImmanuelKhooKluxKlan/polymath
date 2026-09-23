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

test('featured library identity survives piano normalization', () => {
  const song = normalizeSong({
    title: 'Administrator song',
    featuredSongId: 'featured_song_123',
    libraryType: 'free',
    notes: [{ note: 'C4', time: 0, duration: 0.5 }],
  });

  assert.equal(song.featuredSongId, 'featured_song_123');
  assert.equal(song.libraryId, 'featured:featured_song_123');
});

test('compact piano range preserves the score and folds only notes below A1', () => {
  const song = normalizeSong({
    title: 'Compact register fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [
      { note: 'A0', time: 0, duration: 0.5 },
      { note: 'C2', time: 1, duration: 0.5 },
      { note: 'C4', time: 2, duration: 0.5 },
    ],
  });

  assert.deepEqual(song.notes.map((note) => note.note), ['A1', 'C2', 'C4']);
  assert.deepEqual(song.notes.map((note) => note.octaveShiftSemitones), [12, 0, 0]);
  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 0);
  assert.equal(song.pianoRangeNormalization.shiftedNotes, 1);
  assert.equal(song.pianoRangeNormalization.edgeFoldedNotes, 1);
  assert.equal(foldMidiIntoPianellaRange(60), 60);
});

test('browser scheduling preserves explicit melody and accompaniment roles', () => {
  const song = normalizeSong({
    title: 'Role preservation fixture',
    notes: [
      { note: 'C3', time: 0, duration: 0.5, arrangementRole: 'accompaniment' },
      { note: 'G4', time: 0, duration: 0.5, arrangementRole: 'melody' },
    ],
  });

  assert.deepEqual(
    song.notes.map((note) => note.arrangementRole),
    ['accompaniment', 'melody'],
  );
});

test('range detection keeps mid and upper notes in their authored octaves', () => {
  const source = {
    notes: [
      { note: 'C3', time: 0, duration: 0.5 },
      { note: 'C6', time: 1, duration: 0.5 },
    ],
  };
  assert.equal(planPianellaRangeShift(source).globalShiftSemitones, 0);
  const song = normalizeSong({
    title: 'Upper register fixture',
    performance: { autoShiftPianoRegister: true },
    ...source,
  });

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 0);
  assert.deepEqual(song.notes.map((note) => note.note), ['C3', 'C6']);
});

test('legacy positive shift settings cannot restore the blanket octave lift', () => {
  const plan = planPianellaRangeShift(
    [{ note: 'A1' }, { note: 'C4' }, { note: 'C8' }],
    { preferredShiftSemitones: 24 },
  );

  assert.equal(plan.preferredShiftSemitones, 0);
  assert.equal(plan.globalShiftSemitones, 0);
  assert.equal(foldMidiIntoPianellaRange(33, plan.globalShiftSemitones), 33);
  assert.equal(foldMidiIntoPianellaRange(60, plan.globalShiftSemitones), 60);
  assert.equal(foldMidiIntoPianellaRange(108, plan.globalShiftSemitones), 108);
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

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 0);
  assert.equal(song.pianoRangeNormalization.edgeFoldedNotes, 1);
  assert.deepEqual(song.notes.map((note) => note.note), ['A1', 'C4', 'C8']);
  assert.ok(song.notes.every((note) => note.midi >= 33 && note.midi <= 108));
});

test('normalizing an uploaded score twice never applies the global shift twice', () => {
  const once = normalizeSong({
    title: 'Idempotent fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [{ note: 'C4', time: 0, duration: 0.5 }],
  });
  const twice = normalizeSong(once);

  assert.equal(once.notes[0].note, 'C4');
  assert.equal(twice.notes[0].note, 'C4');
  assert.equal(twice.notes[0].originalMidi, 60);
  assert.equal(twice.pianoRangeNormalization.globalShiftSemitones, 0);
});

test('an edge fold that lands on an occupied key becomes one playable strike', () => {
  const repeatedA1 = Array.from({ length: 100 }, (_, index) => ({
    note: 'A1',
    time: index * 0.5,
    duration: 0.35,
    velocity: 0.7,
  }));
  const song = normalizeSong({
    title: 'Collision fixture',
    performance: { autoShiftPianoRegister: true },
    notes: [
      ...repeatedA1,
      { note: 'A0', time: 0, duration: 0.8, velocity: 0.9 },
    ],
  });

  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 0);
  assert.equal(song.pianoRangeNormalization.rangeFoldCollisionsRemoved, 1);
  assert.equal(song.notes.length, 100);
  assert.equal(song.notes[0].note, 'A1');
  assert.equal(song.notes[0].velocity, 0.9);
});

test('default piano is one A1-C8 row even when source JSON had a low outlier', () => {
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
  assert.equal(layout.rangeLabel, 'A1-C8');
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
  assert.equal(song.notes[0].note, 'C4');
  assert.equal(song.pianoRangeNormalization.globalShiftSemitones, 0);
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
