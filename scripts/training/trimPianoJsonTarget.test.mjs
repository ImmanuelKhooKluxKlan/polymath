import assert from 'node:assert/strict';
import test from 'node:test';

import { trimPianoJsonTarget } from './trimPianoJsonTarget.mjs';

test('keeps only trusted onsets and clips a note crossing the end boundary', () => {
  const source = {
    title: 'Reference',
    notes: [
      { midi: 60, time: 9, duration: 2 },
      { midi: 62, time: 10, duration: 2 },
      { midi: 64, time: 14.5, duration: 2 },
      { midi: 65, time: 15, duration: 1 },
    ],
  };
  const result = trimPianoJsonTarget(source, {
    startSeconds: 10,
    endSeconds: 15,
    reason: 'Reviewed boundary',
  });

  assert.deepEqual(result.notes.map((note) => note.midi), [62, 64]);
  assert.deepEqual(result.notes.map((note) => note.time), [0, 4.5]);
  assert.equal(result.notes[1].duration, 0.5);
  assert.equal(result.trainingExcerpt.sourceNoteCount, 4);
  assert.equal(result.trainingExcerpt.retainedNoteCount, 2);
  assert.equal(result.trainingExcerpt.reason, 'Reviewed boundary');
  assert.deepEqual(source.notes.map((note) => note.time), [9, 10, 14.5, 15]);
});
test('rejects an empty or inverted excerpt', () => {
  assert.throws(
    () => trimPianoJsonTarget({ notes: [{ midi: 60, time: 1, duration: 1 }] }, {
      startSeconds: 2,
      endSeconds: 1,
    }),
    /endSeconds/,
  );
  assert.throws(
    () => trimPianoJsonTarget({ notes: [{ midi: 60, time: 1, duration: 1 }] }, {
      startSeconds: 2,
      endSeconds: 3,
    }),
    /no valid notes/,
  );
});
