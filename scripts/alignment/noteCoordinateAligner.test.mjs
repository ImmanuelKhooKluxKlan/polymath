import assert from 'node:assert/strict';
import test from 'node:test';
import {
  alignNoteCoordinates,
  createAlignmentSvg,
  mapReferenceTime,
} from './noteCoordinateAligner.mjs';

function desiredPerformance() {
  const pattern = [60, 64, 67, 62, 65, 69, 71, 67, 64, 60, 65, 72];
  return Array.from({ length: 72 }, (_, index) => ({
    midi: pattern[index % pattern.length] + (Math.floor(index / pattern.length) % 2) * 12,
    time: index * 1.37 + (index % 5) * 0.035,
    duration: 0.42 + (index % 4) * 0.08,
    velocity: 0.7 + (index % 3) * 0.06,
    instrument: 'piano',
  }));
}

function sourceTime(time) {
  // Seven-second capture delay, roughly 2% slower, and local tempo movement.
  return 7 + time * 1.02 + Math.sin(time / 14) * 0.16;
}

test('aligner ignores opening noise and tolerates octave mistakes, duplicates, and missing notes', () => {
  const reference = desiredPerformance();
  const observed = reference
    .filter((_, index) => ![13, 41].includes(index))
    .map((note, index) => ({
      ...note,
      midi: index % 17 === 0 ? note.midi + 12 : note.midi,
      time: sourceTime(note.time) + ((index % 7) - 3) * 0.006,
      instrument: 'acoustic_piano',
    }));
  observed.push(
    { midi: 91, time: 0.2, duration: 0.02, instrument: 'noise' },
    { midi: 48, time: 1.1, duration: 0.03, instrument: 'noise' },
    { midi: 60, time: 2.3, duration: 0.04, instrument: 'noise' },
    { midi: 60, time: sourceTime(reference[20].time) + 0.02, duration: 0.02, instrument: 'duplicate' },
  );

  const result = alignNoteCoordinates(reference, observed, {
    ransacIterations: 12000,
    matchWindowSeconds: 1.2,
  });

  assert.ok(result.metrics.matchedReferencePercent > 85, JSON.stringify(result.metrics));
  assert.ok(Math.abs(result.metrics.coarseOffsetSeconds - 7) < 0.4, JSON.stringify(result.metrics));
  assert.ok(Math.abs(result.metrics.coarseScale - 1.02) < 0.02, JSON.stringify(result.metrics));
  assert.ok(result.metrics.octaveEquivalentMatches >= 2, JSON.stringify(result.metrics));
  assert.ok(Math.abs(mapReferenceTime(70, result.anchors) - sourceTime(70)) < 0.25);
  const svg = createAlignmentSvg(result);
  assert.match(svg, /Polymath note-coordinate alignment/);
  assert.match(svg, /nonlinear time map/);
});

test('aligner refuses files without enough musical evidence', () => {
  assert.throws(
    () => alignNoteCoordinates([{ midi: 60, time: 0 }], [{ midi: 60, time: 7 }]),
    /At least eight usable notes/,
  );
});
