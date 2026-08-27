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

test('local alignment follows tempo drift and a recording pause while keeping labels inside video time', () => {
  const reference = desiredPerformance();
  const warpedTime = (time) => {
    const opening = 5.2 + time * 0.985;
    const pause = time >= 38 ? 2.4 : 0;
    const lateDrift = time >= 70 ? (time - 70) * 0.035 : 0;
    return opening + pause + lateDrift;
  };
  const observed = reference
    .filter((_, index) => index % 19 !== 0)
    .map((note, index) => ({
      ...note,
      midi: index % 23 === 0 ? note.midi + 12 : note.midi,
      time: warpedTime(note.time) + ((index % 5) - 2) * 0.008,
    }));
  observed.push(
    { midi: 35, time: 0.2, duration: 0.04, instrument: 'noise' },
    { midi: 80, time: 3.4, duration: 0.03, instrument: 'noise' },
  );
  const sourceDurationSeconds = warpedTime(reference.at(-1).time + 2) + 1;
  const result = alignNoteCoordinates(reference, observed, { sourceDurationSeconds });

  assert.ok(Math.abs(mapReferenceTime(20, result.anchors) - warpedTime(20)) < 0.35);
  assert.ok(Math.abs(mapReferenceTime(55, result.anchors) - warpedTime(55)) < 0.45);
  assert.ok(Math.abs(mapReferenceTime(90, result.anchors) - warpedTime(90)) < 0.55);
  assert.ok(result.tempoSegments.some((segment) => segment.flags.includes('pause-cut-or-strong-tempo-change')));
  assert.ok(result.qualityWindows.length >= 15);
  assert.ok(result.alignedReference.every((note) => note.time >= 0 && note.time + note.duration <= sourceDurationSeconds + 0.011));
  assert.equal(result.supervisionPackage.schema, 'polymath-supervision-package-v1');
});

test('manual anchors and review decisions create auditable training eligibility', () => {
  const reference = desiredPerformance();
  const observed = reference.map((note) => ({ ...note, time: 4 + note.time * 1.01 }));
  const result = alignNoteCoordinates(reference, observed, {
    sourceDurationSeconds: 110,
    manualAnchors: [
      { referenceTime: 10, observedTime: 14.1 },
      { referenceTime: 50, observedTime: 54.5 },
      { referenceTime: 90, observedTime: 94.9 },
    ],
    reviewDecisions: { w0001: 'reject', w0002: 'accept' },
  });

  assert.equal(result.metrics.manualAnchorCount, 3);
  assert.equal(result.qualityWindows[0].status, 'rejected');
  assert.equal(result.qualityWindows[0].trainingEligible, false);
  assert.equal(result.qualityWindows[1].status, 'accepted-manually');
  assert.equal(result.qualityWindows[1].trainingEligible, true);
  assert.ok(result.alignedReference.some((note) => note.qualityWindowId === 'w0001' && !note.trainingEligible));
  assert.ok(result.alignedReference.some((note) => note.qualityWindowId === 'w0002' && note.trainingEligible));
  assert.deepEqual(result.supervisionPackage.alignment.manualAnchors.map((anchor) => anchor.kind), ['manual', 'manual', 'manual']);
});
