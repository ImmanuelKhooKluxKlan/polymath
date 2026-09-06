import assert from 'node:assert/strict';
import test from 'node:test';
import {
  analyzePracticeAttempt,
  buildLearningArrangement,
  emptyLearningProgress,
  recommendedLearningLevel,
  recordLearningAttempt,
} from '../../src/engine/learningCoach.js';

const notes = [
  { id: 'bass-c', note: 'C3', time: 0, duration: 1, velocity: 0.5, hand: 'left' },
  { id: 'mid-e', note: 'E4', time: 0, duration: 0.5, velocity: 0.6, hand: 'right' },
  { id: 'top-g', note: 'G4', time: 0, duration: 0.5, velocity: 0.8, hand: 'right', scoreRole: 'melody' },
  { id: 'bass-f', note: 'F3', time: 1, duration: 1, velocity: 0.5, hand: 'left' },
  { id: 'top-a', note: 'A4', time: 1, duration: 0.75, velocity: 0.75, hand: 'right' },
];

test('learning arrangements progressively restore musical density', () => {
  const melody = buildLearningArrangement(notes, 'melody');
  const beginner = buildLearningArrangement(notes, 'beginner');
  const intermediate = buildLearningArrangement(notes, 'intermediate');
  const original = buildLearningArrangement(notes, 'original');
  assert.deepEqual(melody.map((note) => note.note), ['G4', 'A4']);
  assert.equal(beginner.length, 4);
  assert.ok(intermediate.length >= beginner.length);
  assert.equal(original.length, notes.length);
});

test('learning levels respect the arranger melody role instead of blindly taking the highest note', () => {
  const arranged = buildLearningArrangement([
    { id: 'bass', note: 'C3', time: 0, duration: 0.8, velocity: 0.48, arrangementRole: 'bass' },
    { id: 'melody', note: 'E4', time: 0, duration: 0.7, velocity: 0.86, arrangementRole: 'melody' },
    { id: 'ornament', note: 'C6', time: 0, duration: 0.15, velocity: 0.55, arrangementRole: 'harmony' },
  ], 'melody');

  assert.equal(arranged.length, 1);
  assert.equal(arranged[0].id, 'melody');
  assert.equal(arranged[0].learningRole, 'melody');
  assert.equal(arranged[0].hand, 'right');
});

test('rapid duplicate melody evidence becomes one sustained learning note', () => {
  const arranged = buildLearningArrangement([
    { id: 'first', note: 'C5', time: 0, duration: 0.06, velocity: 0.72, arrangementRole: 'melody' },
    { id: 'duplicate', note: 'C5', time: 0.04, duration: 0.32, velocity: 0.81, arrangementRole: 'melody' },
  ], 'melody');

  assert.equal(arranged.length, 1);
  assert.ok(arranged[0].duration >= 0.36);
  assert.equal(arranged[0].velocity, 0.81);
  assert.ok(arranged[0].audioDuration > 0.3);
});

test('a precise performance receives a strong report', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: notes,
    playedNotes: notes.map((note) => ({
      note: note.note,
      start: note.time + 0.015,
      duration: note.duration * 0.98,
      velocity: note.velocity,
      dynamicCapable: true,
    })),
    range: { start: 0, end: 2 },
    levelId: 'original',
  });
  assert.ok(report.score >= 95);
  assert.equal(report.missedCount, 0);
  assert.equal(report.extraCount, 0);
  assert.equal(report.metrics.dynamics.available, true);
});

test('missed and extra notes lower the score and create a useful focus', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: notes,
    playedNotes: [{ note: 'C3', start: 0.2, duration: 0.1 }, { note: 'D5', start: 0.5, duration: 0.1 }],
    range: { start: 0, end: 2 },
  });
  assert.ok(report.score < 50);
  assert.equal(report.missedCount, 4);
  assert.equal(report.extraCount, 1);
  assert.equal(report.focus, 'Notes');
  assert.ok(report.missedNotes.length > 0);
});

test('progress keeps personal bests without discarding recent results', () => {
  const first = recordLearningAttempt(emptyLearningProgress(), 'song-a', {
    levelId: 'beginner', range: { start: 0, end: 10 }, score: 82, createdAt: 'one',
  });
  const second = recordLearningAttempt(first, 'song-a', {
    levelId: 'beginner', range: { start: 0, end: 10 }, score: 65, createdAt: 'two',
  });
  assert.equal(second.totalAttempts, 2);
  assert.equal(second.practiceSeconds, 20);
  assert.equal(second.songs['song-a'].bestScore, 82);
  assert.equal(second.songs['song-a'].lastScore, 65);
  assert.equal(second.songs['song-a'].lastLevelId, 'beginner');
});

test('difficulty recommendations react to recent evidence without changing it silently', () => {
  assert.equal(recommendedLearningLevel(null), 'beginner');
  assert.equal(recommendedLearningLevel({ attempts: 2, lastLevelId: 'beginner', lastScore: 91 }), 'intermediate');
  assert.equal(recommendedLearningLevel({ attempts: 1, lastLevelId: 'beginner', lastScore: 44 }), 'melody');
  assert.equal(recommendedLearningLevel({ attempts: 3, lastLevelId: 'intermediate', lastScore: 76 }), 'intermediate');
});

test('timing feedback identifies whether matched notes tend to be early', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: notes.slice(0, 3),
    playedNotes: notes.slice(0, 3).map((note) => ({ note: note.note, start: note.time - 0.14, duration: note.duration })),
    range: { start: 0, end: 1 },
  });
  assert.equal(report.timingDirection, 'early');
  assert.ok(report.timingBiasMs < -100);
  assert.equal(report.evidence.schema, 'polymath-practice-evidence-v1');
  assert.equal(report.evidence.timing.worst[0].direction, 'early');
  assert.ok(report.evidence.timing.worst[0].errorMs < -100);
});

test('practice evidence preserves exact measured releases and touch for grounded coaching', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: [{ note: 'G4', time: 0, duration: 0.9, velocity: 0.72 }],
    playedNotes: [{ note: 'G4', start: 0.018, duration: 0.33, velocity: 0.41, dynamicCapable: true }],
    range: { start: 0, end: 1 },
  });
  assert.deepEqual(report.evidence.holds.worst[0], {
    note: 'G4',
    targetMs: 900,
    actualMs: 330,
    differenceMs: -570,
    direction: 'short',
  });
  assert.equal(report.evidence.dynamics.targetAveragePercent, 72);
  assert.equal(report.evidence.dynamics.playedAveragePercent, 41);
});

test('pedal changes are measured separately from notes', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: notes,
    playedNotes: notes.map((note) => ({ note: note.note, start: note.time, duration: note.duration })),
    expectedPedals: [{ time: 0.2, down: true }, { time: 1.2, down: false }],
    playedPedals: [{ time: 0.25, down: true }, { time: 1.25, down: false }],
    range: { start: 0, end: 2 },
  });
  assert.equal(report.metrics.pedal.available, true);
  assert.ok(report.metrics.pedal.score >= 90);
  assert.equal(report.evidence.pedal.expectedCount, 2);
  assert.equal(report.evidence.pedal.matchedCount, 2);
  assert.equal(report.evidence.pedal.events[0].errorMs, 50);
});

test('a pedal held into a phrase must be pressed at the phrase start', () => {
  const report = analyzePracticeAttempt({
    expectedNotes: notes,
    playedNotes: notes.map((note) => ({ note: note.note, start: note.time, duration: note.duration })),
    expectedPedals: [{ time: 0.5, down: true }, { time: 2.5, down: false }],
    playedPedals: [{ time: 1.02, down: true }],
    range: { start: 1, end: 2 },
  });
  assert.equal(report.metrics.pedal.available, true);
  assert.equal(report.metrics.pedal.score, 97);
});
