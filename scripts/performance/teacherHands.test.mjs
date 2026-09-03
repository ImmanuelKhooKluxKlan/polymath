import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildTeacherHandTargets,
  fingerForMidi,
  pianoPercentForMidi,
  prepareTeacherHandTimeline,
  teacherReply,
  teacherHandForEvent,
  TEACHER_PROFILES,
} from '../../src/engine/teacherHands.js';

test('explicit metadata wins and pitch provides a safe fallback', () => {
  assert.equal(teacherHandForEvent({ note: 'C5', hand: 'left' }), 'left');
  assert.equal(teacherHandForEvent({ note: 'C2' }), 'left');
  assert.equal(teacherHandForEvent({ note: 'C5' }), 'right');
});

test('left and right fingering runs in opposite directions', () => {
  const notes = [{ midi: 48 }, { midi: 52 }, { midi: 55 }];
  assert.equal(fingerForMidi(48, 'left', notes), 5);
  assert.equal(fingerForMidi(55, 'left', notes), 3);
  assert.equal(fingerForMidi(48, 'right', notes), 1);
  assert.equal(fingerForMidi(55, 'right', notes), 3);
});

test('teacher targets follow held notes and preserve chord fingers', () => {
  const notes = [
    { id: 'bass', note: 'C3', time: 1, duration: 1, hand: 'left' },
    { id: 'melody-a', note: 'E4', time: 1, duration: 0.5, hand: 'right' },
    { id: 'melody-b', note: 'G4', time: 1, duration: 0.5, hand: 'right' },
  ];
  const targets = buildTeacherHandTargets(notes, 1.2);
  assert.equal(targets.hasTargets, true);
  assert.deepEqual(targets.left.notes.map((note) => note.note), ['C3']);
  assert.deepEqual(targets.right.notes.map((note) => note.note), ['E4', 'G4']);
  assert.equal(targets.left.isPressing, true);
  assert.equal(targets.right.notes[0].finger, 1);
  assert.equal(targets.right.notes[1].finger, 2);
});

test('single-hand practice hides the other hand targets', () => {
  const notes = [
    { note: 'C3', time: 2, duration: 0.4, hand: 'left' },
    { note: 'C5', time: 2, duration: 0.4, hand: 'right' },
  ];
  const targets = buildTeacherHandTargets(notes, 1.8, { handMode: 'right' });
  assert.equal(targets.left.notes.length, 0);
  assert.equal(targets.right.notes.length, 1);
});

test('piano positions remain inside the teacher keyboard', () => {
  assert.equal(pianoPercentForMidi(21), 2);
  assert.equal(pianoPercentForMidi(108), 98);
});

test('the prepared timeline sorts once and can be reused during playback', () => {
  const timeline = prepareTeacherHandTimeline([
    { note: 'C5', time: 4, duration: 0.2 },
    { note: 'C3', time: 1, duration: 2 },
  ]);
  assert.deepEqual(timeline.entries.map((event) => event.note), ['C3', 'C5']);
  const targets = buildTeacherHandTargets(timeline, 2.5);
  assert.equal(targets.left.notes[0].note, 'C3');
});

test('the four human teachers have deployable image assets', () => {
  assert.deepEqual(TEACHER_PROFILES.map((teacher) => teacher.id), ['padme', 'anakin', 'taylor', 'mace']);
  for (const teacher of TEACHER_PROFILES) {
    assert.match(teacher.image, /^\/teachers\/.+\.webp$/);
    assert.match(teacher.armImage, /^\/teachers\/arm-.+\.webp$/);
  }
});

test('Mace stays stern and does not give empty praise', () => {
  const mace = TEACHER_PROFILES.find((teacher) => teacher.id === 'mace');
  const reply = teacherReply(mace, 'Can you help my hands?', buildTeacherHandTargets([], 0));
  assert.match(reply, /Again/);
  assert.doesNotMatch(reply, /great|perfect|amazing/i);
});
