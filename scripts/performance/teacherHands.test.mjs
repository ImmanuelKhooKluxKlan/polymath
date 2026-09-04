import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildTeacherHandTargets,
  fingerForMidi,
  pianoPercentForMidi,
  prepareTeacherHandTimeline,
  teacherReply,
  teacherHandForEvent,
  teacherRowHandPlacement,
  TEACHER_PROFILES,
} from '../../src/engine/teacherHands.js';
import { buildAdaptivePianoLayout, buildLearningHandLayout } from '../../src/engine/grandPianoLayout.js';
import {
  clampTeacherDepth,
  clampTeacherOffset,
  teacherPoseById,
  TEACHER_POSES,
} from '../../src/engine/teacherAvatarControls.js';
import {
  canonicalTeacherBone,
  draggedJointRotation,
  teacherRigPose,
} from '../../src/engine/teacherRig.js';
import {
  TEACHER_KEYBOARD_GEOMETRY,
  separateTeacherHandCentres,
  teacherHandCenter,
  teacherKeyGeometry,
} from '../../src/engine/teacherPerformanceRig.js';

test('explicit metadata wins and pitch provides a safe fallback', () => {
  assert.equal(teacherHandForEvent({ note: 'C5', hand: 'left' }), 'left');
  assert.equal(teacherHandForEvent({ note: 'C2' }), 'left');
  assert.equal(teacherHandForEvent({ note: 'C5' }), 'right');
});

test('left and right fingering runs in opposite directions', () => {
  const notes = [{ midi: 48 }, { midi: 52 }, { midi: 55 }];
  assert.equal(fingerForMidi(48, 'left', notes), 5);
  assert.equal(fingerForMidi(55, 'left', notes), 1);
  assert.equal(fingerForMidi(48, 'right', notes), 1);
  assert.equal(fingerForMidi(55, 'right', notes), 5);
  assert.equal(fingerForMidi(60, 'right', [{ midi: 60 }]), 3);
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
  assert.equal(targets.right.notes[1].finger, 5);
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

test('photographic hands map chords onto the same adaptive piano row', () => {
  const row = buildAdaptivePianoLayout([{ note: 'C4' }, { note: 'G4' }]).rows[0];
  const placement = teacherRowHandPlacement({
    notes: [{ midi: 60 }, { midi: 67 }],
  }, row, 'right');
  const c = row.getPosition(60).centerPercent;
  const g = row.getPosition(67).centerPercent;
  assert.ok(placement.centerPercent > c && placement.centerPercent < g);
  assert.ok(placement.widthPercent > 21);
  assert.ok(placement.widthPercent <= 33);
  assert.equal(placement.horizontalScale, 0.9);
  assert.equal(placement.fingerDepthPercent, 25);
  assert.ok(placement.wristTiltDegrees >= -4.5 && placement.wristTiltDegrees <= 4.5);
  assert.ok(placement.verticalFlex >= 0.965 && placement.verticalFlex <= 1);
  assert.ok(placement.approachLiftPixels >= 4.75 && placement.approachLiftPixels <= 7);
  assert.equal(placement.pressDepthPixels, 4);

  const blackPlacement = teacherRowHandPlacement({ notes: [{ midi: 61 }] }, row, 'right');
  assert.equal(blackPlacement.fingerDepthPercent, 13);
  assert.equal(blackPlacement.pressDepthPixels, 2.25);
  assert.equal(teacherRowHandPlacement({ notes: [{ midi: 10 }] }, row, 'right'), null);
  assert.equal(teacherRowHandPlacement({ notes: [] }, row, 'left', true).centerPercent, 36);
});

test('two-hand learning uses one physical 88-key piano instead of duplicate storeys', () => {
  const layout = buildLearningHandLayout([{ note: 'A0' }, { note: 'C8' }], 'both');
  assert.equal(layout.mode, 'learn-grand-single');
  assert.equal(layout.rows.length, 1);
  assert.equal(layout.rows[0].keys.length, 88);
  assert.equal(layout.rows[0].whiteKeys.length, 52);
  assert.equal(layout.getPosition(21).rowId, 'main');
  assert.equal(layout.getPosition(108).rowId, 'main');
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

test('the human teachers have deployable image assets', () => {
  assert.deepEqual(TEACHER_PROFILES.map((teacher) => teacher.id), ['aria', 'nova', 'anakin', 'taylor', 'mace']);
  for (const teacher of TEACHER_PROFILES) {
    assert.match(teacher.image, /^\/teachers\/.+\.(webp|jpg)$/);
    assert.match(teacher.stageImage, /^\/teachers\/.+\.jpg$/);
    assert.match(teacher.handCameraImage, /^\/teachers\/.+\.webp$/);
    assert.match(teacher.pressedHandCameraImage, /^\/teachers\/.+\.webp$/);
  }
  const padme = TEACHER_PROFILES.find((teacher) => teacher.id === 'nova');
  const anakin = TEACHER_PROFILES.find((teacher) => teacher.id === 'anakin');
  assert.equal(padme.name, 'Padme');
  assert.equal(padme.stageImage, '/teachers/padme-teacher-studio-v1.jpg');
  assert.equal(padme.requiresAdultConfirmation, true);
  assert.equal(anakin.stageImage, '/teachers/anakin-teacher-studio-v2.jpg');
});

test('the demonstration piano contains a physical 88-key layout', () => {
  assert.equal(TEACHER_KEYBOARD_GEOMETRY.length, 88);
  assert.equal(TEACHER_KEYBOARD_GEOMETRY.filter((key) => !key.black).length, 52);
  assert.equal(TEACHER_KEYBOARD_GEOMETRY.filter((key) => key.black).length, 36);
  assert.equal(teacherKeyGeometry(21).black, false);
  assert.equal(teacherKeyGeometry(22).black, true);
  assert.equal(teacherKeyGeometry(108).black, false);
});

test('the overhead hands track the playable keyboard without leaving its range', () => {
  const left = teacherHandCenter({ notes: [{ midi: 21 }, { midi: 24 }, { midi: 28 }] }, 'left');
  const right = teacherHandCenter({ notes: [{ midi: 96 }, { midi: 100 }, { midi: 108 }] }, 'right');
  assert.ok(left >= 66 && left <= 1134);
  assert.ok(right >= 66 && right <= 1134);
  assert.ok(left < right);
  assert.deepEqual(separateTeacherHandCentres(500, 620), { left: 440, right: 680 });
});

test('Mace stays stern and does not give empty praise', () => {
  const mace = TEACHER_PROFILES.find((teacher) => teacher.id === 'mace');
  const reply = teacherReply(mace, 'Can you help my hands?', buildTeacherHandTargets([], 0));
  assert.match(reply, /Again/);
  assert.doesNotMatch(reply, /great|perfect|amazing/i);
});

test('teacher pose controls remain bounded and always have a safe fallback', () => {
  assert.equal(TEACHER_POSES.length, 6);
  assert.equal(teacherPoseById('rest').rotation, 88);
  assert.equal(teacherPoseById('not-real').id, 'ready');
  assert.deepEqual(clampTeacherOffset({ x: 900, y: -900 }, { maximumX: 120, maximumY: 80 }), { x: 120, y: -80 });
  assert.equal(clampTeacherDepth(9), 1.45);
  assert.equal(clampTeacherDepth(-2), 0.7);
});

test('3D teacher rigs recognise common human skeleton names', () => {
  assert.equal(canonicalTeacherBone('mixamorigLeftArm'), 'upperArmL');
  assert.equal(canonicalTeacherBone('mixamorigRightForeArm'), 'lowerArmR');
  assert.equal(canonicalTeacherBone('LeftUpLeg'), 'upperLegL');
  assert.equal(canonicalTeacherBone('RightFoot'), 'footR');
  assert.equal(canonicalTeacherBone('Hips'), 'hips');
});

test('3D teacher joint dragging and pose fallback stay anatomically bounded', () => {
  const rotation = draggedJointRotation('head', { x: 0, y: 0, z: 0 }, 20, -20);
  assert.equal(rotation.x, -0.75);
  assert.equal(rotation.y, 1.05);
  assert.equal(teacherRigPose('kneel').joints.lowerLegL.x, 2.08);
  assert.equal(teacherRigPose('unknown'), teacherRigPose('ready'));
});
