import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FALLING_NOTE_STRIKE_PERCENT,
  fallingNoteGeometry,
  synchronizedUntilPress,
} from '../../src/engine/fallingNoteGeometry.js';

test('a falling note touches the piano boundary exactly at its strike time', () => {
  const geometry = fallingNoteGeometry({
    eventTime: 12.5,
    currentTime: 12.5,
    duration: 0.6,
    leadTime: 3.4,
  });

  assert.equal(geometry.bottom, FALLING_NOTE_STRIKE_PERCENT);
  assert.equal(geometry.top + geometry.height, FALLING_NOTE_STRIKE_PERCENT);
  assert.equal(geometry.touching, true);
});

test('only the scheduled strike snaps across a short animation-frame delay', () => {
  assert.equal(synchronizedUntilPress(0.04, true), 0);
  assert.equal(synchronizedUntilPress(0.04, false), 0.04);
  assert.equal(synchronizedUntilPress(0.2, true), 0.2);
  assert.equal(synchronizedUntilPress(-0.01, true), -0.01);
});

test('compact rows keep the same strike boundary', () => {
  const regular = fallingNoteGeometry({
    eventTime: 4,
    currentTime: 4,
    duration: 1,
    leadTime: 3,
  });
  const compact = fallingNoteGeometry({
    eventTime: 4,
    currentTime: 4,
    duration: 1,
    leadTime: 3,
    compact: true,
  });

  assert.equal(regular.bottom, 100);
  assert.equal(compact.bottom, 100);
  assert.ok(compact.height < regular.height);
});
