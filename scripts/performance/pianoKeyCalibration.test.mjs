import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PIANO_KEY_CALIBRATION,
  pianoCompactSourceChannel,
  pianoKeyCalibrationDb,
  pianoKeyCalibrationGain,
} from '../../src/engine/pianoKeyCalibration.js';

const FULL_ATTACK_REFERENCE_DB = [
  -31.992, -31.707, -31.405, -26.831, -28.592, -26.207, -26.442, -29.737,
  -26.547, -20.771, -23.76, -25.989, -25.99, -27.486, -25.0, -20.662,
  -21.507, -22.904, -20.76, -22.365, -23.494, -27.888, -26.734, -27.792,
  -25.66, -24.9, -24.99, -23.929, -23.361, -25.558, -23.744, -22.294,
  -27.867, -24.688, -25.144, -23.279, -27.525, -25.429, -25.632, -27.712,
  -26.148, -22.391, -24.218, -27.567, -26.34, -24.344, -22.536, -22.562,
  -24.075, -21.904, -21.344, -21.537, -25.971, -26.657, -24.212, -20.221,
  -22.094, -20.457, -23.306, -28.218, -25.361, -21.796, -26.374, -27.586,
  -32.78, -25.857, -26.287, -22.374, -24.132, -27.335, -30.946, -25.753,
  -29.264, -25.961, -33.219, -35.653, -32.539, -36.419, -36.918, -34.759,
  -37.282, -26.45, -40.196, -38.955, -38.391, -42.507, -38.17, -41.794,
];

const FULL_BODY_REFERENCE_DB = [
  -34.884, -34.693, -34.508, -29.894, -31.981, -29.228, -28.849, -32.466,
  -31.862, -28.482, -31.245, -30.51, -29.9, -31.644, -31.219, -25.521,
  -28.141, -28.596, -28.279, -29.195, -29.645, -36.637, -35.029, -35.323,
  -32.986, -31.915, -32.868, -32.123, -30.675, -34.914, -32.177, -33.348,
  -34.735, -33.32, -34.217, -33.332, -39.711, -38.894, -36.073, -37.671,
  -37.637, -32.086, -34.239, -37.494, -36.274, -34.166, -32.832, -35.233,
  -35.328, -34.261, -38.294, -35.413, -37.638, -40.831, -38.017, -35.416,
  -38.917, -31.811, -39.132, -37.089, -38.138, -37.853, -39.482, -40.872,
  -43.528, -38.141, -36.163, -40.429, -35.872, -42.725, -41.21, -40.168,
  -42.52, -50.697, -46.972, -47.845, -45.851, -56.745, -52.795, -52.657,
  -53.085, -46.526, -60.725, -59.8, -63.298, -67.743, -67.248, -69.053,
];

const COMPACT_ATTACK_REFERENCE_DB = [
  -32.053, -31.548, -31.146, -24.736, -26.074, -25.757, -25.764, -29.648,
  -26.014, -21.162, -23.588, -24.095, -25.234, -25.553, -23.219, -18.473,
  -18.952, -20.842, -17.717, -19.973, -22.43, -26.411, -24.482, -24.885,
  -23.011, -22.597, -22.98, -21.767, -22.278, -23.827, -21.903, -20.092,
  -27.067, -23.861, -22.491, -22.135, -27.181, -23.698, -24.498, -26.372,
  -26.475, -22.777, -26.003, -27.996, -26.082, -25.078, -24.734, -23.399,
  -24.513, -24.615, -22.126, -21.705, -26.296, -29.335, -26.73, -21.625,
  -24.754, -24.288, -26.559, -29.965, -27.61, -25.321, -31.154, -30.731,
  -36.403, -29.648, -30.973, -28.862, -29.607, -33.03, -38.267, -33.158,
  -39.794, -36.561, -42.783, -46.31, -43.937, -48.941, -47.42, -46.048,
  -48.386, -38.469, -50.034, -49.499, -48.775, -50.053, -49.151, -50.28,
];

const COMPACT_BODY_REFERENCE_DB = [
  -34.945, -34.611, -34.248, -27.786, -29.405, -28.329, -28.027, -31.326,
  -30.719, -28.321, -31.142, -27.932, -28.714, -30.677, -30.224, -22.782,
  -25.323, -25.438, -25.635, -26.348, -28.396, -34.701, -32.225, -31.305,
  -29.924, -29.02, -29.665, -29.43, -29.018, -34.078, -32.313, -31.665,
  -33.984, -32.49, -31.327, -32.671, -38.889, -34.952, -36.503, -37.108,
  -38.091, -32.513, -35.435, -38.017, -36.82, -36.517, -34.96, -35.616,
  -36.617, -36.704, -40.546, -35.783, -38.633, -43.798, -39.688, -38.006,
  -42.011, -35.238, -42.429, -38.885, -40.404, -42.491, -43.514, -45.733,
  -47.526, -42.117, -40.364, -49.699, -41.188, -48.42, -47.495, -47.98,
  -51.347, -60.675, -56.564, -57.809, -59.09, -67.021, -64.024, -63.354,
  -69.141, -62.392, -74.471, -73.706, -74.964, -77.897, -77.067, -79.842,
];

function calibratedLevels(reference, profile, phase) {
  return reference.map((level, index) => (
    level + pianoKeyCalibrationDb(21 + index, profile, phase)
  ));
}

function maximumAdjacentJump(values) {
  return Math.max(...values.slice(1).map((value, index) => Math.abs(value - values[index])));
}

test('the envelope calibration covers every key on an 88-key piano', () => {
  assert.equal(PIANO_KEY_CALIBRATION.version, 'iowa-mf-88-compact-harmonic-envelope-v4');
  assert.equal(PIANO_KEY_CALIBRATION.keyCount, 88);
  for (const values of [
    PIANO_KEY_CALIBRATION.fullRangeAttackDb,
    PIANO_KEY_CALIBRATION.fullRangeBodyDb,
    PIANO_KEY_CALIBRATION.compactAttackDb,
    PIANO_KEY_CALIBRATION.compactBodyDb,
  ]) assert.equal(values.length, 88);
  assert.equal(PIANO_KEY_CALIBRATION.compactSourceChannel.length, 88);
  assert.ok(PIANO_KEY_CALIBRATION.compactSourceChannel.every((channel) => channel === 0 || channel === 1));
  assert.equal(pianoCompactSourceChannel(79), 0, 'G5 uses the phase-safe left microphone');
  assert.equal(pianoCompactSourceChannel(86), 1, 'D6 uses the phase-safe right microphone');
  assert.equal(pianoKeyCalibrationDb(20), pianoKeyCalibrationDb(21));
  assert.equal(pianoKeyCalibrationDb(109), pianoKeyCalibrationDb(108));
  assert.ok(Number.isFinite(pianoKeyCalibrationGain(60, 'full-range', 'body')));
});

test('full-range attack and body remain continuous between neighboring keys', () => {
  const attack = calibratedLevels(FULL_ATTACK_REFERENCE_DB, 'full-range', 'attack');
  const body = calibratedLevels(FULL_BODY_REFERENCE_DB, 'full-range', 'body');
  assert.ok(maximumAdjacentJump(attack) <= 1.05);
  assert.ok(maximumAdjacentJump(body) <= 1.5);
  assert.ok(attack[87] < attack[39] - 10, 'the natural high-register contour is preserved');
});

test('compact attack and body remain continuous between neighboring keys', () => {
  const attack = calibratedLevels(COMPACT_ATTACK_REFERENCE_DB, 'small-speaker', 'attack');
  const body = calibratedLevels(COMPACT_BODY_REFERENCE_DB, 'small-speaker', 'body');
  assert.ok(maximumAdjacentJump(attack) <= 1.51);
  assert.ok(maximumAdjacentJump(body) <= 1.51);
  assert.ok(body[87] < body[39] - 20, 'the compact curve remains acoustic, not flat');
});

test('all attack and body corrections stay inside measured safe headroom', () => {
  const allDb = [
    ...PIANO_KEY_CALIBRATION.fullRangeAttackDb,
    ...PIANO_KEY_CALIBRATION.fullRangeBodyDb,
    ...PIANO_KEY_CALIBRATION.compactAttackDb,
    ...PIANO_KEY_CALIBRATION.compactBodyDb,
  ];
  const gains = allDb.map((db) => 10 ** (db / 20));
  assert.ok(Math.min(...gains) >= 0.29);
  assert.ok(Math.max(...gains) <= 2.91);
});
