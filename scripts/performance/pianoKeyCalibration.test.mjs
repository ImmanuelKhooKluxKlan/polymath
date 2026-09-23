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
  -24.407, -24.028, -23.751, -18.871, -20.501, -19.058, -19.142, -23.71,
  -20.264, -15.391, -19.552, -20.033, -21.59, -22.067, -19.977, -14.949,
  -15.68, -17.994, -15.012, -16.076, -19.421, -23.706, -22.298, -22.537,
  -20.974, -20.505, -20.833, -20.06, -20.611, -22.348, -19.98, -18.369,
  -25.835, -22.357, -21.651, -21.198, -26.172, -22.835, -23.572, -25.261,
  -25.402, -21.682, -24.862, -26.81, -24.896, -23.875, -23.483, -22.132,
  -23.209, -23.276, -20.759, -20.305, -24.878, -27.897, -25.276, -20.152,
  -23.259, -22.795, -25.044, -28.423, -26.034, -23.705, -29.492, -29.016,
  -34.635, -27.817, -29.069, -26.88, -27.534, -30.859, -35.996, -30.755,
  -37.287, -33.901, -40.024, -43.416, -40.977, -45.956, -44.559, -43.021,
  -47.437, -37.397, -47.528, -46.902, -46.247, -47.725, -46.693, -47.979,
];

const COMPACT_BODY_REFERENCE_DB = [
  -27.487, -27.292, -27.067, -22.181, -23.698, -21.735, -21.484, -25.462,
  -24.339, -22.868, -27.021, -23.975, -25.156, -27.353, -27.337, -19.473,
  -22.361, -22.82, -23.334, -22.497, -25.49, -32.207, -29.987, -28.561,
  -27.781, -26.875, -27.499, -27.545, -27.306, -32.515, -31.047, -30.361,
  -32.25, -30.88, -30.503, -31.733, -37.969, -34.19, -35.561, -36.025,
  -37.014, -31.41, -34.279, -36.829, -35.654, -35.27, -33.685, -34.347,
  -35.327, -35.366, -39.154, -34.379, -37.207, -42.352, -38.232, -36.528,
  -40.489, -33.743, -40.912, -37.343, -38.823, -40.873, -41.853, -44.017,
  -45.758, -40.286, -38.461, -47.716, -39.114, -46.248, -45.212, -45.573,
  -48.817, -58.01, -53.759, -54.861, -56.057, -63.959, -60.853, -60.198,
  -62.608, -55.975, -71.541, -70.859, -72.229, -75.297, -74.536, -77.372,
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
  assert.equal(PIANO_KEY_CALIBRATION.version, 'iowa-mf-88-phase-safe-envelope-v3');
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
  assert.ok(maximumAdjacentJump(attack) <= 1.42);
  assert.ok(maximumAdjacentJump(body) <= 1.86);
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
  assert.ok(Math.max(...gains) <= 2.8);
});
