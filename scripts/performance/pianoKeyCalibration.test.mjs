import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PIANO_KEY_CALIBRATION,
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
  -27.626, -27.29, -26.944, -23.221, -26.033, -22.736, -23.625, -25.678,
  -23.535, -18.545, -22.31, -24.967, -25.519, -24.719, -22.563, -18.156,
  -21.728, -20.112, -19.587, -21.779, -21.923, -27.273, -27.256, -27.936,
  -24.939, -24.821, -24.276, -24.437, -24.481, -26.099, -24.362, -21.545,
  -27.405, -27.033, -27.679, -29.095, -31.051, -32.218, -30.182, -31.887,
  -28.776, -28.08, -28.43, -30.52, -27.61, -27.95, -24.808, -27.927,
  -28.662, -33.221, -24.135, -28.693, -29.403, -31.109, -27.36, -23.535,
  -24.862, -26.728, -39.975, -33.791, -31.668, -31.892, -32.172, -35.167,
  -40.702, -35.19, -32.352, -28.719, -34.984, -38.77, -41.128, -34.785,
  -44.462, -38.779, -44.524, -46.976, -44.727, -48.075, -51.92, -48.6,
  -50.145, -41.379, -52.727, -51.488, -50.66, -52.869, -50.49, -52.417,
];

const COMPACT_BODY_REFERENCE_DB = [
  -30.855, -30.589, -30.343, -26.467, -29.359, -25.96, -26.55, -28.205,
  -28.204, -25.102, -30.011, -28.875, -29.585, -30.214, -29.349, -22.854,
  -28.34, -25.346, -27.895, -27.278, -28.785, -36.891, -35.182, -34.245,
  -31.236, -31.658, -31.731, -31.023, -30.472, -35.032, -35.097, -34.354,
  -33.822, -33.808, -36.255, -38.324, -41.419, -42.196, -41.486, -41.865,
  -41.554, -37.518, -39.06, -40.627, -40.404, -40.748, -35.234, -40.438,
  -39.453, -43.869, -42.618, -42.632, -41.15, -43.941, -41.087, -38.4,
  -43.399, -38.261, -52.812, -42.697, -45.172, -47.198, -46.053, -46.638,
  -49.339, -48.958, -43.165, -47.256, -46.197, -53.732, -52.546, -49.735,
  -58.65, -63.652, -63.025, -58.777, -59.479, -69.011, -73.357, -67.975,
  -71.88, -65.712, -77.444, -72.441, -76.863, -78.365, -78.826, -80.002,
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
  assert.equal(PIANO_KEY_CALIBRATION.keyCount, 88);
  for (const values of [
    PIANO_KEY_CALIBRATION.fullRangeAttackDb,
    PIANO_KEY_CALIBRATION.fullRangeBodyDb,
    PIANO_KEY_CALIBRATION.compactAttackDb,
    PIANO_KEY_CALIBRATION.compactBodyDb,
  ]) assert.equal(values.length, 88);
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
  assert.ok(maximumAdjacentJump(attack) <= 1.3);
  assert.ok(maximumAdjacentJump(body) <= 1.91);
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
