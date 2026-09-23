const GRAND_START_MIDI = 21;
const GRAND_END_MIDI = 108;
const SMALL_SPEAKER_PROFILE = 'small-speaker';

/**
 * Iowa Steinway mf, 88-key, per-recording calibration.
 *
 * These are decibel corrections, not a blanket equal-volume curve. Every WAV
 * was measured in two windows: hammer/attack (20-250 ms) and resonant body
 * (250-1200 ms), using A-weighted energy. The compact pass additionally used
 * a phase-safe microphone channel and a conservative phone-speaker response.
 * A 13-key rolling median followed by five binomial smoothing passes preserves
 * the piano's broad acoustic contour while removing local recording jumps.
 *
 * Source-set SHA-256:
 * 8886e848eee73976b5a6693e159d55df7a4438f0732eacc7c1dae60cdcd56ed0
 */
const FULL_RANGE_ATTACK_DB = Object.freeze([
  3.473, 3.349, 3.324, -0.895, 1.24, -0.808, -0.297, 3.223,
  0.247, -5.281, -1.973, 0.648, 1.075, 2.952, 0.721, -3.532,
  -2.75, -1.49, -3.758, -2.212, -1.083, 3.327, 2.158, 3.153,
  0.927, 0.075, 0.098, -1.0, -1.585, 0.602, -1.234, -2.726,
  2.784, -0.471, -0.091, -2.022, 2.172, 0.04, 0.232, 2.34,
  0.858, -2.751, -0.72, 2.861, 1.859, 0.057, -1.583, -1.384,
  0.339, -1.58, -1.878, -1.467, 3.09, 3.777, 1.194, -3.07,
  -1.584, -3.669, -1.261, 3.278, 0.132, -3.656, 0.729, 1.757,
  6.771, -0.326, -0.077, -4.207, -2.746, 0.036, 3.06, -2.902,
  -0.32, -4.644, 1.575, 3.006, -1.053, 1.944, 1.631, -1.25,
  0.655, -10.691, 2.636, 1.055, 0.215, 4.114, -0.38, 3.153,
]);

const FULL_RANGE_BODY_DB = Object.freeze([
  2.865, 2.705, 2.582, -1.935, 0.282, -2.309, -2.503, 1.322,
  0.946, -2.192, 0.809, 0.289, -0.148, 1.713, 1.337, -4.387,
  -1.88, -1.628, -2.233, -1.66, -1.569, 5.08, 3.161, 3.174,
  0.573, -0.745, -0.011, -0.937, -2.526, 1.595, -1.268, -0.259,
  0.921, -0.737, -0.101, -1.253, 4.846, 3.74, 0.642, 2.018,
  1.857, -3.715, -1.498, 1.866, 0.755, -1.272, -2.563, -0.157,
  -0.094, -1.242, 2.633, -0.507, 1.358, 4.128, 0.89, -2.073,
  1.16, -6.121, 1.092, -1.024, -0.049, -0.446, 1.001, 2.118,
  4.413, -1.402, -3.848, -0.081, -5.173, 1.081, -1.131, -2.998,
  -1.608, 5.487, 0.588, 0.213, -3.091, 6.441, 1.086, -0.496,
  -1.542, -9.578, 3.201, 0.985, 3.372, 6.911, 5.752, 7.167,
]);

const COMPACT_ATTACK_DB = Object.freeze([
  3.526, 3.211, 3.044, -1.699, 0.07, -1.255, -1.075, 3.577,
  0.219, -4.555, -0.287, 0.298, 1.94, 2.469, 0.388, -4.676,
  -4.017, -1.798, -4.882, -3.924, -0.69, 3.474, 1.94, 2.057,
  0.378, -0.193, 0.041, -0.822, -0.367, 1.249, -1.283, -3.111,
  4.09, 0.31, -0.725, -1.53, 3.088, -0.577, -0.092, 1.448,
  1.542, -2.149, 1.097, 3.118, 1.267, 0.29, -0.075, -1.404,
  -0.296, -0.187, -2.66, -3.095, 1.438, 4.33, 1.476, -3.979,
  -1.274, -2.18, -0.386, 2.527, -0.341, -3.162, 2.125, 1.149,
  6.266, -1.076, -0.41, -3.302, -3.51, -1.226, 2.707, -3.865,
  1.267, -3.529, 1.23, 3.334, -0.305, 3.557, 1.133, -1.319,
  2.322, -8.328, 1.353, 0.415, -0.446, 0.902, -0.207, 1.042,
]);

const COMPACT_BODY_DB = Object.freeze([
  3.37, 3.147, 2.886, -2.033, -0.536, -2.508, -2.755, 1.248,
  0.168, -1.251, 2.951, -0.072, 1.088, 3.2, 3.015, -5.114,
  -2.577, -2.535, -2.463, -3.727, -1.115, 5.27, 2.75, 1.015,
  -0.121, -1.453, -1.31, -1.761, -2.47, 2.32, 0.476, -0.565,
  0.962, -0.793, -1.593, -0.835, 4.88, 0.558, 1.41, 1.437,
  2.102, -3.711, -0.959, 1.532, 0.326, -0.081, -1.695, -1.083,
  -0.188, -0.289, 3.284, -1.798, 0.628, 5.294, 0.644, -1.612,
  1.801, -5.46, 1.26, -2.671, -1.48, 0.309, 0.98, 2.711,
  3.849, -2.388, -5.078, 3.277, -6.234, -0.072, -2.243, -3.254,
  -1.603, 5.884, -0.052, -0.531, -0.819, 5.628, 1.014, -1.262,
  -0.609, -9.094, 4.64, 2.307, 2.348, 4.467, 3.123, 5.669,
]);

// 0 = left microphone, 1 = right microphone. The choice is based on the
// combined phone-band attack/body energy for that exact Iowa recording. A
// single real microphone is copied to both phone speakers so phase cancellation
// cannot make an otherwise loud piano key disappear.
const COMPACT_SOURCE_CHANNEL = Object.freeze([
  1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
  1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
  0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
  0, 0, 1, 1, 1, 1, 1, 1,
]);

function keyIndex(midi) {
  const key = Math.round(Number(midi));
  if (!Number.isFinite(key)) return 60 - GRAND_START_MIDI;
  return Math.max(0, Math.min(GRAND_END_MIDI - GRAND_START_MIDI, key - GRAND_START_MIDI));
}

function correctionsFor(profile, phase) {
  if (profile === SMALL_SPEAKER_PROFILE) {
    return phase === 'body' ? COMPACT_BODY_DB : COMPACT_ATTACK_DB;
  }
  return phase === 'body' ? FULL_RANGE_BODY_DB : FULL_RANGE_ATTACK_DB;
}

export function pianoKeyCalibrationDb(midi, profile = 'full-range', phase = 'attack') {
  return correctionsFor(profile, phase)[keyIndex(midi)];
}

export function pianoKeyCalibrationGain(midi, profile = 'full-range', phase = 'attack') {
  return 10 ** (pianoKeyCalibrationDb(midi, profile, phase) / 20);
}

export function pianoCompactSourceChannel(midi) {
  return COMPACT_SOURCE_CHANNEL[keyIndex(midi)];
}

export const PIANO_KEY_CALIBRATION = Object.freeze({
  version: 'iowa-mf-88-phase-safe-envelope-v3',
  startMidi: GRAND_START_MIDI,
  endMidi: GRAND_END_MIDI,
  keyCount: 88,
  attackWindowSeconds: Object.freeze([0.02, 0.25]),
  bodyWindowSeconds: Object.freeze([0.25, 1.2]),
  sourceSetSha256: '8886e848eee73976b5a6693e159d55df7a4438f0732eacc7c1dae60cdcd56ed0',
  fullRangeAttackDb: FULL_RANGE_ATTACK_DB,
  fullRangeBodyDb: FULL_RANGE_BODY_DB,
  compactAttackDb: COMPACT_ATTACK_DB,
  compactBodyDb: COMPACT_BODY_DB,
  compactSourceChannel: COMPACT_SOURCE_CHANNEL,
});
