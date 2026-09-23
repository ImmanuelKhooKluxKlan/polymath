const GRAND_START_MIDI = 21;
const GRAND_END_MIDI = 108;
const SMALL_SPEAKER_PROFILE = 'small-speaker';

/**
 * Iowa Steinway mf, 88-key, per-recording calibration.
 *
 * These are decibel corrections, not a blanket equal-volume curve. Every WAV
 * was measured in two windows: hammer/attack (20-250 ms) and resonant body
 * (250-1200 ms), using A-weighted energy. The compact pass additionally used
 * dual-mono audio and a conservative phone-speaker response. A 13-key rolling
 * median followed by five binomial smoothing passes preserves the piano's
 * broad acoustic contour while removing local sample and microphone jumps.
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
  1.997, 1.823, 1.729, -1.714, 1.347, -1.751, -0.688, 1.557,
  -0.344, -5.044, -0.964, 2.002, 2.83, 2.241, 0.213, -4.158,
  -0.637, -2.382, -3.107, -1.186, -1.372, 3.625, 3.278, 3.687,
  0.486, 0.213, -0.467, -0.453, -0.599, 0.75, -1.351, -4.618,
  0.748, -0.111, 0.098, 1.145, 2.808, 3.758, 1.579, 3.207,
  0.072, -0.609, -0.217, 1.928, -0.92, -0.517, -3.591, -0.397,
  0.414, 5.034, -4.038, 0.456, 1.003, 2.435, -1.694, -5.994,
  -5.208, -3.905, 8.803, 2.126, -0.456, -0.699, -0.922, 1.541,
  6.535, 0.484, -2.927, -7.23, -1.784, 1.014, 2.234, -5.349,
  3.056, -3.853, 0.765, 2.187, -1.036, 1.352, 4.255, 0.06,
  0.855, -8.498, 2.426, 0.896, -0.135, 1.922, -0.567, 1.292,
]);

const COMPACT_BODY_DB = Object.freeze([
  2.045, 1.864, 1.724, -2.074, 0.839, -2.588, -2.035, -0.387,
  -0.35, -3.38, 1.604, 0.516, 1.225, 1.796, 0.82, -5.826,
  -0.523, -3.733, -1.446, -2.382, -1.253, 6.429, 4.267, 2.853,
  -0.655, -0.74, -1.149, -2.279, -3.179, 1.075, 0.822, -0.312,
  -1.348, -1.993, -0.289, 0.97, 3.247, 3.271, 1.933, 1.844,
  1.227, -2.981, -1.521, 0.011, -0.229, 0.098, -5.446, -0.294,
  -1.365, 2.918, 1.474, 1.232, -0.563, 1.883, -1.332, -4.392,
  0.2, -5.399, 8.639, -2.014, -0.07, 1.46, -0.141, 0.012,
  2.272, 1.391, -5.025, -1.742, -3.856, 2.34, -0.456, -5.074,
  1.946, 5.056, 2.585, -3.467, -4.545, 3.229, 5.87, -1.11,
  1.348, -6.102, 4.489, -1.552, 1.925, 2.614, 2.464, 3.284,
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

export const PIANO_KEY_CALIBRATION = Object.freeze({
  version: 'iowa-mf-88-perceptual-envelope-v2',
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
});
