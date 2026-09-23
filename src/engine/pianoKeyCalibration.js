const GRAND_START_MIDI = 21;
const GRAND_END_MIDI = 108;
const SMALL_SPEAKER_PROFILE = 'small-speaker';

/**
 * Iowa Steinway mf, 88-key, per-recording calibration.
 *
 * These are decibel corrections, not a blanket equal-volume curve. Every WAV
 * was measured in two windows: hammer/attack (20-250 ms) and resonant body
 * (250-1200 ms), using A-weighted energy. The compact pass additionally used
 * a phase-safe microphone channel, the harmonic bass path, and a conservative
 * phone-speaker response.
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
  5.533, 5.086, 4.789, -1.483, 0.008, -0.156, 0.008, 4.073,
  0.671, -3.876, -1.071, -0.134, 1.434, 2.125, 0.059, -4.53,
  -3.974, -2.037, -5.106, -2.774, -0.239, 3.8, 1.892, 2.276,
  0.353, -0.122, 0.208, -1.034, -0.536, 0.996, -0.985, -2.915,
  3.872, 0.427, -1.214, -1.863, 2.881, -0.889, -0.325, 1.399,
  1.445, -2.237, 1.046, 3.106, 1.246, 0.273, -0.059, -1.391,
  -0.27, -0.152, -2.626, -3.057, 1.466, 4.347, 1.482, -3.98,
  -1.278, -2.215, -0.445, 2.437, -0.462, -3.3, 1.999, 1.063,
  6.224, -1.082, -0.408, -3.32, -3.559, -1.3, 2.622, -3.907,
  1.249, -3.471, 1.307, 3.474, -0.149, 3.729, 1.216, -0.998,
  0.663, -9.764, 1.441, 0.659, -0.239, 0.911, -0.085, 0.987,
]);

const COMPACT_BODY_DB = Object.freeze([
  4.913, 4.523, 4.105, -2.36, -0.66, -1.573, -1.654, 1.894,
  1.536, -0.628, 2.397, -0.65, 0.245, 2.271, 1.832, -5.639,
  -3.166, -3.15, -3.075, -2.494, -0.585, 5.561, 2.883, 1.694,
  -0.023, -1.31, -1.049, -1.632, -2.346, 2.436, 0.382, -0.598,
  1.342, -0.572, -2.181, -1.302, 4.435, 0.012, 1.098, 1.298,
  1.969, -3.82, -1.025, 1.485, 0.245, -0.092, -1.686, -1.085,
  -0.172, -0.235, 3.368, -1.739, 0.665, 5.305, 0.625, -1.644,
  1.784, -5.528, 1.191, -2.735, -1.517, 0.3, 1.006, 2.772,
  3.918, -2.324, -5.032, 3.308, -6.206, -0.04, -2.21, -3.225,
  -1.358, 6.47, 0.859, 0.604, 0.385, 6.816, 2.319, 0.149,
  4.436, -3.813, 6.766, 4.501, 4.259, 5.692, 3.362, 5.284,
]);

// 0 = left microphone, 1 = right microphone. The choice is based on the
// combined phone-band attack/body energy for that exact Iowa recording. A
// single real microphone is copied to both phone speakers so phase cancellation
// cannot make an otherwise loud piano key disappear.
const COMPACT_SOURCE_CHANNEL = Object.freeze([
  1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
  1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
  0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,
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
  version: 'iowa-mf-88-compact-harmonic-envelope-v4',
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
