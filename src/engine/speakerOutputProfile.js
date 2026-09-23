const OUTPUT_PROFILE_FULL_RANGE = 'full-range';
const OUTPUT_PROFILE_SMALL_SPEAKER = 'small-speaker';

export const SPEAKER_OUTPUT_MODE_LABELS = Object.freeze({
  auto: 'Automatic',
  small: 'Phone / laptop speakers',
  full: 'Headphones / full-range speakers',
});

export function normalizeSpeakerOutputMode(value, fallback = 'auto') {
  return Object.prototype.hasOwnProperty.call(SPEAKER_OUTPUT_MODE_LABELS, value)
    ? value
    : fallback;
}

export function resolveSpeakerOutputProfile(
  mode = 'auto',
  { deviceClass = 'desktop', performanceTier = 'full' } = {},
) {
  const normalized = normalizeSpeakerOutputMode(mode);
  if (normalized === 'small') return OUTPUT_PROFILE_SMALL_SPEAKER;
  if (normalized === 'full') return OUTPUT_PROFILE_FULL_RANGE;

  return deviceClass === 'phone'
    || deviceClass === 'tablet'
    || performanceTier === 'lite'
    ? OUTPUT_PROFILE_SMALL_SPEAKER
    : OUTPUT_PROFILE_FULL_RANGE;
}

function interpolate(points, value) {
  if (value <= points[0][0]) return points[0][1];
  for (let index = 1; index < points.length; index += 1) {
    const [rightValue, rightGain] = points[index];
    const [leftValue, leftGain] = points[index - 1];
    if (value <= rightValue) {
      const progress = (value - leftValue) / (rightValue - leftValue);
      return leftGain + (rightGain - leftGain) * progress;
    }
  }
  return points[points.length - 1][1];
}

/**
 * Per-key compensation for speakers that cannot reproduce piano fundamentals.
 *
 * Raising sub-bass only wastes limiter headroom on a phone. Instead, preserve
 * the C2-C3 body and gently trim the increasingly piercing C4-C8 range. The
 * curve is continuous so crossing middle C cannot create a sudden level jump.
 */
export function speakerRegisterGain(midi, profile = OUTPUT_PROFILE_FULL_RANGE) {
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return 1;
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
  return interpolate(
    [
      [21, 0.94],
      [33, 1.02],
      [48, 1.06],
      [55, 1.03],
      [60, 0.97],
      [72, 0.89],
      [84, 0.82],
      [96, 0.79],
      [108, 0.77],
    ],
    key,
  );
}

/**
 * Retune the shared output EQ without changing MIDI notes or hammer velocity.
 * The low shelf targets audible harmonics rather than inaudible sub-bass, while
 * the presence and air trims prevent small transducers from turning upper keys
 * into a screech. Full-range output remains byte-for-byte numerically unchanged.
 */
export function tonePresetForSpeaker(preset, profile = OUTPUT_PROFILE_FULL_RANGE) {
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return preset;
  return {
    ...preset,
    highPassFrequency: Math.max(34, Number(preset.highPassFrequency) || 0),
    lowShelfFrequency: Math.max(205, Number(preset.lowShelfFrequency) || 0),
    lowShelfGain: (Number(preset.lowShelfGain) || 0) + 1.35,
    presenceGain: (Number(preset.presenceGain) || 0) - 1.55,
    airGain: (Number(preset.airGain) || 0) - 1.2,
    dryGain: Math.min(0.92, Math.max(0.86, Number(preset.dryGain) || 0.86)),
    wetGain: Math.min(0.11, Number(preset.wetGain) || 0),
    resonanceGain: Math.min(0.025, Number(preset.resonanceGain) || 0),
  };
}

export const SPEAKER_OUTPUT_PROFILES = Object.freeze({
  fullRange: OUTPUT_PROFILE_FULL_RANGE,
  smallSpeaker: OUTPUT_PROFILE_SMALL_SPEAKER,
});
