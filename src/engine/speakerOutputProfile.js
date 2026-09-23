const OUTPUT_PROFILE_FULL_RANGE = 'full-range';
const OUTPUT_PROFILE_SMALL_SPEAKER = 'small-speaker';

export const SPEAKER_OUTPUT_MODE_LABELS = Object.freeze({
  auto: 'Automatic (recommended)',
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
  {
    deviceClass = 'desktop',
    performanceTier = 'full',
    portableSpeakerHint = false,
  } = {},
) {
  const normalized = normalizeSpeakerOutputMode(mode);
  if (normalized === 'small') return OUTPUT_PROFILE_SMALL_SPEAKER;
  if (normalized === 'full') return OUTPUT_PROFILE_FULL_RANGE;

  return deviceClass === 'phone'
    || deviceClass === 'tablet'
    || performanceTier === 'lite'
    || portableSpeakerHint
    ? OUTPUT_PROFILE_SMALL_SPEAKER
    : OUTPUT_PROFILE_FULL_RANGE;
}

/**
 * Browsers do not reveal whether sound is routed to built-in speakers,
 * headphones, or an external monitor. This deliberately conservative hint
 * catches compact computers while leaving a large desktop display alone. A
 * user can always override it with the explicit output selector.
 */
export function inferPortableSpeakerHint({
  deviceClass = 'desktop',
  screenWidth = 0,
  screenHeight = 0,
  hasBattery = false,
} = {}) {
  if (deviceClass === 'phone' || deviceClass === 'tablet') return true;
  if (hasBattery) return true;

  const width = Number(screenWidth) || 0;
  const height = Number(screenHeight) || 0;
  if (!width || !height) return false;
  const shortestSide = Math.min(width, height);
  const longestSide = Math.max(width, height);
  return shortestSide <= 1000 && longestSide <= 1728;
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
      [21, 1.28],
      [24, 1.3],
      [33, 1.24],
      [36, 1.2],
      [48, 1.1],
      [55, 1.04],
      [60, 0.96],
      [72, 0.8],
      [84, 0.7],
      [96, 0.64],
      [108, 0.6],
    ],
    key,
  );
}

/**
 * Arrangement gain is allowed to be expressive on studio monitors, but a
 * 0.25x accompaniment beside a 1.5x melody becomes almost inaudible on a
 * phone. Compress only that mix-control range; MIDI velocity still controls
 * the hammer character and remains untouched.
 */
export function speakerPerformanceGain(
  value,
  midi = 60,
  profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  const gain = Math.max(0.25, Math.min(1.5, Number(value) || 1));
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return gain;
  const compressed = 1 + ((gain - 1) * 0.44);
  const maximum = Number(midi) >= 60 ? 1.1 : 1.16;
  return Math.max(0.7, Math.min(maximum, compressed));
}

/**
 * Keep the musical stems on separate compact-speaker compressors. Role labels
 * win when the arranger supplied them; plain MIDI uploads fall back to middle
 * C so they still receive deterministic treatment.
 */
export function speakerMixBus(
  midi,
  arrangementRole = '',
  profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return 'direct';
  const role = String(arrangementRole || '').trim().toLowerCase();
  if (/melody|lead|vocal|right/.test(role)) return 'melody';
  if (/accompaniment|harmony|bass|left|support/.test(role)) return 'accompaniment';
  return Number(midi) >= 60 ? 'melody' : 'accompaniment';
}

/**
 * Per-voice EQ for a real piano sample on compact speakers. For bass notes we
 * boost an existing upper harmonic between 185 and 370 Hz instead of wasting
 * headroom on a fundamental the speaker cannot reproduce. For treble notes we
 * trim the hammer/presence bands that become a phone-speaker screech.
 */
export function speakerVoiceProfile(midi, profile = OUTPUT_PROFILE_FULL_RANGE) {
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) {
    return Object.freeze({
      compact: false,
      highPassFrequency: null,
      bodyType: null,
      bodyFrequency: null,
      bodyQ: null,
      bodyGainOffset: 0,
      hammerGainOffset: 0,
      airGainOffset: 0,
    });
  }

  const fundamental = 440 * (2 ** ((key - 69) / 12));
  let audibleHarmonic = fundamental;
  while (audibleHarmonic < 185) audibleHarmonic *= 2;
  while (audibleHarmonic > 370) audibleHarmonic /= 2;
  const bassAmount = Math.max(0, Math.min(1, (60 - key) / 24));
  const trebleAmount = Math.max(0, Math.min(1, (key - 60) / 36));

  return {
    compact: true,
    highPassFrequency: key < 48 ? 40 : 30,
    bodyType: key < 60 ? 'peaking' : 'lowshelf',
    bodyFrequency: key < 60 ? audibleHarmonic : 170,
    bodyQ: key < 60 ? 0.72 : 0.7,
    bodyGainOffset: key < 60 ? 1.1 + (3.3 * bassAmount) : -0.35 * trebleAmount,
    hammerGainOffset: -0.45 - (3.1 * trebleAmount),
    airGainOffset: -0.3 - (2.4 * trebleAmount),
  };
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
    masterLevel: Math.max(0.88, Math.min(0.94, Number(preset.masterLevel) || 0.9)),
    inputGain: Math.max(1, Math.min(1.06, Number(preset.inputGain) || 1)),
    highPassFrequency: Math.max(38, Number(preset.highPassFrequency) || 0),
    lowShelfFrequency: Math.max(210, Number(preset.lowShelfFrequency) || 0),
    lowShelfGain: (Number(preset.lowShelfGain) || 0) + 1.8,
    mudFrequency: 480,
    mudQ: 0.72,
    mudGain: Math.max(0.25, (Number(preset.mudGain) || 0) + 1.5),
    presenceGain: (Number(preset.presenceGain) || 0) - 3.6,
    airGain: (Number(preset.airGain) || 0) - 2.8,
    glueThreshold: Math.min(-20, Number(preset.glueThreshold) || -18),
    glueKnee: Math.max(26, Number(preset.glueKnee) || 0),
    glueRatio: Math.max(2.8, Number(preset.glueRatio) || 0),
    glueAttack: Math.max(0.009, Number(preset.glueAttack) || 0),
    glueRelease: Math.min(0.28, Number(preset.glueRelease) || 0.28),
    dryGain: Math.min(0.95, Math.max(0.9, Number(preset.dryGain) || 0.9)),
    wetGain: Math.min(0.075, Number(preset.wetGain) || 0),
    resonanceGain: Math.min(0.012, Number(preset.resonanceGain) || 0),
    panWidth: Math.min(0.075, Number(preset.panWidth) || 0),
    monoOutput: true,
    accompanimentBusGain: 1.06,
    accompanimentBusThreshold: -22,
    accompanimentBusKnee: 28,
    accompanimentBusRatio: 2.6,
    accompanimentBusAttack: 0.012,
    accompanimentBusRelease: 0.24,
    melodyBusGain: 0.86,
    melodyBusThreshold: -27,
    melodyBusKnee: 26,
    melodyBusRatio: 4.2,
    melodyBusAttack: 0.004,
    melodyBusRelease: 0.18,
  };
}

export const SPEAKER_OUTPUT_PROFILES = Object.freeze({
  fullRange: OUTPUT_PROFILE_FULL_RANGE,
  smallSpeaker: OUTPUT_PROFILE_SMALL_SPEAKER,
});
