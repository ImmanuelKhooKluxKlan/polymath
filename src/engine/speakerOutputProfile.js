const OUTPUT_PROFILE_FULL_RANGE = 'full-range';
const OUTPUT_PROFILE_SMALL_SPEAKER = 'small-speaker';

export const SPEAKER_OUTPUT_MODE_LABELS = Object.freeze({
  auto: 'Early September piano balance',
});

export function normalizeSpeakerOutputMode(value, fallback = 'auto') {
  return Object.prototype.hasOwnProperty.call(SPEAKER_OUTPUT_MODE_LABELS, value)
    ? value
    : fallback;
}

export function resolveSpeakerOutputProfile(
  mode = 'auto',
  _device = {},
) {
  // Restore the early-September playback path (aaeee7c).
  // The compact-speaker experiment made phones and battery-powered laptops
  // use a materially different piano. Until it is validated with recordings
  // from real devices, every device receives the same full-range piano path.
  normalizeSpeakerOutputMode(mode);
  return OUTPUT_PROFILE_FULL_RANGE;
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
 * Production melody emphasis layered over the early-September piano engine.
 * Accompaniment is never attenuated; positive authored melody emphasis is
 * widened gently so vocals remain clear without changing manual key balance.
 */
export function productionPerformanceGain(value) {
  const authored = Math.max(0.25, Math.min(1.5, Number(value) || 1));
  return authored > 1
    ? Math.max(1, Math.min(1.3, 1 + ((authored - 1) * 1.5)))
    : 1;
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
      [21, 1.26],
      [24, 1.28],
      [33, 1.24],
      [36, 1.2],
      [48, 1.08],
      [55, 1],
      [60, 0.9],
      [72, 0.72],
      [84, 0.61],
      [96, 0.54],
      [108, 0.5],
    ],
    key,
  );
}

/**
 * Gain applied before a piano sample enters its per-voice filters.
 *
 * Full-range output keeps the established acoustic compensation. Compact
 * output deliberately uses much less raw sub-bass gain: a phone cannot turn
 * that energy into sound, and the unused energy only drives its limiter. The
 * missing pitch is restored later with quiet upper harmonics instead.
 */
export function speakerSampleGainCompensation(
  midi,
  profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
  if (profile === OUTPUT_PROFILE_SMALL_SPEAKER) {
    return interpolate(
      [
        [21, 1.72],
        [24, 1.66],
        [28, 1.56],
        [33, 1.46],
        [36, 1.36],
        [40, 1.22],
        [48, 1],
      ],
      key,
    );
  }

  return interpolate(
    [
      [21, 3.2],
      [23, 3.1],
      [24, 2.5],
      [28, 2.25],
      [33, 1.9],
      [36, 1.65],
      [40, 1.3],
      [48, 1],
    ],
    key,
  );
}

/**
 * Dense chords need progressively more shared headroom on a phone. This gain
 * is placed before the master EQ and follows every active/releasing voice, so
 * one loud chord cannot force the device amplifier into audible crackle.
 */
export function speakerPolyphonyHeadroom(
  voiceCount,
  profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return 1;
  const count = Math.max(0, Number(voiceCount) || 0);
  return interpolate(
    [
      [0, 1],
      [3, 1],
      [6, 0.91],
      [10, 0.82],
      [16, 0.72],
      [24, 0.64],
      [36, 0.58],
      [64, 0.5],
    ],
    count,
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
  const compressed = 1 + ((gain - 1) * 0.36);
  const maximum = Number(midi) >= 60 ? 0.98 : 1.08;
  return Math.max(0.74, Math.min(maximum, compressed));
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
  source = '',
) {
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER) return 'direct';

  // A person sweeping across the keyboard must hear one continuous piano.
  // Stem buses intentionally have different dynamics and levels, so routing
  // unlabelled manual notes by middle C would create an artificial step there.
  if (String(source || '').trim().toLowerCase() === 'manual') return 'direct';

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
      harmonicDrive: 1,
    });
  }

  const fundamental = 440 * (2 ** ((key - 69) / 12));
  let harmonicNumber = 1;
  while (fundamental * harmonicNumber < 110) harmonicNumber += 2;
  let audibleHarmonic = fundamental * harmonicNumber;
  while (audibleHarmonic > 240 && harmonicNumber > 1) {
    harmonicNumber = Math.max(1, harmonicNumber - 2);
    audibleHarmonic = fundamental * harmonicNumber;
  }
  const bassAmount = Math.max(0, Math.min(1, (60 - key) / 24));
  const trebleAmount = Math.max(0, Math.min(1, (key - 60) / 36));
  const harmonicAmount = Math.max(0, Math.min(1, (52 - key) / 24));

  return {
    compact: true,
    highPassFrequency: key < 36 ? 45 : key < 48 ? 38 : 30,
    bodyType: key < 60 ? 'peaking' : 'lowshelf',
    bodyFrequency: key < 60 ? audibleHarmonic : 170,
    bodyQ: key < 60 ? 0.82 : 0.7,
    bodyGainOffset: key < 60 ? 1.4 + (4.4 * bassAmount) : -0.5 * trebleAmount,
    hammerGainOffset: -0.65 - (3.7 * trebleAmount),
    airGainOffset: -0.5 - (3 * trebleAmount),
    harmonicDrive: harmonicAmount > 0 ? 1.15 + (0.85 * harmonicAmount) : 1,
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
    masterLevel: Math.max(0.8, Math.min(0.84, Number(preset.masterLevel) || 0.82)),
    inputGain: Math.max(0.92, Math.min(0.96, Number(preset.inputGain) || 0.94)),
    highPassFrequency: Math.max(42, Number(preset.highPassFrequency) || 0),
    lowShelfFrequency: Math.max(180, Number(preset.lowShelfFrequency) || 0),
    lowShelfGain: (Number(preset.lowShelfGain) || 0) + 1.2,
    mudFrequency: 420,
    mudQ: 0.72,
    mudGain: Math.max(0.1, (Number(preset.mudGain) || 0) + 1.15),
    presenceGain: (Number(preset.presenceGain) || 0) - 4.6,
    airGain: (Number(preset.airGain) || 0) - 4,
    glueThreshold: Math.min(-23, Number(preset.glueThreshold) || -18),
    glueKnee: Math.max(26, Number(preset.glueKnee) || 0),
    glueRatio: Math.max(3.2, Number(preset.glueRatio) || 0),
    glueAttack: Math.max(0.009, Number(preset.glueAttack) || 0),
    glueRelease: Math.min(0.24, Number(preset.glueRelease) || 0.24),
    limiterThreshold: Math.min(-8, Number(preset.limiterThreshold) || -3),
    limiterKnee: Math.max(6, Number(preset.limiterKnee) || 0),
    limiterRatio: Math.max(16, Number(preset.limiterRatio) || 0),
    limiterAttack: Math.min(0.001, Number(preset.limiterAttack) || 0.001),
    limiterRelease: Math.max(0.11, Number(preset.limiterRelease) || 0.11),
    dryGain: Math.min(0.94, Math.max(0.9, Number(preset.dryGain) || 0.92)),
    wetGain: Math.min(0.05, Number(preset.wetGain) || 0),
    resonanceGain: Math.min(0.008, Number(preset.resonanceGain) || 0),
    panWidth: Math.min(0.04, Number(preset.panWidth) || 0),
    monoOutput: true,
    compactPeakProtection: true,
    accompanimentBusGain: 1.08,
    accompanimentBusThreshold: -24,
    accompanimentBusKnee: 28,
    accompanimentBusRatio: 2.8,
    accompanimentBusAttack: 0.012,
    accompanimentBusRelease: 0.24,
    melodyBusGain: 0.74,
    melodyBusThreshold: -30,
    melodyBusKnee: 26,
    melodyBusRatio: 5,
    melodyBusAttack: 0.002,
    melodyBusRelease: 0.15,
  };
}

export const SPEAKER_OUTPUT_PROFILES = Object.freeze({
  fullRange: OUTPUT_PROFILE_FULL_RANGE,
  smallSpeaker: OUTPUT_PROFILE_SMALL_SPEAKER,
});
