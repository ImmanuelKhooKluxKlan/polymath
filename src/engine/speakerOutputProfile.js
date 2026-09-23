const OUTPUT_PROFILE_FULL_RANGE = 'full-range';
const OUTPUT_PROFILE_SMALL_SPEAKER = 'small-speaker';

export const SPEAKER_OUTPUT_MODE_LABELS = Object.freeze({
  auto: 'Automatic',
  small: 'Phone / laptop speakers',
  full: 'Headphones / external speakers',
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
    portableSpeakerHint = false,
  } = {},
) {
  const normalized = normalizeSpeakerOutputMode(mode);
  if (normalized === 'small') return OUTPUT_PROFILE_SMALL_SPEAKER;
  if (normalized === 'full') return OUTPUT_PROFILE_FULL_RANGE;
  return deviceClass === 'phone'
    || deviceClass === 'tablet'
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
 * Production melody emphasis layered over the early-September piano engine.
 * Accompaniment is never attenuated; positive authored melody emphasis is
 * widened gently so vocals remain clear without changing manual key balance.
 */
export function productionPerformanceGain(
  value,
  { midi = 60, role = '', source = '' } = {},
) {
  if (String(source).trim().toLowerCase() === 'manual') return 1;

  const authored = Math.max(0.25, Math.min(1.5, Number(value) || 1));
  const normalizedRole = String(role).trim().toLowerCase();
  const melodyLead = 10 ** (1.5 / 20);

  if (/melody|lead|vocal|right/.test(normalizedRole)) {
    return Math.min(1.24, Math.max(melodyLead, authored));
  }

  if (/accompaniment|harmony|bass|left|support/.test(normalizedRole)) {
    return Math.min(1.05, Math.max(1, authored));
  }

  // Older MIDI/JSON files may not contain roles. Give their upper register a
  // smooth maximum +0.75 dB lift rather than creating a hard middle-C step.
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
  const upperAmount = Math.max(0, Math.min(1, (key - 55) / (84 - 55)));
  const upperRegisterGain = 1 + (((10 ** (0.75 / 20)) - 1) * upperAmount);
  const positiveAuthoredGain = Math.min(melodyLead, Math.max(1, authored));
  return Math.max(upperRegisterGain, positiveAuthoredGain);
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
      [21, 1.08],
      [33, 1.08],
      [48, 1.04],
      [60, 1],
      [72, 1],
      [84, 0.96],
      [96, 0.92],
      [108, 0.9],
    ],
    key,
  );
}

/**
 * Gain applied before a piano sample enters its per-voice filters.
 *
 * Both paths keep the listener-approved sample compensation so switching
 * output mode cannot make the left hand disappear. The compact path adds a
 * separate, quiet upper-harmonic cue for fundamentals its speaker cannot emit.
 */
export function speakerSampleGainCompensation(
  midi,
  _profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
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
      [8, 1],
      [16, 0.94],
      [24, 0.9],
      [36, 0.86],
      [64, 0.8],
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
  const compressed = 1 + ((gain - 1) * 0.5);
  const maximum = Number(midi) >= 60 ? 1.2 : 1.08;
  return Math.max(0.9, Math.min(maximum, compressed));
}

/**
 * Keep every note on one continuous dynamics path. Role balance is already
 * applied before this routing decision; separate stem compressors previously
 * exaggerated the melody/accompaniment gap on phones.
 */
export function speakerMixBus(
  midi,
  arrangementRole = '',
  profile = OUTPUT_PROFILE_FULL_RANGE,
  source = '',
) {
  void midi;
  void arrangementRole;
  void profile;
  void source;
  // Role balance is applied once by productionPerformanceGain. Running notes
  // through different compressors made the same piano behave discontinuously.
  return 'direct';
}

/**
 * Per-voice EQ for a real piano sample on compact speakers. For bass notes we
 * boost an existing upper harmonic between 130 and 260 Hz instead of wasting
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
      phaseSafeMono: false,
      usePerKeyCalibration: false,
    });
  }

  const fundamental = 440 * (2 ** ((key - 69) / 12));
  let harmonicNumber = 1;
  while (fundamental * harmonicNumber < 130) harmonicNumber += 1;
  let audibleHarmonic = fundamental * harmonicNumber;
  while (audibleHarmonic > 260 && harmonicNumber > 1) {
    harmonicNumber = Math.max(1, harmonicNumber - 1);
    audibleHarmonic = fundamental * harmonicNumber;
  }
  const bassAmount = Math.max(0, Math.min(1, (60 - key) / 39));
  const trebleAmount = Math.max(0, Math.min(1, (key - 72) / 36));

  return {
    compact: true,
    highPassFrequency: key < 36 ? 14 : key < 48 ? 18 : 26,
    bodyType: key < 60 ? 'peaking' : 'lowshelf',
    bodyFrequency: key < 60 ? audibleHarmonic : 170,
    bodyQ: key < 60 ? 0.9 : 0.7,
    bodyGainOffset: key < 60 ? 1.2 + (3 * bassAmount) : -0.2 * trebleAmount,
    hammerGainOffset: -0.15 - (1.2 * trebleAmount),
    airGainOffset: -0.1 - (0.9 * trebleAmount),
    harmonicDrive: 1,
    phaseSafeMono: false,
    usePerKeyCalibration: false,
  };
}

/**
 * Virtual-bass partials use the missing-fundamental effect: a phone renders
 * quiet upper harmonics while the listener still perceives the original low
 * piano pitch. This avoids wasting headroom by amplifying inaudible sub-bass.
 */
export function speakerVirtualBassProfile(
  midi,
  profile = OUTPUT_PROFILE_FULL_RANGE,
) {
  const key = Math.max(21, Math.min(108, Number(midi) || 60));
  if (profile !== OUTPUT_PROFILE_SMALL_SPEAKER || key > 45) {
    return Object.freeze({ enabled: false, gain: 0, harmonics: [] });
  }

  const fundamental = 440 * (2 ** ((key - 69) / 12));
  const firstHarmonic = Math.max(2, Math.ceil(120 / fundamental));
  return {
    enabled: true,
    gain: interpolate(
      [[21, 0.052], [33, 0.044], [40, 0.032], [45, 0.02]],
      key,
    ),
    harmonics: [firstHarmonic, firstHarmonic + 1, firstHarmonic + 2],
    lowPassFrequency: Math.min(720, fundamental * (firstHarmonic + 3.5)),
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
    masterLevel: Math.max(0.82, Math.min(0.84, Number(preset.masterLevel) || 0.84)),
    inputGain: Math.max(0.94, Math.min(0.96, Number(preset.inputGain) || 0.96)),
    highPassFrequency: Math.max(25, Number(preset.highPassFrequency) || 0),
    lowShelfFrequency: Math.max(180, Number(preset.lowShelfFrequency) || 0),
    lowShelfGain: (Number(preset.lowShelfGain) || 0) + 1.4,
    mudFrequency: 360,
    mudQ: 0.82,
    mudGain: (Number(preset.mudGain) || 0) + 0.5,
    presenceGain: (Number(preset.presenceGain) || 0) - 1.25,
    airGain: (Number(preset.airGain) || 0) - 1.1,
    glueThreshold: Math.min(-20, Number(preset.glueThreshold) || -18),
    glueKnee: Math.max(24, Number(preset.glueKnee) || 0),
    glueRatio: Math.max(2.6, Number(preset.glueRatio) || 0),
    glueAttack: Math.max(0.007, Number(preset.glueAttack) || 0),
    glueRelease: Math.min(0.3, Number(preset.glueRelease) || 0.3),
    limiterThreshold: Math.min(-4.5, Number(preset.limiterThreshold) || -3),
    limiterKnee: Math.max(3, Number(preset.limiterKnee) || 0),
    limiterRatio: Math.max(14, Number(preset.limiterRatio) || 0),
    limiterAttack: Math.min(0.0012, Number(preset.limiterAttack) || 0.0012),
    limiterRelease: Math.max(0.09, Number(preset.limiterRelease) || 0.09),
    dryGain: Math.min(0.9, Math.max(0.84, Number(preset.dryGain) || 0.86)),
    wetGain: Math.min(0.1, Number(preset.wetGain) || 0),
    resonanceGain: Math.min(0.02, Number(preset.resonanceGain) || 0),
    panWidth: Math.min(0.14, Number(preset.panWidth) || 0),
    monoOutput: false,
    compactPeakProtection: false,
    accompanimentBusGain: 1,
    melodyBusGain: 1,
  };
}

export const SPEAKER_OUTPUT_PROFILES = Object.freeze({
  fullRange: OUTPUT_PROFILE_FULL_RANGE,
  smallSpeaker: OUTPUT_PROFILE_SMALL_SPEAKER,
});
