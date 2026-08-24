import { IOWA_MF_SAMPLES } from '../data/iowaSampleManifest.js';
import {
  GRAND_END_MIDI,
  GRAND_START_MIDI,
} from './grandPianoLayout.js';
import {
  midiToNote,
  noteToFrequency,
  parseNote,
} from './noteMath.js';

const IOWA_MF_BASE_URL = '/samples/iowa-mf';

const MIN_GAIN = 0.00008;
const SAMPLE_FADE_SECONDS = 0.006;
const MIN_AUTOPLAY_NOTE_SECONDS = 0.035;

const STOP_ALL_RELEASE_SECONDS = 0.055;
const MIN_KEYBOARD_RELEASE_SECONDS = 0.22;
const MAX_RELEASE_SECONDS = 3.2;
const MAX_POLYPHONY = 96;
const MAX_PENDING_VOICES = 128;

export const TONE_MODE_LABELS = {
  pianella: 'Polymath Musician render',
  grand: 'Clean grand piano',
};

const TONE_PRESETS = {
  pianella: {
    label: TONE_MODE_LABELS.pianella,

    description:
      'Polymath Musician signature render: smoother attack, pedal resonance, balanced room, glue compression, and safe limiting.',

    masterLevel: 0.86,
    inputGain: 0.98,

    highPassFrequency: 25,
    highPassQ: 0.7,

    lowShelfFrequency: 135,
    lowShelfGain: 0.85,

    mudFrequency: 315,
    mudQ: 0.95,
    mudGain: -1.05,

    presenceFrequency: 2500,
    presenceQ: 0.82,
    presenceGain: 0.95,

    airFrequency: 7800,
    airGain: 0.38,

    glueThreshold: -18,
    glueKnee: 24,
    glueRatio: 2.2,
    glueAttack: 0.006,
    glueRelease: 0.34,

    limiterThreshold: -3.2,
    limiterKnee: 2,
    limiterRatio: 12,
    limiterAttack: 0.0015,
    limiterRelease: 0.075,

    dryGain: 0.8,
    wetGain: 0.2,
    resonanceGain: 0.064,

    reverbSeconds: 2.35,
    reverbSeed: 1729,
    reverbDecayPower: 2.75,
    reverbEarlyGain: 1.08,
    reverbBodyGain: 0.063,

    resonanceSeconds: 4.1,
    resonanceSeed: 2718,
    resonanceDecayPower: 3.25,
    resonanceEarlyGain: 0.55,
    resonanceBodyGain: 0.033,

    panWidth: 0.2,

    autoReleaseSeconds: 0.72,
    retriggerReleaseSeconds: 0.14,
    manualReleaseSeconds: 0.78,

    pedalReleaseExtraSeconds: 0.28,
    releaseTailRatio: 0.42,
    releaseInitialDropPortion: 0.3,
    autoplayReleaseFloorRatio: 0.62,

    velocityGainFloor: 0.38,
    velocityPower: 0.9,

    sampleAttackSeconds: 0.006,

    bodyBoost: 0.22,
    hammerSoft: -0.55,
    hammerHard: 0.82,
    highAirGain: -0.12,
  },

  grand: {
    label: TONE_MODE_LABELS.grand,

    description:
      'Cleaner raw grand piano: wider stereo, less reverb, less compression, more natural dynamics.',

    masterLevel: 0.82,
    inputGain: 0.94,

    highPassFrequency: 22,
    highPassQ: 0.62,

    lowShelfFrequency: 120,
    lowShelfGain: 0.45,

    mudFrequency: 360,
    mudQ: 0.9,
    mudGain: -0.45,

    presenceFrequency: 2200,
    presenceQ: 0.75,
    presenceGain: 0.35,

    airFrequency: 8500,
    airGain: 0.18,

    glueThreshold: -15,
    glueKnee: 18,
    glueRatio: 1.55,
    glueAttack: 0.01,
    glueRelease: 0.42,

    limiterThreshold: -2.2,
    limiterKnee: 2,
    limiterRatio: 8,
    limiterAttack: 0.0018,
    limiterRelease: 0.09,

    dryGain: 0.92,
    wetGain: 0.085,
    resonanceGain: 0.022,

    reverbSeconds: 1.65,
    reverbSeed: 9871,
    reverbDecayPower: 2.95,
    reverbEarlyGain: 0.9,
    reverbBodyGain: 0.045,

    resonanceSeconds: 3.2,
    resonanceSeed: 3617,
    resonanceDecayPower: 3.35,
    resonanceEarlyGain: 0.38,
    resonanceBodyGain: 0.018,

    panWidth: 0.34,

    autoReleaseSeconds: 0.6,
    retriggerReleaseSeconds: 0.085,
    manualReleaseSeconds: 0.68,

    pedalReleaseExtraSeconds: 0.22,
    releaseTailRatio: 0.34,
    releaseInitialDropPortion: 0.27,
    autoplayReleaseFloorRatio: 0.58,

    velocityGainFloor: 0.3,
    velocityPower: 1.08,

    sampleAttackSeconds: 0.005,

    bodyBoost: 0,
    hammerSoft: -0.25,
    hammerHard: 0.65,
    highAirGain: 0.12,
  },
};

function clamp(value, min, max) {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

function hasFiniteNumber(value) {
  return (
    value !== null &&
    value !== undefined &&
    value !== '' &&
    Number.isFinite(Number(value))
  );
}

function positiveMod(value, mod) {
  return (
    (value % mod) + mod
  ) % mod;
}

function interpolate(points, x) {
  if (x <= points[0][0]) {
    return points[0][1];
  }

  for (
    let index = 1;
    index < points.length;
    index += 1
  ) {
    const [x2, y2] = points[index];
    const [x1, y1] = points[index - 1];

    if (x <= x2) {
      const progress =
        (x - x1) /
        (x2 - x1);

      return (
        y1 +
        (y2 - y1) * progress
      );
    }
  }

  return points[
    points.length - 1
  ][1];
}

function readPreset(mode) {
  return (
    TONE_PRESETS[mode] ||
    TONE_PRESETS.pianella
  );
}

function normalizeNoteName(note) {
  return midiToNote(
    parseNote(note).midi
  );
}

function noteToFlatName(noteOrMidi) {
  const midi =
    typeof noteOrMidi === 'number'
      ? noteOrMidi
      : parseNote(noteOrMidi).midi;

  const pitchClass =
    positiveMod(midi, 12);

  const octave =
    Math.floor(midi / 12) - 1;

  const flatNames = [
    'C',
    'Db',
    'D',
    'Eb',
    'E',
    'F',
    'Gb',
    'G',
    'Ab',
    'A',
    'Bb',
    'B',
  ];

  return `${flatNames[pitchClass]}${octave}`;
}

function flatNameToMidi(note) {
  return parseNote(note).midi;
}

function buildAvailableSampleMidis() {
  const midis = [];

  for (
    const sample of IOWA_MF_SAMPLES
  ) {
    try {
      const midi =
        flatNameToMidi(sample);

      if (
        midi >= GRAND_START_MIDI &&
        midi <= GRAND_END_MIDI
      ) {
        midis.push(midi);
      }
    } catch {
      // Ignore invalid sample names.
    }
  }

  return [
    ...new Set(midis),
  ].sort((a, b) => a - b);
}

const AVAILABLE_SAMPLE_MIDIS =
  buildAvailableSampleMidis();

const AVAILABLE_SAMPLE_MIDI_SET =
  new Set(
    AVAILABLE_SAMPLE_MIDIS
  );

function chooseSampleMidi(
  requestedMidi
) {
  if (
    AVAILABLE_SAMPLE_MIDI_SET.has(
      requestedMidi
    )
  ) {
    return requestedMidi;
  }

  if (
    !AVAILABLE_SAMPLE_MIDIS.length
  ) {
    return null;
  }

  let best =
    AVAILABLE_SAMPLE_MIDIS[0];

  let bestDistance =
    Math.abs(
      requestedMidi - best
    );

  for (
    const candidate of
    AVAILABLE_SAMPLE_MIDIS
  ) {
    const distance =
      Math.abs(
        requestedMidi -
        candidate
      );

    if (
      distance < bestDistance
    ) {
      best = candidate;
      bestDistance = distance;
    }
  }

  return best;
}

function buildSamplePlan(
  requestedMidi
) {
  const midi =
    Math.round(
      Number(requestedMidi)
    );

  if (
    !Number.isFinite(midi) ||
    midi < GRAND_START_MIDI ||
    midi > GRAND_END_MIDI
  ) {
    return null;
  }

  const sampleMidi =
    chooseSampleMidi(midi);

  if (sampleMidi === null) {
    return null;
  }

  const flatName =
    noteToFlatName(sampleMidi);

  const playbackRate =
    Math.pow(
      2,
      (
        midi -
        sampleMidi
      ) / 12
    );

  return {
    requestedMidi: midi,

    requestedNote:
      midiToNote(midi),

    sampleMidi,

    sampleNote:
      midiToNote(sampleMidi),

    sampleFileNote:
      flatName,

    provider:
      sampleMidi === midi
        ? 'iowa-steinway-mf-exact-key'
        : 'iowa-steinway-mf-nearest-key-fallback',

    cacheKey:
      `iowa-mf:${flatName}`,

    urlCandidates: [
      `${IOWA_MF_BASE_URL}/${flatName}.wav`,
    ],

    playbackRate,

    exact:
      sampleMidi === midi,
  };
}

function targetRmsForMidi(midi) {
  return interpolate(
    [
      [21, 0.078],
      [33, 0.076],
      [45, 0.073],
      [60, 0.069],
      [72, 0.064],
      [84, 0.057],
      [96, 0.049],
      [108, 0.041],
    ],
    midi
  );
}

function analyzeSample(
  buffer,
  requestedMidi
) {
  let sumSquares = 0;
  let samples = 0;
  let peak = 0;

  const sampleRate =
    buffer.sampleRate;

  const start =
    Math.floor(
      sampleRate * 0.02
    );

  const end =
    Math.min(
      buffer.length,
      Math.floor(
        sampleRate * 1.2
      )
    );

  for (
    let channel = 0;
    channel <
    buffer.numberOfChannels;
    channel += 1
  ) {
    const data =
      buffer.getChannelData(
        channel
      );

    for (
      let index = start;
      index < end;
      index += 1
    ) {
      const value =
        Math.abs(data[index]);

      peak =
        Math.max(
          peak,
          value
        );

      sumSquares +=
        value * value;

      samples += 1;
    }
  }

  const rms =
    samples
      ? Math.sqrt(
          sumSquares / samples
        )
      : 0.05;

  const rmsGain =
    targetRmsForMidi(
      requestedMidi
    ) /
    Math.max(rms, 0.0001);

  const peakGainLimit =
    0.92 /
    Math.max(peak, 0.0001);

  return {
    rms,
    peak,

    targetRms:
      targetRmsForMidi(
        requestedMidi
      ),

    gain: clamp(
      Math.min(
        rmsGain,
        peakGainLimit
      ),
      0.35,
      1.95
    ),
  };
}

function loudnessSafe(value) {
  return clamp(
    Number(value) || 0.82,
    0.02,
    1.18
  );
}

function seededNoise(seed) {
  let state = seed >>> 0;

  return () => {
    state =
      (
        state * 1664525 +
        1013904223
      ) >>> 0;

    return (
      state /
      0xffffffff
    ) * 2 - 1;
  };
}

class PianoAudioEngine {
  constructor() {
    this.context = null;

    this.toneMode =
      'pianella';

    this.master = null;
    this.masterInput = null;

    this.highPass = null;
    this.warmth = null;
    this.mudControl = null;
    this.presence = null;
    this.air = null;

    this.glue = null;
    this.limiter = null;

    this.dryGain = null;
    this.wetGain = null;
    this.resonanceGain = null;

    this.reverb = null;
    this.resonance = null;

    this.wetInput = null;
    this.resonanceInput = null;

    this.active = new Map();

    this.pedalDown = false;

    this.sustainedVoices =
      new Set();

    this.bufferCache =
      new Map();

    // Sample RMS analysis used to run again for every played note.
    // Cache it once per requested MIDI so long songs do not slowly
    // consume the main thread and make Web Audio crackle.
    this.analysisCache =
      new Map();

    this.maxPolyphony =
      MAX_POLYPHONY;

    this.sampleFailures =
      new Set();

    this.warmupStarted =
      false;

    this.warnedMissingIowa =
      false;

    this.preloadPromise =
      null;

    this.preloadLoaded =
      0;

    this.preloadTotal =
      0;

    this.preloadProgressListener =
      null;
  }

  ensure() {
    if (!this.context) {
      const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

      this.context =
        new AudioContextClass();

      this.buildMasterChain();
    }

    if (
      this.context.state ===
      'suspended'
    ) {
      this.context.resume();
    }

  }

  async prepareKeyboard(onProgress) {
    this.ensure();
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
    return this.preloadCoreSamples(onProgress);
  }

  getCurrentTime() {
    this.ensure();

    return (
      this.context.currentTime
    );
  }

  getToneMode() {
    return this.toneMode;
  }

  setToneMode(
    mode = 'pianella'
  ) {
    const nextMode =
      TONE_PRESETS[mode]
        ? mode
        : 'pianella';

    this.toneMode =
      nextMode;

    if (this.context) {
      this.applyTonePreset(
        nextMode
      );
    }

    return nextMode;
  }

  getTonePresetInfo() {
    const preset =
      readPreset(
        this.toneMode
      );

    return {
      mode: this.toneMode,
      label: preset.label,

      description:
        preset.description,
    };
  }

  buildMasterChain() {
    const context =
      this.context;

    this.master =
      context.createGain();

    this.masterInput =
      context.createGain();

    this.highPass =
      context.createBiquadFilter();

    this.highPass.type =
      'highpass';

    this.warmth =
      context.createBiquadFilter();

    this.warmth.type =
      'lowshelf';

    this.mudControl =
      context.createBiquadFilter();

    this.mudControl.type =
      'peaking';

    this.presence =
      context.createBiquadFilter();

    this.presence.type =
      'peaking';

    this.air =
      context.createBiquadFilter();

    this.air.type =
      'highshelf';

    this.glue =
      context
        .createDynamicsCompressor();

    this.limiter =
      context
        .createDynamicsCompressor();

    this.dryGain =
      context.createGain();

    this.wetGain =
      context.createGain();

    this.resonanceGain =
      context.createGain();

    this.reverb =
      context.createConvolver();

    this.resonance =
      context.createConvolver();

    this.dryGain.connect(
      this.masterInput
    );

    this.reverb.connect(
      this.wetGain
    );

    this.wetGain.connect(
      this.masterInput
    );

    this.resonance.connect(
      this.resonanceGain
    );

    this.resonanceGain.connect(
      this.masterInput
    );

    this.masterInput.connect(
      this.highPass
    );

    this.highPass.connect(
      this.warmth
    );

    this.warmth.connect(
      this.mudControl
    );

    this.mudControl.connect(
      this.presence
    );

    this.presence.connect(
      this.air
    );

    this.air.connect(
      this.glue
    );

    this.glue.connect(
      this.limiter
    );

    this.limiter.connect(
      this.master
    );

    this.master.connect(
      context.destination
    );

    this.wetInput =
      this.reverb;

    this.resonanceInput =
      this.resonance;

    this.applyTonePreset(
      this.toneMode
    );
  }

  setParam(
    audioParam,
    value,
    time =
      this.context.currentTime
  ) {
    audioParam
      .cancelScheduledValues(
        time
      );

    audioParam.setValueAtTime(
      value,
      time
    );
  }

  applyTonePreset(
    mode = this.toneMode
  ) {
    if (!this.context) {
      return;
    }

    const preset =
      readPreset(mode);

    const now =
      this.context.currentTime;

    this.setParam(
      this.master.gain,
      preset.masterLevel,
      now
    );

    this.setParam(
      this.masterInput.gain,
      preset.inputGain,
      now
    );

    this.setParam(
      this.highPass.frequency,
      preset.highPassFrequency,
      now
    );

    this.setParam(
      this.highPass.Q,
      preset.highPassQ,
      now
    );

    this.setParam(
      this.warmth.frequency,
      preset.lowShelfFrequency,
      now
    );

    this.setParam(
      this.warmth.gain,
      preset.lowShelfGain,
      now
    );

    this.setParam(
      this.mudControl.frequency,
      preset.mudFrequency,
      now
    );

    this.setParam(
      this.mudControl.Q,
      preset.mudQ,
      now
    );

    this.setParam(
      this.mudControl.gain,
      preset.mudGain,
      now
    );

    this.setParam(
      this.presence.frequency,
      preset.presenceFrequency,
      now
    );

    this.setParam(
      this.presence.Q,
      preset.presenceQ,
      now
    );

    this.setParam(
      this.presence.gain,
      preset.presenceGain,
      now
    );

    this.setParam(
      this.air.frequency,
      preset.airFrequency,
      now
    );

    this.setParam(
      this.air.gain,
      preset.airGain,
      now
    );

    this.setParam(
      this.glue.threshold,
      preset.glueThreshold,
      now
    );

    this.setParam(
      this.glue.knee,
      preset.glueKnee,
      now
    );

    this.setParam(
      this.glue.ratio,
      preset.glueRatio,
      now
    );

    this.setParam(
      this.glue.attack,
      preset.glueAttack,
      now
    );

    this.setParam(
      this.glue.release,
      preset.glueRelease,
      now
    );

    this.setParam(
      this.limiter.threshold,
      preset.limiterThreshold,
      now
    );

    this.setParam(
      this.limiter.knee,
      preset.limiterKnee,
      now
    );

    this.setParam(
      this.limiter.ratio,
      preset.limiterRatio,
      now
    );

    this.setParam(
      this.limiter.attack,
      preset.limiterAttack,
      now
    );

    this.setParam(
      this.limiter.release,
      preset.limiterRelease,
      now
    );

    this.setParam(
      this.dryGain.gain,
      preset.dryGain,
      now
    );

    this.setParam(
      this.wetGain.gain,
      preset.wetGain,
      now
    );

    this.setParam(
      this.resonanceGain.gain,
      preset.resonanceGain,
      now
    );

    this.reverb.buffer =
      this.createDeterministicImpulse(
        preset.reverbSeconds,
        {
          seed:
            preset.reverbSeed,

          decayPower:
            preset
              .reverbDecayPower,

          earlyGain:
            preset
              .reverbEarlyGain,

          bodyGain:
            preset
              .reverbBodyGain,
        }
      );

    this.resonance.buffer =
      this.createDeterministicImpulse(
        preset.resonanceSeconds,
        {
          seed:
            preset
              .resonanceSeed,

          decayPower:
            preset
              .resonanceDecayPower,

          earlyGain:
            preset
              .resonanceEarlyGain,

          bodyGain:
            preset
              .resonanceBodyGain,
        }
      );
  }

  beginWarmup() {
    if (
      this.warmupStarted ||
      typeof window ===
        'undefined'
    ) {
      return;
    }

    this.warmupStarted =
      true;

    const runWarmup =
      () =>
        this.preloadCoreSamples();

    if (
      'requestIdleCallback' in
      window
    ) {
      window
        .requestIdleCallback(
          runWarmup,
          {
            timeout: 1200,
          }
        );
    } else {
      window.setTimeout(
        runWarmup,
        250
      );
    }
  }

  preloadCoreSamples(onProgress) {
    if (!this.context) {
      return Promise.resolve();
    }

    if (typeof onProgress === 'function') {
      this.preloadProgressListener = onProgress;
    }

    const reportProgress = () => {
      this.preloadProgressListener?.({
        loaded: this.preloadLoaded,
        total: this.preloadTotal,
        percent: this.preloadTotal
          ? Math.round((this.preloadLoaded / this.preloadTotal) * 100)
          : 100,
      });
    };

    if (this.preloadPromise) {
      reportProgress();
      return this.preloadPromise;
    }

    const coreStart =
      parseNote('A1').midi;

    const coreEnd =
      parseNote('C7').midi;

    const samples = [];

    for (
      const midi of
      AVAILABLE_SAMPLE_MIDIS
    ) {
      if (
        midi < coreStart ||
        midi > coreEnd
      ) {
        continue;
      }

      const info =
        buildSamplePlan(midi);

      if (info?.exact) {
        samples.push(info);
      }
    }

    this.preloadLoaded = 0;
    this.preloadTotal = samples.length;
    reportProgress();

    let nextIndex = 0;
    const worker = async () => {
      while (nextIndex < samples.length) {
        const index = nextIndex;
        nextIndex += 1;
        try {
          await this.loadSampleByInfo(samples[index]);
        } catch {
          // A missing recording can still use the built-in synth fallback.
        } finally {
          this.preloadLoaded += 1;
          reportProgress();
        }
      }
    };

    const concurrency = Math.min(4, samples.length);
    this.preloadPromise = Promise
      .all(Array.from({ length: concurrency }, () => worker()))
      .then(() => undefined);

    return this.preloadPromise;
  }

  async preloadSongNotes(song) {
    this.ensure();

    const uniqueMidis =
      new Set();

    for (
      const event of
      song?.notes || []
    ) {
      try {
        const midi =
          typeof event.midi ===
          'number'
            ? event.midi
            : parseNote(
                event.note
              ).midi;

        if (
          midi >=
            GRAND_START_MIDI &&
          midi <=
            GRAND_END_MIDI
        ) {
          uniqueMidis.add(
            midi
          );
        }
      } catch {
        // Ignore invalid song notes.
      }
    }

    const requests = [
      ...uniqueMidis,
    ].map((midi) => {
      const info =
        buildSamplePlan(midi);

      return info
        ? this
            .loadSampleByInfo(
              info
            )
            .catch(
              () => undefined
            )
        : Promise.resolve(
            undefined
          );
    });

    return Promise.all(
      requests
    );
  }

  createDeterministicImpulse(
    seconds = 2.25,
    options = {}
  ) {
    const {
      seed = 1234,
      decayPower = 2.7,
      earlyGain = 1,
      bodyGain = 0.06,
    } = options;

    const rate =
      this.context.sampleRate;

    const length =
      Math.floor(
        rate * seconds
      );

    const impulse =
      this.context.createBuffer(
        2,
        length,
        rate
      );

    for (
      let channel = 0;
      channel < 2;
      channel += 1
    ) {
      const data =
        impulse.getChannelData(
          channel
        );

      const noise =
        seededNoise(
          seed +
          channel * 1009
        );

      for (
        let index = 0;
        index < length;
        index += 1
      ) {
        const progress =
          index / length;

        const early =
          index <
          rate * 0.045
            ? earlyGain
            : 1;

        const stereoSkew =
          channel === 0
            ? 0.97
            : 1.03;

        const decay =
          Math.pow(
            1 - progress,
            decayPower
          );

        data[index] =
          noise() *
          decay *
          early *
          bodyGain *
          stereoSkew;
      }
    }

    return impulse;
  }

  getSampleInfo(
    note = 'C4'
  ) {
    const { midi } =
      parseNote(note);

    const info =
      buildSamplePlan(midi);

    if (!info) {
      return {
        requestedNote:
          note,

        mode:
          'A0-C8 grand piano; no octave folding',

        error:
          'No local Iowa samples found. Run npm run acquire:iowa-88.',

        tonePreset:
          this.getTonePresetInfo(),
      };
    }

    return {
      requestedMidi:
        info.requestedMidi,

      requestedNote:
        info.requestedNote,

      sampleNote:
        info.sampleNote,

      sampleFile:
        `${info.sampleFileNote}.wav`,

      exact:
        info.exact,

      playbackRate:
        Number(
          info.playbackRate
            .toFixed(4)
        ),

      mode:
        'A0-C8 grand piano; exact sample first; no octave folding',

      provider:
        info.provider,

      expectedFolder:
        `${IOWA_MF_BASE_URL}/`,

      fallback:
        'nearest local key only if a sample is missing; never remaps song notes to another octave',

      installedSampleCount:
        AVAILABLE_SAMPLE_MIDIS
          .length,

      tonePreset:
        this.getTonePresetInfo(),
    };
  }

  async fetchAndDecode(url) {
    const response =
      await fetch(
        url,
        {
          cache:
            'force-cache',
        }
      );

    const contentType =
      response.headers.get(
        'content-type'
      ) || '';

    if (
      !response.ok ||
      contentType.includes(
        'text/html'
      )
    ) {
      throw new Error(
        `Unable to load ${url} (${response.status}, ${
          contentType ||
          'unknown content type'
        })`
      );
    }

    const arrayBuffer =
      await response
        .arrayBuffer();

    return this.context
      .decodeAudioData(
        arrayBuffer
      );
  }

  async loadSampleByInfo(
    info
  ) {
    if (
      this.sampleFailures.has(
        info.cacheKey
      )
    ) {
      return null;
    }

    if (
      !this.bufferCache.has(
        info.cacheKey
      )
    ) {
      const request =
        (async () => {
          let lastError =
            null;

          for (
            const url of
            info.urlCandidates
          ) {
            try {
              return await this
                .fetchAndDecode(
                  url
                );
            } catch (error) {
              lastError =
                error;
            }
          }

          this.sampleFailures.add(
            info.cacheKey
          );

          if (
            !this
              .warnedMissingIowa
          ) {
            this.warnedMissingIowa =
              true;

            console.warn(
              'Polymath Musician: Iowa samples missing. Run `npm run acquire:iowa-88`, then restart the dev server. Temporary synth fallback is playing.'
            );
          }

          console.warn(
            `Polymath Musician: missing sample for ${info.sampleNote}.`,
            lastError
          );

          return null;
        })();

      this.bufferCache.set(
        info.cacheKey,
        request
      );
    }

    const buffer =
      await this.bufferCache.get(
        info.cacheKey
      );

    if (!buffer) {
      return null;
    }

    const analysisKey =
      `${info.cacheKey}:${info.requestedMidi}`;

    if (!this.analysisCache.has(analysisKey)) {
      this.analysisCache.set(
        analysisKey,
        analyzeSample(
          buffer,
          info.requestedMidi
        )
      );
    }

    return {
      buffer,
      info,
      analysis:
        this.analysisCache.get(
          analysisKey
        ),
    };
  }

  async loadSamplePlan(note) {
    const { midi } =
      parseNote(note);

    const plan =
      buildSamplePlan(midi);

    if (!plan) {
      return [];
    }

    const sample =
      await this
        .loadSampleByInfo(
          plan
        );

    return sample
      ? [sample]
      : [];
  }

  play(
    note,
    velocity = 0.86,
    duration = null,
    options = {}
  ) {
    return this.playAt(
      note,
      velocity,
      duration,
      null,
      options
    );
  }

  playAt(
    note,
    velocity = 0.86,
    duration = null,
    startAt = null,
    options = {}
  ) {
    this.ensure();

    let normalizedNote;
    let normalizedMidi;

    try {
      normalizedNote =
        normalizeNoteName(
          note
        );

      normalizedMidi =
        parseNote(
          normalizedNote
        ).midi;

      if (
        normalizedMidi <
          GRAND_START_MIDI ||
        normalizedMidi >
          GRAND_END_MIDI
      ) {
        return null;
      }
    } catch {
      return null;
    }

    const preset =
      readPreset(
        this.toneMode
      );

    const loudness =
      loudnessSafe(
        velocity
      );

    if (
      options.retriggerSameNote
    ) {
      this.release(
        normalizedNote,
        {
          releaseSeconds:
            options
              .retriggerReleaseSeconds ??
            preset
              .retriggerReleaseSeconds,

          keepPending: false,
          ignorePedal: true,
          strictRelease: true,
        }
      );
    }

    const voiceSource =
      options.source ||
      (
        duration !== null
          ? 'autoplay'
          : 'manual'
      );

    const voice =
      this.createVoiceShell(
        normalizedNote,
        voiceSource,
        normalizedMidi
      );

    voice.pendingDuration =
      duration;

    voice.pendingReleaseSeconds =
      options.releaseSeconds ??
      null;

    voice.requestedStartAt =
      startAt;

    this.trackVoice(
      normalizedNote,
      voice
    );

    this.loadSamplePlan(
      normalizedNote
    )
      .then((samples) => {
        if (
          voice.cancelPending
        ) {
          this.removeVoice(
            normalizedNote,
            voice
          );

          return;
        }

        if (
          voice.released &&
          !voice.started
        ) {
          this.removeVoice(
            normalizedNote,
            voice
          );

          return;
        }

        const safeStartAt =
          this.resolveStartAt(
            voice.requestedStartAt
          );

        if (!samples.length) {
          if (!voice.released) {
            this.playFallbackSynth(
              normalizedNote,
              loudness,
              duration,
              voice,
              safeStartAt
            );
          } else {
            this.removeVoice(
              normalizedNote,
              voice
            );
          }

          return;
        }

        if (!voice.released) {
          this.startSampleVoice(
            normalizedNote,
            loudness,
            duration,
            voice,
            samples,
            safeStartAt
          );
        }
      })
      .catch((error) => {
        console.warn(
          `Polymath Musician: failed to play ${normalizedNote}.`,
          error
        );

        this.removeVoice(
          normalizedNote,
          voice
        );
      });

    return voice;
  }

  resolveStartAt(startAt) {
    const now =
      this.context.currentTime;

    const requested =
      Number(startAt);

    if (
      !Number.isFinite(
        requested
      )
    ) {
      return now + 0.002;
    }

    return Math.max(
      requested,
      now + 0.002
    );
  }

  createVoiceShell(
    note,
    source = 'manual',
    midi = null
  ) {
    return {
      note,
      midi,
      source,

      sources: [],
      voiceGain: null,
      fallbackNodes: [],

      started: false,
      released: false,
      cancelPending: false,

      releaseTimerId: null,
      releaseCleanupTimerId:
        null,

      pendingDuration: null,
      pendingReleaseSeconds:
        null,

      requestedStartAt: null,
      startedAt: null,

      finalGain: 0.5,

      sustainedByPedal:
        false,

      createdAt:
        typeof performance !== 'undefined'
          ? performance.now()
          : Date.now(),

      graphNodes:
        new Set(),

      endedSourceCount: 0,
      cleanedUp: false,
    };
  }

  trackVoice(note, voice) {
    const voices =
      this.active.get(note) ||
      [];

    voices.push(voice);

    this.active.set(
      note,
      voices
    );

    this.enforcePolyphonyLimit();
  }

  getAllVoices() {
    return [
      ...this.active.values(),
    ].flat();
  }

  registerVoiceNodes(
    voice,
    ...nodes
  ) {
    if (!voice?.graphNodes) return;

    nodes
      .flat()
      .filter(Boolean)
      .forEach((node) => {
        voice.graphNodes.add(node);
      });
  }

  cleanupVoiceGraph(voice) {
    if (!voice || voice.cleanedUp) {
      return;
    }

    voice.cleanedUp = true;

    this.clearVoiceReleaseTimer(
      voice
    );

    for (const node of voice.graphNodes || []) {
      try {
        node.disconnect();
      } catch {
        // The node may already be disconnected.
      }
    }

    voice.graphNodes?.clear();
    voice.sources = [];
    voice.fallbackNodes = [];
    voice.voiceGain = null;
  }

  enforcePolyphonyLimit() {
    const voices = this.getAllVoices();

    if (
      voices.length <= this.maxPolyphony &&
      voices.filter((voice) => !voice.started).length <= MAX_PENDING_VOICES
    ) {
      return;
    }

    const now = this.context?.currentTime || 0;
    const targetCount = Math.max(1, this.maxPolyphony - 8);
    const victims = [...voices]
      .sort((a, b) => {
        // Preserve manual notes where possible. Old autoplay and
        // pedal-held voices are the first voices to be stolen.
        const sourcePriority =
          Number(a.source === 'manual') -
          Number(b.source === 'manual');

        if (sourcePriority !== 0) {
          return sourcePriority;
        }

        return (a.createdAt || 0) - (b.createdAt || 0);
      })
      .slice(0, Math.max(0, voices.length - targetCount));

    victims.forEach((voice) => {
      this.releaseVoice(
        voice,
        now,
        {
          ignorePedal: true,
          cancelPending: !voice.started,
          releaseSeconds: 0.018,
          strictRelease: true,
        }
      );
    });
  }

  panForMidi(midi) {
    const preset =
      readPreset(
        this.toneMode
      );

    const center =
      (
        GRAND_START_MIDI +
        GRAND_END_MIDI
      ) / 2;

    const radius =
      Math.max(
        1,
        (
          GRAND_END_MIDI -
          GRAND_START_MIDI
        ) / 2
      );

    return clamp(
      (
        (
          midi - center
        ) / radius
      ) *
      preset.panWidth,

      -0.42,
      0.42
    );
  }

  startSampleVoice(
    note,
    velocity,
    duration,
    voice,
    samples,
    startAt =
      this.context.currentTime
  ) {
    const preset =
      readPreset(
        this.toneMode
      );

    const requestedMidi =
      samples[0]
        .info
        .requestedMidi;

    voice.midi =
      requestedMidi;

    voice.startedAt =
      startAt;

    const voiceGain =
      this.context
        .createGain();

    const panNode =
      this.context
        .createStereoPanner
        ? this.context
            .createStereoPanner()
        : null;

    const highPass =
      this.context
        .createBiquadFilter();

    const body =
      this.context
        .createBiquadFilter();

    const hammerControl =
      this.context
        .createBiquadFilter();

    const airControl =
      this.context
        .createBiquadFilter();

    highPass.type =
      'highpass';

    highPass.frequency.value =
      requestedMidi < 36
        ? 14
        : requestedMidi < 48
          ? 18
          : 26;

    highPass.Q.value =
      0.6;

    body.type =
      'lowshelf';

    body.frequency.value =
      requestedMidi < 48
        ? 128
        : 170;

    body.gain.value =
      (
        requestedMidi < 40
          ? 0.72
          : requestedMidi < 58
            ? 0.24
            : -0.12
      ) +
      preset.bodyBoost;

    hammerControl.type =
      'peaking';

    hammerControl
      .frequency
      .value =
        requestedMidi < 60
          ? 1750
          : requestedMidi < 78
            ? 2450
            : 3150;

    hammerControl.Q.value =
      0.9;

    hammerControl.gain.value =
      interpolate(
        [
          [
            0.02,
            preset.hammerSoft,
          ],

          [
            0.45,
            preset.hammerSoft *
            0.32,
          ],

          [
            0.75,
            preset.hammerHard *
            0.36,
          ],

          [
            1.18,
            preset.hammerHard,
          ],
        ],

        velocity
      );

    airControl.type =
      'highshelf';

    airControl
      .frequency
      .value =
        7200;

    airControl.gain.value =
      requestedMidi > 84
        ? preset.highAirGain
        : 0.12;

    const velocityGain =
      preset.velocityGainFloor +
      Math.pow(
        velocity,
        preset.velocityPower
      ) *
      (
        1 -
        preset.velocityGainFloor
      );

    const registerGain =
      interpolate(
        [
          [21, 0.82],
          [36, 0.86],
          [60, 0.9],
          [72, 0.92],
          [96, 0.88],
          [108, 0.82],
        ],

        requestedMidi
      );

    const finalGain =
      clamp(
        registerGain *
        velocityGain,

        0.055,
        0.98
      );

    voice.finalGain =
      finalGain;

    this.registerVoiceNodes(
      voice,
      voiceGain,
      panNode,
      highPass,
      body,
      hammerControl,
      airControl
    );

    voiceGain.gain
      .setValueAtTime(
        MIN_GAIN,

        Math.max(
          this.context
            .currentTime,

          startAt - 0.002
        )
      );

    voiceGain.gain
      .exponentialRampToValueAtTime(
        finalGain,

        startAt +
        (
          preset
            .sampleAttackSeconds ??
          SAMPLE_FADE_SECONDS
        )
      );

    samples.forEach(
      (sample) => {
        const source =
          this.context
            .createBufferSource();

        const sourceGain =
          this.context
            .createGain();

        source.buffer =
          sample.buffer;

        source.playbackRate.value =
          sample.info
            .playbackRate;

        sourceGain.gain.value =
          sample.analysis.gain;

        source.connect(
          sourceGain
        );

        sourceGain.connect(
          highPass
        );

        try {
          source.start(
            startAt
          );
        } catch {
          source.start(
            this.context
              .currentTime +
            0.002
          );
        }

        this.registerVoiceNodes(
          voice,
          source,
          sourceGain
        );

        source.onended = () => {
          voice.endedSourceCount += 1;

          if (
            voice.endedSourceCount >=
            voice.sources.length
          ) {
            this.removeVoice(
              note,
              voice
            );
          }
        };

        voice.sources.push(
          source
        );
      }
    );

    highPass.connect(body);

    body.connect(
      hammerControl
    );

    hammerControl.connect(
      airControl
    );

    airControl.connect(
      voiceGain
    );

    if (panNode) {
      panNode.pan.value =
        this.panForMidi(
          requestedMidi
        );

      voiceGain.connect(
        panNode
      );

      panNode.connect(
        this.dryGain
      );

      panNode.connect(
        this.wetInput
      );

      panNode.connect(
        this.resonanceInput
      );
    } else {
      voiceGain.connect(
        this.dryGain
      );

      voiceGain.connect(
        this.wetInput
      );

      voiceGain.connect(
        this.resonanceInput
      );
    }

    voice.voiceGain =
      voiceGain;

    voice.started =
      true;

    const releaseDuration =
      duration ??
      voice.pendingDuration;

    if (
      releaseDuration !== null &&
      !voice.releaseTimerId
    ) {
      this.scheduleVoiceReleaseTimer(
        voice,
        releaseDuration,
        voice.pendingReleaseSeconds,
        startAt
      );
    }
  }

  playFallbackSynth(
    note,
    velocity = 0.8,
    duration = null,

    voice =
      this.createVoiceShell(
        note
      ),

    startAt =
      this.context.currentTime
  ) {
    const frequency =
      noteToFrequency(note);

    try {
      voice.midi =
        parseNote(note).midi;
    } catch {
      voice.midi = null;
    }

    voice.startedAt =
      startAt;

    const voiceGain =
      this.context
        .createGain();

    const toneFilter =
      this.context
        .createBiquadFilter();

    const bodyFilter =
      this.context
        .createBiquadFilter();

    const decayTime =
      this.getDecayTime(
        frequency
      );

    toneFilter.type =
      'lowpass';

    toneFilter.frequency
      .setValueAtTime(
        this.getBrightness(
          frequency
        ),

        startAt
      );

    toneFilter.frequency
      .exponentialRampToValueAtTime(
        this.getBrightness(
          frequency
        ) * 0.36,

        startAt + 0.28
      );

    toneFilter.Q
      .setValueAtTime(
        0.55,
        startAt
      );

    bodyFilter.type =
      'peaking';

    bodyFilter.frequency
      .setValueAtTime(
        410,
        startAt
      );

    bodyFilter.Q
      .setValueAtTime(
        0.82,
        startAt
      );

    bodyFilter.gain
      .setValueAtTime(
        1,
        startAt
      );

    voiceGain.gain
      .setValueAtTime(
        MIN_GAIN,

        Math.max(
          this.context
            .currentTime,

          startAt - 0.002
        )
      );

    voiceGain.gain
      .exponentialRampToValueAtTime(
        0.11 * velocity,

        startAt + 0.012
      );

    voiceGain.gain
      .exponentialRampToValueAtTime(
        0.046 * velocity,

        startAt + 0.19
      );

    voiceGain.gain
      .exponentialRampToValueAtTime(
        MIN_GAIN,

        startAt +
        decayTime
      );

    const partials = [
      {
        ratio: 1,
        gain: 1,
        detune: -1.5,
      },

      {
        ratio: 2.001,
        gain: 0.28,
        detune: 0.8,
      },

      {
        ratio: 3.002,
        gain: 0.1,
        detune: -1,
      },

      {
        ratio: 4.006,
        gain: 0.045,
        detune: 1.4,
      },
    ];

    const fallbackNodes =
      partials.map(
        (
          partial,
          index
        ) => {
          const oscillator =
            this.context
              .createOscillator();

          const gain =
            this.context
              .createGain();

          oscillator.type =
            'sine';

          oscillator.frequency
            .setValueAtTime(
              frequency *
              partial.ratio,

              startAt
            );

          oscillator.detune
            .setValueAtTime(
              partial.detune,
              startAt
            );

          gain.gain
            .setValueAtTime(
              partial.gain *
              0.32 *
              velocity,

              startAt
            );

          gain.gain
            .exponentialRampToValueAtTime(
              MIN_GAIN,

              startAt +
              decayTime +
              0.3
            );

          oscillator.connect(
            gain
          );

          gain.connect(
            toneFilter
          );

          try {
            oscillator.start(
              startAt
            );

            oscillator.stop(
              startAt +
              decayTime +
              0.5
            );
          } catch {
            const now =
              this.context
                .currentTime;

            oscillator.start(
              now + 0.002
            );

            oscillator.stop(
              now +
              decayTime +
              0.5
            );
          }

          if (
            index ===
            partials.length - 1
          ) {
            oscillator.onended =
              () =>
                this.removeVoice(
                  note,
                  voice
                );
          }

          return {
            osc: oscillator,
            gain,
          };
        }
      );

    toneFilter.connect(
      bodyFilter
    );

    bodyFilter.connect(
      voiceGain
    );

    voiceGain.connect(
      this.dryGain
    );

    voiceGain.connect(
      this.wetInput
    );

    voiceGain.connect(
      this.resonanceInput
    );

    this.registerVoiceNodes(
      voice,
      voiceGain,
      toneFilter,
      bodyFilter,
      fallbackNodes.flatMap(({ osc, gain }) => [osc, gain])
    );

    if (
      !this.active
        .get(note)
        ?.includes(voice)
    ) {
      this.trackVoice(
        note,
        voice
      );
    }

    voice.voiceGain =
      voiceGain;

    voice.fallbackNodes.push(
      ...fallbackNodes
    );

    voice.started =
      true;

    if (
      duration !== null &&
      !voice.releaseTimerId
    ) {
      this.scheduleVoiceReleaseTimer(
        voice,
        duration,
        voice.pendingReleaseSeconds,
        startAt
      );
    }

    return voice;
  }

  scheduleVoiceReleaseTimer(
    voice,
    duration,
    releaseSeconds = null,

    startAt =
      this.context.currentTime
  ) {
    const durationSeconds =
      Number(duration);

    if (
      !voice ||
      !Number.isFinite(
        durationSeconds
      ) ||
      durationSeconds < 0
    ) {
      return;
    }

    this.clearVoiceReleaseTimer(
      voice
    );

    const releaseAt =
      startAt +
      Math.max(
        MIN_AUTOPLAY_NOTE_SECONDS,
        durationSeconds
      );

    const scheduleEarlySeconds =
      0.012;

    const waitMs =
      Math.max(
        0,

        (
          releaseAt -
          this.context
            .currentTime -
          scheduleEarlySeconds
        ) * 1000
      );

    voice.releaseTimerId =
      window.setTimeout(
        () => {
          voice.releaseTimerId =
            null;

          if (!this.context) {
            return;
          }

          const releaseOptions =
            {};

          if (
            hasFiniteNumber(
              releaseSeconds
            )
          ) {
            releaseOptions
              .releaseSeconds =
                Number(
                  releaseSeconds
                );
          }

          this.releaseVoice(
            voice,
            Math.max(
              releaseAt,
              this.context
                .currentTime
            ),
            releaseOptions
          );
        },

        waitMs
      );
  }

  clearVoiceReleaseTimer(
    voice
  ) {
    if (
      voice?.releaseTimerId
    ) {
      window.clearTimeout(
        voice.releaseTimerId
      );

      voice.releaseTimerId =
        null;
    }

    if (
      voice
        ?.releaseCleanupTimerId
    ) {
      window.clearTimeout(
        voice
          .releaseCleanupTimerId
      );

      voice.releaseCleanupTimerId =
        null;
    }
  }

  getBrightness(frequency) {
    if (frequency < 90) {
      return 1900;
    }

    if (frequency < 160) {
      return 2450;
    }

    if (frequency < 320) {
      return 3800;
    }

    if (frequency < 640) {
      return 5700;
    }

    return 7400;
  }

  getDecayTime(frequency) {
    if (frequency < 90) {
      return 6.4;
    }

    if (frequency < 160) {
      return 5.5;
    }

    if (frequency < 320) {
      return 4.35;
    }

    if (frequency < 640) {
      return 3.25;
    }

    return 2.15;
  }

  getKeyboardReleaseSeconds(
    voice,
    options = {}
  ) {
    if (
      options.cancelPending
    ) {
      return (
        STOP_ALL_RELEASE_SECONDS
      );
    }

    const preset =
      readPreset(
        this.toneMode
      );

    let midi =
      Number(voice?.midi);

    if (
      !Number.isFinite(midi)
    ) {
      try {
        midi =
          parseNote(
            voice?.note ||
            'C4'
          ).midi;
      } catch {
        midi = 60;
      }
    }

    const registerMultiplier =
      interpolate(
        [
          [21, 1.38],
          [36, 1.25],
          [48, 1.12],
          [60, 1],
          [72, 0.9],
          [84, 0.8],
          [96, 0.72],
          [108, 0.66],
        ],

        midi
      );

    const baseRelease =
      voice?.source ===
      'autoplay'
        ? preset
            .autoReleaseSeconds
        : preset
            .manualReleaseSeconds;

    const pedalExtra =
      options.pedalRelease
        ? preset
            .pedalReleaseExtraSeconds
        : 0;

    return clamp(
      (
        baseRelease *
        registerMultiplier
      ) +
      pedalExtra,

      MIN_KEYBOARD_RELEASE_SECONDS,
      MAX_RELEASE_SECONDS
    );
  }

  holdAudioParamAtTime(
    audioParam,
    time,
    fallbackValue = 0.5
  ) {
    const safeFallback =
      clamp(
        Number(fallbackValue) ||
        0.5,

        MIN_GAIN,
        4
      );

    const currentValue =
      clamp(
        Number(
          audioParam.value
        ) ||
        safeFallback,

        MIN_GAIN,
        4
      );

    if (
      typeof audioParam
        .cancelAndHoldAtTime ===
      'function'
    ) {
      audioParam
        .cancelAndHoldAtTime(
          time
        );
    } else {
      audioParam
        .cancelScheduledValues(
          time
        );

      audioParam
        .setValueAtTime(
          currentValue,
          time
        );
    }

    return currentValue;
  }

  scheduleKeyboardRelease(
    audioParam,
    now,
    releaseSeconds,
    fallbackValue,
    preset
  ) {
    const heldValue =
      this.holdAudioParamAtTime(
        audioParam,
        now,
        fallbackValue
      );

    const initialDropTime =
      now +
      Math.max(
        0.025,

        releaseSeconds *
        preset
          .releaseInitialDropPortion
      );

    const tailValue =
      Math.max(
        MIN_GAIN * 4,

        heldValue *
        preset.releaseTailRatio
      );

    audioParam
      .exponentialRampToValueAtTime(
        tailValue,
        initialDropTime
      );

    audioParam
      .exponentialRampToValueAtTime(
        MIN_GAIN,
        now + releaseSeconds
      );
  }

  setSustainPedal(isDown) {
    this.ensure();

    const nextState =
      Boolean(isDown);

    if (
      nextState ===
      this.pedalDown
    ) {
      return this.pedalDown;
    }

    this.pedalDown =
      nextState;

    if (!this.pedalDown) {
      this.releaseSustainedVoices();
    }

    return this.pedalDown;
  }

  pedalDownNow() {
    return this.setSustainPedal(
      true
    );
  }

  pedalUpNow() {
    return this.setSustainPedal(
      false
    );
  }

  isSustainPedalDown() {
    return this.pedalDown;
  }

  holdVoiceWithPedal(
    voice
  ) {
    if (
      !voice ||
      voice.released
    ) {
      return;
    }

    this.clearVoiceReleaseTimer(
      voice
    );

    voice.sustainedByPedal =
      true;

    this.sustainedVoices.add(
      voice
    );
  }

  releaseSustainedVoices(
    releaseSeconds = null
  ) {
    if (
      !this.context ||
      !this.sustainedVoices
        .size
    ) {
      return;
    }

    const now =
      this.context.currentTime;

    const voices = [
      ...this.sustainedVoices,
    ];

    this.sustainedVoices.clear();

    voices.forEach(
      (voice) => {
        voice.sustainedByPedal =
          false;

        const options = {
          ignorePedal: true,
          pedalRelease: true,
        };

        if (
          hasFiniteNumber(
            releaseSeconds
          )
        ) {
          options.releaseSeconds =
            Number(
              releaseSeconds
            );
        }

        this.releaseVoice(
          voice,
          now,
          options
        );
      }
    );
  }

  release(
    note,
    options = {}
  ) {
    if (!this.context) {
      return;
    }

    let normalizedNote;

    try {
      normalizedNote =
        normalizeNoteName(
          note
        );
    } catch {
      normalizedNote =
        note;
    }

    const voices =
      this.active.get(
        normalizedNote
      ) || [];

    const now =
      this.context.currentTime;

    voices.forEach(
      (voice) => {
        const releaseOptions = {
          ...options,
        };

        if (
          !hasFiniteNumber(
            releaseOptions
              .releaseSeconds
          )
        ) {
          delete releaseOptions
            .releaseSeconds;
        }

        this.releaseVoice(
          voice,
          now,
          releaseOptions
        );
      }
    );
  }

  releaseVoice(
    voice,

    now =
      this.context.currentTime,

    options = {}
  ) {
    if (
      !voice ||
      voice.released
    ) {
      return;
    }

    const pedalMayHold =
      this.pedalDown &&
      !options.ignorePedal &&
      !options.cancelPending;

    if (pedalMayHold) {
      this.holdVoiceWithPedal(
        voice
      );

      return;
    }

    voice.released =
      true;

    voice.sustainedByPedal =
      false;

    this.sustainedVoices
      .delete(voice);

    if (
      options.cancelPending
    ) {
      voice.cancelPending =
        true;
    }

    this.clearVoiceReleaseTimer(
      voice
    );

    const preset =
      readPreset(
        this.toneMode
      );

    const naturalRelease =
      this.getKeyboardReleaseSeconds(
        voice,
        {
          cancelPending:
            options
              .cancelPending,

          pedalRelease:
            options
              .pedalRelease,
        }
      );

    const explicitRelease =
      Number(
        options.releaseSeconds
      );

    const musicalAutoplayFloor =
      voice?.source ===
        'autoplay' &&
      !options.strictRelease &&
      !options.cancelPending
        ? clamp(
            naturalRelease *
            (
              preset
                .autoplayReleaseFloorRatio ??
              0.6
            ),
            0.28,
            0.92
          )
        : 0.008;

    const releaseSeconds =
      hasFiniteNumber(
        options.releaseSeconds
      )
        ? clamp(
            Math.max(
              explicitRelease,
              musicalAutoplayFloor
            ),
            0.008,
            MAX_RELEASE_SECONDS
          )
        : naturalRelease;

    const stopAt =
      now +
      releaseSeconds +
      0.18;

    try {
      if (voice.voiceGain) {
        this.scheduleKeyboardRelease(
          voice.voiceGain.gain,
          now,
          releaseSeconds,
          voice.finalGain,
          preset
        );
      }

      voice.sources.forEach(
        (source) => {
          try {
            source.stop(
              stopAt
            );
          } catch {
            // Source already stopped.
          }
        }
      );

      voice.fallbackNodes.forEach(
        ({
          osc,
          gain,
        }) => {
          this.scheduleKeyboardRelease(
            gain.gain,
            now,
            releaseSeconds,
            gain.gain.value,
            preset
          );

          try {
            osc.stop(
              stopAt
            );
          } catch {
            // Oscillator already stopped.
          }
        }
      );

      voice.releaseCleanupTimerId =
        window.setTimeout(
          () => {
            this.removeVoice(
              voice.note,
              voice
            );
          },

          (
            releaseSeconds +
            0.3
          ) * 1000
        );
    } catch {
      // Voice may already be disconnected.
    }
  }

  removeVoice(note, voice) {
    this.sustainedVoices
      .delete(voice);

    const voices =
      this.active.get(note) ||
      [];

    const remaining =
      voices.filter(
        (candidate) =>
          candidate !== voice
      );

    if (remaining.length) {
      this.active.set(
        note,
        remaining
      );
    } else {
      this.active.delete(
        note
      );
    }

    this.cleanupVoiceGraph(
      voice
    );
  }

  stopAll(options = {}) {
    this.pedalDown =
      false;

    this.sustainedVoices
      .clear();

    [
      ...this.active.keys(),
    ].forEach((note) => {
      this.release(
        note,
        {
          cancelPending: true,
          ignorePedal: true,

          releaseSeconds:
            hasFiniteNumber(options.releaseSeconds)
              ? Number(options.releaseSeconds)
              : STOP_ALL_RELEASE_SECONDS,

          strictRelease: true,
        }
      );
    });
  }

  getDiagnostics() {
    const voices = this.getAllVoices();

    return {
      activeVoices: voices.length,
      sustainedVoices: this.sustainedVoices.size,
      pendingVoices: voices.filter((voice) => !voice.started).length,
      loadedSamples: this.bufferCache.size,
      analyzedSamples: this.analysisCache.size,
      contextState: this.context?.state || 'not-created',
      maxPolyphony: this.maxPolyphony,
    };
  }

  debugMapping(
    note = 'C4'
  ) {
    return this.getSampleInfo(
      note
    );
  }
}

export const pianoAudio =
  new PianoAudioEngine();
