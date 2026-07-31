const OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64];
const MIN_GAIN = 0.0001;
const MAX_GUITAR_VOICES = 30;
const ACOUSTIC_SAMPLE_MANIFEST = `${import.meta.env.BASE_URL}samples/acoustic-guitar/manifest.json`;

export const GUITAR_TONE_LABELS = {
  lounge: 'Close-miked acoustic karaoke',
  fingerstyle: 'Warm contemporary acoustic',
  acoustic: 'Bluegrass dreadnought · Matt-style',
  bright: 'Recorded bright steel-string',
  clean: 'Clean electric guitar',
};

const TONE_PRESETS = {
  lounge: {
    input: 0.88,
    master: 0.97,
    lowShelf: 0.35,
    bodyFrequency: 188,
    bodyGain: 0.62,
    presenceFrequency: 2750,
    presenceGain: 0.82,
    highCut: 8350,
    dry: 0.975,
    wet: 0.025,
    damping: 0.9975,
    brightness: 0.94,
    pickGain: 0.12,
    sampleBodyGain: 0.7,
    samplePresenceGain: 0.72,
    sampleHighCut: 8700,
  },
  fingerstyle: {
    input: 0.86,
    master: 0.96,
    lowShelf: 0.75,
    bodyFrequency: 190,
    bodyGain: 0.55,
    presenceFrequency: 2600,
    presenceGain: 0.35,
    highCut: 7900,
    dry: 0.97,
    wet: 0.03,
    damping: 0.99745,
    brightness: 0.88,
    pickGain: 0.08,
    sampleBodyGain: 0.55,
    samplePresenceGain: 0.3,
    sampleHighCut: 8200,
  },
  acoustic: {
    input: 0.9,
    master: 0.98,
    lowShelf: 1.05,
    bodyFrequency: 205,
    bodyGain: 0.82,
    presenceFrequency: 2850,
    presenceGain: 1.15,
    highCut: 8800,
    dry: 0.95,
    wet: 0.05,
    damping: 0.99735,
    brightness: 1.02,
    pickGain: 0.2,
    sampleBodyGain: 0.75,
    samplePresenceGain: 1,
    sampleHighCut: 11200,
  },
  bright: {
    input: 0.88,
    master: 0.96,
    lowShelf: 0.6,
    bodyFrequency: 225,
    bodyGain: 0.72,
    presenceFrequency: 3150,
    presenceGain: 1.8,
    highCut: 9200,
    dry: 0.955,
    wet: 0.045,
    damping: 0.9971,
    brightness: 1.16,
    pickGain: 0.25,
    sampleBodyGain: 0.45,
    samplePresenceGain: 1.55,
    sampleHighCut: 12800,
  },
  clean: {
    input: 0.86,
    master: 0.94,
    lowShelf: -0.2,
    bodyFrequency: 180,
    bodyGain: 0.4,
    presenceFrequency: 2100,
    presenceGain: 1.4,
    highCut: 6600,
    dry: 0.96,
    wet: 0.04,
    damping: 0.9967,
    brightness: 0.86,
    pickGain: 0.13,
    sampleBodyGain: 0.25,
    samplePresenceGain: 1.15,
    sampleHighCut: 6800,
  },
};

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function midiToFrequency(midi) {
  return 440 * (2 ** ((midi - 69) / 12));
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0xffffffff;
  };
}

function createGuitarBodyImpulse(context, seconds = 1.05) {
  const length = Math.floor(context.sampleRate * seconds);
  const impulse = context.createBuffer(2, length, context.sampleRate);
    const modes = [98, 193, 382, 585, 810];

  for (let channel = 0; channel < 2; channel += 1) {
    const data = impulse.getChannelData(channel);
    const random = seededRandom(9191 + channel * 71);

    for (let index = 0; index < length; index += 1) {
      const time = index / context.sampleRate;
      const progress = index / length;
      const decay = (1 - progress) ** 3.8;
      let resonances = 0;

      modes.forEach((frequency, modeIndex) => {
        resonances += Math.sin(
          2 * Math.PI * frequency * time + channel * 0.21 * modeIndex,
        ) * (0.19 / (modeIndex + 1));
      });

      const noise = (random() * 2 - 1) * 0.018;
      data[index] = (resonances + noise) * decay;
    }
  }

  return impulse;
}

function holdParamAtTime(param, time) {
  if (typeof param.cancelAndHoldAtTime === 'function') {
    param.cancelAndHoldAtTime(time);
    return;
  }

  const current = Math.max(param.value, MIN_GAIN);
  param.cancelScheduledValues(time);
  param.setValueAtTime(current, time);
}

class GuitarEngine {
  constructor() {
    this.context = null;
    this.toneMode = 'lounge';
    this.capoFret = 0;
    this.input = null;
    this.sampleInput = null;
    this.master = null;
    this.lowBody = null;
    this.bodyResonance = null;
    this.presence = null;
    this.highCut = null;
    this.dryGain = null;
    this.wetGain = null;
    this.compressor = null;
    this.sampleBody = null;
    this.samplePresence = null;
    this.sampleHighCut = null;
    this.activeVoices = new Set();
    this.activeByString = new Map();
    this.bufferCache = new Map();
    this.variantCounter = 0;
    this.userVolume = 0.82;
    this.sampleManifest = null;
    this.sampleBuffers = new Map();
    this.sampleLoadState = 'idle';
    this.sampleLoadPromise = null;
  }

  ensure() {
    if (!this.context) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.context = new AudioContextClass({ latencyHint: 'interactive' });
      this.buildAudioGraph();
    }

    if (this.context.state === 'suspended') {
      this.context.resume();
    }

    return this.context;
  }

  getCurrentTime() {
    return this.ensure().currentTime;
  }

  preloadSamples() {
    this.ensure();
    if (this.sampleLoadPromise) return this.sampleLoadPromise;
    this.sampleLoadState = 'loading';
    this.sampleLoadPromise = fetch(ACOUSTIC_SAMPLE_MANIFEST)
      .then((response) => {
        if (!response.ok) throw new Error(`Guitar manifest failed (${response.status})`);
        return response.json();
      })
      .then(async (manifest) => {
        const baseUrl = ACOUSTIC_SAMPLE_MANIFEST.replace(/manifest\.json$/, '');
        const decoded = await Promise.all(manifest.zones.map(async (zone) => {
          const response = await fetch(`${baseUrl}${zone.file}`);
          if (!response.ok) throw new Error(`Guitar sample failed (${response.status}): ${zone.file}`);
          const buffer = await this.context.decodeAudioData(await response.arrayBuffer());
          return [zone.file, buffer];
        }));
        this.sampleManifest = manifest;
        this.sampleBuffers = new Map(decoded);
        this.sampleLoadState = 'ready';
        return true;
      })
      .catch((error) => {
        console.warn('Recorded steel-string guitar samples are unavailable.', error);
        this.sampleLoadState = 'unavailable';
        return false;
      });
    return this.sampleLoadPromise;
  }

  setToneMode(mode = 'lounge') {
    this.toneMode = TONE_PRESETS[mode] ? mode : 'lounge';
    if (this.context) this.applyTonePreset();
    return this.toneMode;
  }

  applyTonePreset() {
    const preset = TONE_PRESETS[this.toneMode];
    const now = this.context.currentTime;
    const set = (param, value) => {
      param.cancelScheduledValues(now);
      param.setTargetAtTime(value, now, 0.018);
    };

    set(this.input.gain, preset.input);
    set(this.master.gain, clamp(preset.master * (this.userVolume / 0.82), 0, 1.2));
    set(this.lowBody.gain, preset.lowShelf);
    set(this.bodyResonance.frequency, preset.bodyFrequency);
    set(this.bodyResonance.gain, preset.bodyGain);
    set(this.presence.frequency, preset.presenceFrequency);
    set(this.presence.gain, preset.presenceGain);
    set(this.highCut.frequency, preset.highCut);
    set(this.dryGain.gain, preset.dry);
    set(this.wetGain.gain, preset.wet);
    set(this.sampleBody.gain, preset.sampleBodyGain);
    set(this.samplePresence.gain, preset.samplePresenceGain);
    set(this.sampleHighCut.frequency, preset.sampleHighCut);
  }

  setCapoFret(fret = 0) {
    this.capoFret = Math.round(clamp(Number(fret) || 0, 0, 12));
    return this.capoFret;
  }

  buildAudioGraph() {
    const context = this.context;
    this.input = context.createGain();
    this.sampleInput = context.createGain();
    this.sampleInput.gain.value = 0.96;

    const highPass = context.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = 34;
    highPass.Q.value = 0.62;

    this.lowBody = context.createBiquadFilter();
    this.lowBody.type = 'lowshelf';
    this.lowBody.frequency.value = 112;

    this.bodyResonance = context.createBiquadFilter();
    this.bodyResonance.type = 'peaking';
    this.bodyResonance.Q.value = 1.05;

    const boxControl = context.createBiquadFilter();
    boxControl.type = 'peaking';
    boxControl.frequency.value = 420;
    boxControl.Q.value = 0.92;
    boxControl.gain.value = -1.35;

    this.presence = context.createBiquadFilter();
    this.presence.type = 'peaking';
    this.presence.Q.value = 0.78;

    this.highCut = context.createBiquadFilter();
    this.highCut.type = 'lowpass';
    this.highCut.Q.value = 0.66;

    this.dryGain = context.createGain();
    this.wetGain = context.createGain();

    const convolver = context.createConvolver();
    convolver.buffer = createGuitarBodyImpulse(context);

    this.compressor = context.createDynamicsCompressor();
    this.compressor.threshold.value = -16;
    this.compressor.knee.value = 18;
    this.compressor.ratio.value = 2.2;
    this.compressor.attack.value = 0.006;
    this.compressor.release.value = 0.22;

    const limiter = context.createDynamicsCompressor();
    limiter.threshold.value = -2.5;
    limiter.knee.value = 1.5;
    limiter.ratio.value = 10;
    limiter.attack.value = 0.0015;
    limiter.release.value = 0.08;

    this.master = context.createGain();

    const sampleHighPass = context.createBiquadFilter();
    sampleHighPass.type = 'highpass';
    sampleHighPass.frequency.value = 38;
    sampleHighPass.Q.value = 0.55;

    this.sampleBody = context.createBiquadFilter();
    this.sampleBody.type = 'peaking';
    this.sampleBody.frequency.value = 195;
    this.sampleBody.Q.value = 0.82;

    this.samplePresence = context.createBiquadFilter();
    this.samplePresence.type = 'peaking';
    this.samplePresence.frequency.value = 2800;
    this.samplePresence.Q.value = 0.7;

    this.sampleHighCut = context.createBiquadFilter();
    this.sampleHighCut.type = 'lowpass';
    this.sampleHighCut.Q.value = 0.58;

    this.input.connect(highPass);
    highPass.connect(this.lowBody);
    this.lowBody.connect(this.bodyResonance);
    this.bodyResonance.connect(boxControl);
    boxControl.connect(this.presence);
    this.presence.connect(this.highCut);

    this.highCut.connect(this.dryGain);
    this.dryGain.connect(this.compressor);

    this.highCut.connect(convolver);
    convolver.connect(this.wetGain);
    this.wetGain.connect(this.compressor);

    // Real recordings already contain the guitar body, pick and room.
    // Keep them out of the synthetic body resonator and feed only subtle,
    // corrective acoustic EQ into the shared compressor/limiter.
    this.sampleInput.connect(sampleHighPass);
    sampleHighPass.connect(this.sampleBody);
    this.sampleBody.connect(this.samplePresence);
    this.samplePresence.connect(this.sampleHighCut);
    this.sampleHighCut.connect(this.compressor);

    this.compressor.connect(limiter);
    limiter.connect(this.master);
    this.master.connect(context.destination);

    this.applyTonePreset();
  }

  createStringBuffer({
    frequency,
    duration,
    stringIndex,
    fret,
    brightness,
    variant,
  }) {
    const context = this.ensure();
    const preset = TONE_PRESETS[this.toneMode];
    const roundedDuration = Math.ceil(clamp(duration, 0.3, 7) * 2) / 2;
    const key = [
      this.toneMode,
      stringIndex,
      fret,
      roundedDuration,
      Math.round(brightness * 10),
      variant,
    ].join(':');

    if (this.bufferCache.has(key)) return this.bufferCache.get(key);

    const sampleRate = context.sampleRate;
    const totalSamples = Math.floor(sampleRate * (roundedDuration + 0.25));
    const buffer = context.createBuffer(1, totalSamples, sampleRate);
    const data = buffer.getChannelData(0);
    const exactDelay = sampleRate / frequency;
    const delayLength = Math.max(3, Math.floor(exactDelay));
    const fractionalDelay = exactDelay - delayLength;
    const random = seededRandom(
      1511 + stringIndex * 97 + fret * 31 + variant * 1009,
    );
    const pickPosition = clamp(0.17 + stringIndex * 0.018 + variant * 0.011, 0.12, 0.32);
    const pickOffset = Math.max(1, Math.floor(delayLength * pickPosition));
    const damping = clamp(
      preset.damping - stringIndex * 0.00009 - fret * 0.000012,
      0.9928,
      0.9983,
    );

    let smoothedNoise = 0;
    for (let index = 0; index < Math.min(delayLength + 2, data.length); index += 1) {
      const noise = random() * 2 - 1;
      smoothedNoise = smoothedNoise * 0.35 + noise * 0.65;
      const pickComb = index >= pickOffset ? data[index - pickOffset] * 0.48 : 0;
      const envelope = 1 - (index / Math.max(delayLength, 1)) * 0.28;
      data[index] = (smoothedNoise - pickComb) * envelope * brightness;
    }

    for (let index = delayLength + 2; index < data.length; index += 1) {
      const first = data[index - delayLength] * (1 - fractionalDelay)
        + data[index - delayLength - 1] * fractionalDelay;
      const second = data[index - delayLength - 1] * (1 - fractionalDelay)
        + data[index - delayLength - 2] * fractionalDelay;
      const average = (first + second) * 0.5;
      const dispersion = average + (first - second) * 0.035;
      data[index] = damping * dispersion;
    }

    this.bufferCache.set(key, buffer);
    return buffer;
  }

  createPickTransient(startAt, velocity, stringIndex, brightness, destination, voice) {
    const context = this.context;
    const length = Math.max(64, Math.floor(context.sampleRate * 0.024));
    const buffer = context.createBuffer(1, length, context.sampleRate);
    const data = buffer.getChannelData(0);
    const random = seededRandom(707 + stringIndex * 29 + this.variantCounter * 13);

    for (let index = 0; index < length; index += 1) {
      const progress = index / length;
      data[index] = (random() * 2 - 1) * ((1 - progress) ** 4);
    }

    const source = context.createBufferSource();
    source.buffer = buffer;

    const filter = context.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 2100 + stringIndex * 330;
    filter.Q.value = 0.75;

    const gain = context.createGain();
    const pickLevel = TONE_PRESETS[this.toneMode].pickGain * velocity * brightness;
    gain.gain.setValueAtTime(Math.max(pickLevel, MIN_GAIN), startAt);
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, startAt + 0.026);

    source.connect(filter);
    filter.connect(gain);
    gain.connect(destination);
    source.start(startAt);
    source.stop(startAt + 0.035);

    voice.nodes.push(source, filter, gain);
  }

  chokeString(stringIndex, at, releaseSeconds = 0.026) {
    const previousVoice = this.activeByString.get(stringIndex);
    if (!previousVoice) return;
    this.releaseVoice(previousVoice, at, releaseSeconds);
  }

  releaseVoice(voice, at = null, releaseSeconds = 0.08) {
    if (!voice || voice.released || !this.context) return;
    voice.released = true;
    const releaseAt = Math.max(
      Number.isFinite(at) ? at : this.context.currentTime,
      this.context.currentTime,
    );
    const release = clamp(Number(releaseSeconds) || 0.08, 0.012, 0.8);

    holdParamAtTime(voice.gain.gain, releaseAt);
    voice.gain.gain.exponentialRampToValueAtTime(MIN_GAIN, releaseAt + release);

    try {
      voice.source.stop(releaseAt + release + 0.025);
    } catch {
      // Source may already have ended.
    }
  }

  enforceVoiceLimit() {
    if (this.activeVoices.size < MAX_GUITAR_VOICES) return;
    const voices = [...this.activeVoices]
      .sort((a, b) => a.startedAt - b.startedAt)
      .slice(0, this.activeVoices.size - MAX_GUITAR_VOICES + 4);
    voices.forEach((voice) => this.releaseVoice(voice, this.context.currentTime, 0.018));
  }

  findSampleZone(midi, velocity) {
    if (this.sampleLoadState !== 'ready' || !this.sampleManifest) return null;
    const midiVelocity = Math.round(clamp(velocity, 0, 1) * 127);
    const matching = this.sampleManifest.zones.filter((zone) => (
      midi >= zone.lowPitch && midi <= zone.highPitch
      && midiVelocity >= zone.lowVelocity && midiVelocity <= zone.highVelocity
    ));
    if (!matching.length) return null;
    matching.sort((left, right) => (
      (left.highVelocity - left.lowVelocity) - (right.highVelocity - right.lowVelocity)
    ));
    const narrowestVelocitySpan = matching[0].highVelocity - matching[0].lowVelocity;
    const preferred = matching.filter((zone) => (
      zone.highVelocity - zone.lowVelocity === narrowestVelocitySpan
    ));
    return preferred[this.variantCounter % preferred.length];
  }

  pluckSample(stringIndex, fret, midi, velocity, startAt) {
    const context = this.context;
    const zone = this.findSampleZone(midi, velocity);
    const buffer = zone ? this.sampleBuffers.get(zone.file) : null;
    if (!zone || !buffer) return null;

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = 2 ** ((midi - zone.rootPitch) / 12);

    const warmth = context.createBiquadFilter();
    warmth.type = 'lowpass';
    warmth.frequency.value = this.toneMode === 'bright'
      ? 12800
      : this.toneMode === 'lounge'
        ? 8700
      : this.toneMode === 'fingerstyle'
        ? 8200
        : 11200;
    warmth.Q.value = 0.55;

    const gain = context.createGain();
    const peak = clamp(velocity * 0.94, 0.045, 0.9);
    const naturalLength = buffer.duration / source.playbackRate.value;
    const stopAt = startAt + naturalLength;
    const fadeAt = Math.max(startAt + 0.08, stopAt - 0.055);
    gain.gain.setValueAtTime(MIN_GAIN, startAt);
    gain.gain.linearRampToValueAtTime(peak, startAt + 0.002);
    gain.gain.setValueAtTime(peak, fadeAt);
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, stopAt);

    source.connect(warmth);
    warmth.connect(gain);
    let finalNode = gain;
    let pan = null;
    if (typeof context.createStereoPanner === 'function') {
      pan = context.createStereoPanner();
      pan.pan.value = ((stringIndex / 5) - 0.5) * 0.16;
      gain.connect(pan);
      finalNode = pan;
    }
    finalNode.connect(this.sampleInput);

    const voice = {
      source, gain, stringIndex, fret, startedAt: startAt, stopAt, released: false,
      nodes: [source, warmth, gain, pan].filter(Boolean),
    };
    this.activeVoices.add(voice);
    this.activeByString.set(stringIndex, voice);
    source.onended = () => {
      this.activeVoices.delete(voice);
      if (this.activeByString.get(stringIndex) === voice) this.activeByString.delete(stringIndex);
      voice.nodes.forEach((node) => {
        try { node.disconnect(); } catch { /* already disconnected */ }
      });
    };
    source.start(startAt);
    source.stop(stopAt + 0.025);
    return voice;
  }

  pluck(
    stringIndex,
    fret = 0,
    velocity = 0.8,
    at = null,
    options = {},
  ) {
    if (!Number.isInteger(stringIndex) || stringIndex < 0 || stringIndex >= OPEN_STRING_MIDI.length) return null;
    const numericFret = Number(fret);
    if (!Number.isFinite(numericFret) || numericFret < 0 || numericFret > 24) return null;

    const context = this.ensure();
    const preset = TONE_PRESETS[this.toneMode];
    const startAt = Math.max(Number.isFinite(at) ? at : context.currentTime + 0.002, context.currentTime + 0.002);
    const safeVelocity = clamp(Number(velocity) || 0.8, 0.03, 1);
    const duration = clamp(Number(options.duration) || 2.35, 0.12, 8);
    const requestedRelease = Number(options.releaseSeconds);
    const releaseSeconds = clamp(
      Number.isFinite(requestedRelease)
        ? Math.max(requestedRelease, 0.3)
        : 0.38,
      0.08,
      2,
    );
    const brightness = clamp((Number(options.brightness) || 1) * preset.brightness, 0.5, 1.45);
    const midi = OPEN_STRING_MIDI[stringIndex] + numericFret + this.capoFret;
    const frequency = midiToFrequency(midi);

    this.enforceVoiceLimit();
    this.chokeString(stringIndex, startAt, Number(options.chokeSeconds) || 0.026);

    const variant = this.variantCounter % 3;
    this.variantCounter += 1;

    if (this.toneMode !== 'clean') {
      const sampledVoice = this.pluckSample(
        stringIndex, numericFret, midi, safeVelocity, startAt,
      );
      if (sampledVoice) return sampledVoice;
      if (this.sampleLoadState === 'idle') this.preloadSamples();
      return null;
    }

    const source = context.createBufferSource();
    source.buffer = this.createStringBuffer({
      frequency,
      duration,
      stringIndex,
      fret: numericFret,
      brightness,
      variant,
    });
    source.detune.value = (variant - 1) * 0.55;

    const fundamental = context.createBiquadFilter();
    fundamental.type = 'highpass';
    fundamental.frequency.value = Math.max(30, frequency * 0.19);
    fundamental.Q.value = 0.52;

    const stringFilter = context.createBiquadFilter();
    stringFilter.type = 'lowpass';
    stringFilter.frequency.value = clamp(
      (5200 + stringIndex * 520 + numericFret * 45) * brightness,
      2200,
      10500,
    );
    stringFilter.Q.value = 0.58;

    const bridge = context.createBiquadFilter();
    bridge.type = 'peaking';
    bridge.frequency.value = clamp(frequency * 3.8, 580, 3400);
    bridge.Q.value = 1.15;
    bridge.gain.value = 0.8 + safeVelocity * 0.9;

    const gain = context.createGain();
    const registerBalance = 0.84 - stringIndex * 0.022;
    const peakGain = clamp(safeVelocity * registerBalance, 0.035, 0.79);
    const attackEnd = startAt + 0.0028;
    const bodyEnd = startAt + Math.min(duration, 0.24);
    const sustainEnd = startAt + duration;
    const stopAt = sustainEnd + releaseSeconds;

    gain.gain.setValueAtTime(MIN_GAIN, startAt);
    gain.gain.linearRampToValueAtTime(peakGain, attackEnd);
    gain.gain.exponentialRampToValueAtTime(Math.max(peakGain * 0.58, MIN_GAIN), bodyEnd);
    gain.gain.exponentialRampToValueAtTime(Math.max(peakGain * 0.1, MIN_GAIN), sustainEnd);
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, stopAt);

    source.connect(fundamental);
    fundamental.connect(stringFilter);
    stringFilter.connect(bridge);
    bridge.connect(gain);

    let finalNode = gain;
    let pan = null;
    if (typeof context.createStereoPanner === 'function') {
      pan = context.createStereoPanner();
      pan.pan.value = ((stringIndex / 5) - 0.5) * 0.18;
      gain.connect(pan);
      finalNode = pan;
    }
    finalNode.connect(this.input);

    const voice = {
      source,
      gain,
      stringIndex,
      fret: numericFret,
      startedAt: startAt,
      stopAt,
      released: false,
      nodes: [source, fundamental, stringFilter, bridge, gain, pan].filter(Boolean),
    };

    this.createPickTransient(startAt, safeVelocity, stringIndex, brightness, this.input, voice);
    this.activeVoices.add(voice);
    this.activeByString.set(stringIndex, voice);

    source.onended = () => {
      this.activeVoices.delete(voice);
      if (this.activeByString.get(stringIndex) === voice) this.activeByString.delete(stringIndex);
      voice.nodes.forEach((node) => {
        try { node.disconnect(); } catch { /* already disconnected */ }
      });
      voice.nodes = [];
    };

    source.start(startAt);
    source.stop(stopAt + 0.04);
    return voice;
  }

  strum(
    frets,
    velocity = 0.8,
    direction = 'down',
    at = null,
    options = {},
  ) {
    const context = this.ensure();
    const baseTime = Math.max(Number.isFinite(at) ? at : context.currentTime + 0.003, context.currentTime + 0.003);
    const indexes = direction === 'up' ? [5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5];
    const playableStrings = indexes.filter((stringIndex) => {
      const fret = Number(frets?.[stringIndex]);
      return Number.isFinite(fret) && fret >= 0;
    });
    const stringSpacing = clamp(
      Number(options.strumSpeed ?? options.spacing ?? (direction === 'up' ? 0.016 : 0.021)),
      0.006,
      0.075,
    );
    const humanize = clamp(Number(options.humanize ?? 0.0018), 0, 0.01);

    const random = seededRandom(
      Math.round(baseTime * 1000)
      + this.variantCounter * 131
      + playableStrings.length * 17,
    );

    playableStrings.forEach((stringIndex, order) => {
      const timingVariation = order === 0 ? 0 : (random() * 2 - 1) * humanize;
      const strength = clamp(
        velocity * (order === 0 ? 1 : 0.94) * (0.97 + random() * 0.06),
        0.04,
        1,
      );
      this.pluck(
        stringIndex,
        Number(frets[stringIndex]),
        strength,
        baseTime + order * stringSpacing + timingVariation,
        options,
      );
    });

    return baseTime + Math.max(0, playableStrings.length - 1) * stringSpacing;
  }

  playEvent(event, at = null, speed = 1) {
    const startAt = Number.isFinite(at) ? at : this.getCurrentTime();
    const duration = Math.max(0.12, Number(event.duration || 1.5) / speed);
    const options = {
      duration,
      releaseSeconds: Math.max(0.3, Number(event.releaseSeconds || 0.38) / speed),
      strumSpeed: Number(event.strumSpeed || 0.021) / speed,
      brightness: event.brightness,
      humanize: event.humanize,
    };

    if (Array.isArray(event.frets)) {
      return this.strum(
        event.frets,
        event.velocity ?? 0.8,
        event.direction || 'down',
        startAt,
        options,
      );
    }

    if (Number.isInteger(event.stringIndex) && Number.isFinite(Number(event.fret))) {
      return this.pluck(
        event.stringIndex,
        Number(event.fret),
        event.velocity ?? 0.8,
        startAt,
        options,
      );
    }

    return null;
  }

  stopAll(releaseSeconds = 0.035) {
    if (!this.context) return;
    const now = this.context.currentTime;
    [...this.activeVoices].forEach((voice) => {
      this.releaseVoice(voice, now, releaseSeconds);
    });
    this.activeByString.clear();
  }

  setMasterVolume(volume) {
    this.userVolume = clamp(Number(volume) || 0, 0, 1.2);
    if (!this.context || !this.master) return this.userVolume;
    const preset = TONE_PRESETS[this.toneMode];
    const target = clamp(preset.master * (this.userVolume / 0.82), 0, 1.2);
    this.master.gain.setTargetAtTime(target, this.context.currentTime, 0.02);
    return this.userVolume;
  }

  getDiagnostics() {
    return {
      activeVoices: this.activeVoices.size,
      cachedStringBuffers: this.bufferCache.size,
      toneMode: this.toneMode,
      capoFret: this.capoFret,
      contextState: this.context?.state || 'not-created',
      acousticSamples: this.sampleLoadState,
      loadedAcousticZones: this.sampleBuffers.size,
    };
  }
}

export const guitarAudio = new GuitarEngine();
