import { parseNote } from './noteMath.js';
import { detectDeviceClass } from './devicePerformance.js';
import { publicAssetUrl, relativeAssetUrl } from '../services/assetUrls.js';

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function mapWithConcurrency(items, limit, task) {
  const results = new Array(items.length);
  let nextIndex = 0;
  const worker = async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await task(items[index], index);
    }
  };
  await Promise.all(Array.from(
    { length: Math.min(Math.max(1, limit), items.length) },
    () => worker(),
  ));
  return results;
}

function noteFrequency(note) {
  const midi = typeof note === 'number' ? note : parseNote(note).midi;
  return 440 * (2 ** ((midi - 69) / 12));
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function createStudioRoomImpulse(context, seconds = 1.7) {
  const length = Math.max(1, Math.floor(context.sampleRate * seconds));
  const impulse = context.createBuffer(2, length, context.sampleRate);

  for (let channel = 0; channel < 2; channel += 1) {
    const data = impulse.getChannelData(channel);
    const random = seededRandom(43117 + channel * 7919);
    for (let index = 0; index < length; index += 1) {
      const progress = index / length;
      const early = index < context.sampleRate * 0.05 ? 1.28 : 1;
      data[index] = (random() * 2 - 1) * ((1 - progress) ** 3.35) * early * 0.34;
    }
  }

  return impulse;
}

const SAMPLE_PACKS = {
  fiddle: 'violin',
  violin: 'violin',
  cello: 'cello',
  'upright-bass': 'upright-bass',
  'electric-guitar': 'electric-guitar',
  clarinet: 'clarinet',
  flute: 'flute',
  trumpet: 'trumpet',
  banjo: 'banjo',
  saxophone: 'saxophone',
};

function sampleManifestUrl(pack) {
  return publicAssetUrl(`samples/instruments/${pack}/manifest.json`);
}

const DRUM_PACK = 'drums';

const SAMPLE_PLAYBACK_PROFILES = {
  fiddle: { attack: 0.032, release: 0.46, level: 0.84, pan: -0.18, tailFloor: 0.2, highpass: 85 },
  violin: { attack: 0.035, release: 0.48, level: 0.82, pan: -0.2, tailFloor: 0.2, highpass: 90 },
  cello: { attack: 0.045, release: 0.58, level: 0.86, pan: -0.1, tailFloor: 0.24, highpass: 38 },
  'upright-bass': { attack: 0.012, release: 0.48, level: 0.92, pan: -0.08, tailFloor: 0.3, highpass: 24 },
  'electric-guitar': { attack: 0.004, release: 0.42, level: 0.9, pan: 0.14, tailFloor: 0.34, highpass: 48 },
  banjo: { attack: 0.003, release: 0.28, level: 0.9, pan: 0.16, tailFloor: 0.24, highpass: 72 },
  flute: { attack: 0.045, release: 0.32, level: 0.82, pan: 0.16, tailFloor: 0.16, highpass: 130 },
  clarinet: { attack: 0.038, release: 0.35, level: 0.84, pan: 0.08, tailFloor: 0.17, highpass: 72 },
  saxophone: { attack: 0.028, release: 0.4, level: 0.84, pan: 0.12, tailFloor: 0.18, highpass: 55 },
  trumpet: { attack: 0.018, release: 0.3, level: 0.8, pan: 0.2, tailFloor: 0.15, highpass: 105 },
};

const PLUCKED_MODELS = {
  banjo: {
    damping: 0.9938, brightness: 1.28, bodyFrequency: 820, bodyGain: 3.4,
    attack: 0.002, release: 0.24, decay: 1.1, doubleCourseCents: 0, pan: 0.16,
  },
  mandolin: {
    damping: 0.9948, brightness: 1.22, bodyFrequency: 275, bodyGain: 2.4,
    attack: 0.0025, release: 0.34, decay: 1.45, doubleCourseCents: 5.5, pan: 0.12,
  },
  dobro: {
    damping: 0.997, brightness: 1.08, bodyFrequency: 930, bodyGain: 4.8,
    attack: 0.003, release: 0.5, decay: 2.8, doubleCourseCents: 0, pan: 0.08,
  },
  ukulele: {
    damping: 0.9954, brightness: 0.94, bodyFrequency: 430, bodyGain: 2.1,
    attack: 0.0025, release: 0.3, decay: 1.55, doubleCourseCents: 0, pan: 0.16,
  },
  'electric-guitar': {
    damping: 0.9971, brightness: 1.04, bodyFrequency: 1850, bodyGain: 1.8,
    attack: 0.0025, release: 0.42, decay: 2.4, doubleCourseCents: 0, pan: 0.14,
  },
};

const PROFILES = {
  violin: {
    attack: 0.05, release: 0.5, filter: 5600, resonance: 1.3,
    oscillators: [{ type: 'sawtooth', gain: 0.58, detune: -2 }, { type: 'triangle', gain: 0.38, detune: 2 }],
    vibratoRate: 5.5, vibratoDepth: 11,
  },
  cello: {
    attack: 0.065, release: 0.62, filter: 3100, resonance: 1.1,
    oscillators: [{ type: 'sawtooth', gain: 0.42, detune: -3 }, { type: 'triangle', gain: 0.52, detune: 3 }],
    vibratoRate: 4.8, vibratoDepth: 9,
  },
  flute: {
    attack: 0.07, release: 0.38, filter: 7200, resonance: 0.8, noise: 0.035,
    oscillators: [{ type: 'sine', gain: 0.78, detune: 0 }, { type: 'triangle', gain: 0.18, detune: 0 }],
    vibratoRate: 5.1, vibratoDepth: 6,
  },
  saxophone: {
    attack: 0.045, release: 0.42, filter: 4300, resonance: 2.2, noise: 0.025,
    oscillators: [{ type: 'sawtooth', gain: 0.38, detune: -2 }, { type: 'square', gain: 0.18, detune: 2 }, { type: 'sine', gain: 0.34, detune: 0 }],
    vibratoRate: 5.2, vibratoDepth: 8,
  },
  trumpet: {
    attack: 0.035, release: 0.34, filter: 5000, resonance: 2.5,
    oscillators: [{ type: 'sawtooth', gain: 0.48, detune: 0 }, { type: 'square', gain: 0.2, detune: 0 }, { type: 'sine', gain: 0.26, detune: 0 }],
    vibratoRate: 5.6, vibratoDepth: 5,
  },
  clarinet: {
    attack: 0.055, release: 0.42, filter: 3900, resonance: 1.8, noise: 0.018,
    oscillators: [{ type: 'square', gain: 0.42, detune: 0 }, { type: 'sine', gain: 0.48, detune: 0 }],
    vibratoRate: 4.9, vibratoDepth: 4,
  },
  fiddle: {
    attack: 0.045,
    release: 0.46,
    filter: 5200,
    resonance: 1.2,
    oscillators: [
      { type: 'sawtooth', gain: 0.62, detune: -3 },
      { type: 'triangle', gain: 0.34, detune: 3 },
    ],
    vibratoRate: 5.3,
    vibratoDepth: 10,
  },
  banjo: {
    attack: 0.004,
    release: 0.44,
    filter: 6200,
    resonance: 2.8,
    decay: 0.52,
    noise: 0.13,
    oscillators: [
      { type: 'triangle', gain: 0.68, detune: 0 },
      { type: 'square', gain: 0.16, detune: 7 },
    ],
  },
  mandolin: {
    attack: 0.005,
    release: 0.5,
    filter: 5600,
    resonance: 1.9,
    decay: 0.9,
    noise: 0.08,
    oscillators: [
      { type: 'triangle', gain: 0.48, detune: -8 },
      { type: 'triangle', gain: 0.48, detune: 8 },
      { type: 'sine', gain: 0.12, detune: 0 },
    ],
  },
  dobro: {
    attack: 0.012,
    release: 0.52,
    filter: 4300,
    resonance: 5.2,
    decay: 1.45,
    noise: 0.05,
    oscillators: [
      { type: 'sawtooth', gain: 0.38, detune: -4 },
      { type: 'triangle', gain: 0.5, detune: 4 },
    ],
    vibratoRate: 4.2,
    vibratoDepth: 5,
  },
  'upright-bass': {
    attack: 0.012,
    release: 0.58,
    filter: 1400,
    resonance: 1.1,
    decay: 1.25,
    noise: 0.04,
    oscillators: [
      { type: 'sine', gain: 0.62, detune: 0 },
      { type: 'triangle', gain: 0.42, detune: -5 },
    ],
  },
  ukulele: {
    attack: 0.004,
    release: 0.3,
    filter: 5200,
    resonance: 1.7,
    decay: 0.72,
    noise: 0.07,
    oscillators: [
      { type: 'triangle', gain: 0.62, detune: -4 },
      { type: 'sine', gain: 0.28, detune: 4 },
    ],
  },
  'electric-guitar': {
    attack: 0.006,
    release: 0.44,
    filter: 3900,
    resonance: 2.4,
    decay: 1.1,
    noise: 0.045,
    oscillators: [
      { type: 'sawtooth', gain: 0.42, detune: -5 },
      { type: 'triangle', gain: 0.42, detune: 5 },
      { type: 'square', gain: 0.08, detune: 0 },
    ],
  },
  synth: {
    attack: 0.018,
    release: 0.5,
    filter: 6500,
    resonance: 1.8,
    oscillators: [
      { type: 'sawtooth', gain: 0.38, detune: -7 },
      { type: 'sawtooth', gain: 0.38, detune: 7 },
      { type: 'sine', gain: 0.22, detune: 0 },
    ],
    vibratoRate: 5.8,
    vibratoDepth: 4,
  },
};

class EnsembleAudioEngine {
  constructor() {
    this.context = null;
    this.input = null;
    this.master = null;
    this.dry = null;
    this.wet = null;
    this.room = null;
    this.voices = new Set();
    this.volume = 0.9;
    this.noiseCounter = 0;
    this.samplePacks = new Map();
    this.sampleLoadPromises = new Map();
    this.sampleLoadControllers = new Map();
    this.samplePackGenerations = new Map();
    this.modelBufferCache = new Map();
  }

  ensure() {
    if (!this.context) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Web Audio is not supported by this browser.');
      this.context = new AudioContextClass({ latencyHint: 'interactive' });

      this.input = this.context.createGain();

      const highPass = this.context.createBiquadFilter();
      highPass.type = 'highpass';
      highPass.frequency.value = 24;
      highPass.Q.value = 0.65;

      const warmth = this.context.createBiquadFilter();
      warmth.type = 'lowshelf';
      warmth.frequency.value = 150;
      warmth.gain.value = 0.8;

      const mudControl = this.context.createBiquadFilter();
      mudControl.type = 'peaking';
      mudControl.frequency.value = 360;
      mudControl.Q.value = 0.9;
      mudControl.gain.value = -1.35;

      const presence = this.context.createBiquadFilter();
      presence.type = 'peaking';
      presence.frequency.value = 2600;
      presence.Q.value = 0.72;
      presence.gain.value = 0.7;

      this.dry = this.context.createGain();
      this.dry.gain.value = 0.86;

      this.wet = this.context.createGain();
      this.wet.gain.value = 0.14;

      this.room = this.context.createConvolver();
      this.room.buffer = createStudioRoomImpulse(this.context);

      const compressor = this.context.createDynamicsCompressor();
      compressor.threshold.value = -17;
      compressor.knee.value = 20;
      compressor.ratio.value = 2.2;
      compressor.attack.value = 0.008;
      compressor.release.value = 0.28;

      const limiter = this.context.createDynamicsCompressor();
      limiter.threshold.value = -2.4;
      limiter.knee.value = 1.5;
      limiter.ratio.value = 10;
      limiter.attack.value = 0.0015;
      limiter.release.value = 0.08;

      this.master = this.context.createGain();
      this.master.gain.value = this.volume;

      this.input.connect(highPass);
      highPass.connect(warmth);
      warmth.connect(mudControl);
      mudControl.connect(presence);

      presence.connect(this.dry);
      this.dry.connect(compressor);

      presence.connect(this.room);
      this.room.connect(this.wet);
      this.wet.connect(compressor);

      compressor.connect(limiter);
      limiter.connect(this.master);
      this.master.connect(this.context.destination);
    }
    if (this.context.state === 'suspended') this.context.resume();
    return this.context;
  }

  setMasterVolume(value) {
    this.volume = clamp(Number(value) || 0.9, 0, 1.2);
    if (this.master && this.context) {
      this.master.gain.setTargetAtTime(this.volume, this.context.currentTime, 0.02);
    }
  }

  getCurrentTime() {
    return this.ensure().currentTime;
  }

  releaseSamplePacksExcept(keepPack) {
    [...this.samplePacks.keys()].forEach((pack) => {
      if (pack === keepPack) return;
      this.sampleLoadControllers.get(pack)?.abort();
      this.sampleLoadControllers.delete(pack);
      this.sampleLoadPromises.delete(pack);
      this.samplePacks.delete(pack);
      this.samplePackGenerations.set(pack, (this.samplePackGenerations.get(pack) || 0) + 1);
    });
  }

  releaseModelBuffersExcept(instrument) {
    [...this.modelBufferCache.keys()].forEach((key) => {
      if (!key.startsWith(`${instrument}:`)) this.modelBufferCache.delete(key);
    });
  }

  preloadInstrument(instrument, options = {}) {
    if (instrument === 'drums') return this.preloadDrums(options);
    const pack = SAMPLE_PACKS[instrument];
    if (options.exclusive) {
      this.releaseSamplePacksExcept(pack || null);
      this.releaseModelBuffersExcept(instrument);
    }
    if (!pack) return Promise.resolve(false);
    if (this.samplePacks.get(pack)?.state === 'ready') return Promise.resolve(true);
    if (this.sampleLoadPromises.has(pack)) return this.sampleLoadPromises.get(pack);
    this.ensure();
    const generation = (this.samplePackGenerations.get(pack) || 0) + 1;
    this.samplePackGenerations.set(pack, generation);
    const controller = new window.AbortController();
    this.sampleLoadControllers.set(pack, controller);
    this.samplePacks.set(pack, { state: 'loading', manifest: null, buffers: new Map() });
    const manifestUrl = sampleManifestUrl(pack);
    const promise = fetch(manifestUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`${pack} manifest failed (${response.status})`);
        return response.json();
      })
      .then(async (manifest) => {
        const concurrency = detectDeviceClass() === 'desktop' ? 4 : 1;
        const decoded = await mapWithConcurrency(manifest.zones, concurrency, async (zone) => {
          const response = await fetch(relativeAssetUrl(manifestUrl, zone.file), {
            cache: 'force-cache',
            signal: controller.signal,
          });
          if (!response.ok) throw new Error(`${pack} sample failed (${response.status}): ${zone.file}`);
          return [zone.file, await this.context.decodeAudioData(await response.arrayBuffer())];
        });
        if (this.samplePackGenerations.get(pack) !== generation) return false;
        this.samplePacks.set(pack, { state: 'ready', manifest, buffers: new Map(decoded) });
        return true;
      })
      .catch((error) => {
        if (error?.name === 'AbortError' || this.samplePackGenerations.get(pack) !== generation) {
          return false;
        }
        console.warn(`${instrument} recordings unavailable; synthesis fallback remains active.`, error);
        this.samplePacks.set(pack, { state: 'fallback', manifest: null, buffers: new Map() });
        return false;
      })
      .finally(() => {
        if (this.samplePackGenerations.get(pack) === generation) {
          this.sampleLoadPromises.delete(pack);
          this.sampleLoadControllers.delete(pack);
        }
      });
    this.sampleLoadPromises.set(pack, promise);
    return promise;
  }

  preloadDrums(options = {}) {
    if (options.exclusive) {
      this.releaseSamplePacksExcept(DRUM_PACK);
      this.modelBufferCache.clear();
    }
    if (this.samplePacks.get(DRUM_PACK)?.state === 'ready') return Promise.resolve(true);
    if (this.sampleLoadPromises.has(DRUM_PACK)) return this.sampleLoadPromises.get(DRUM_PACK);
    this.ensure();
    const generation = (this.samplePackGenerations.get(DRUM_PACK) || 0) + 1;
    this.samplePackGenerations.set(DRUM_PACK, generation);
    const controller = new window.AbortController();
    this.sampleLoadControllers.set(DRUM_PACK, controller);
    this.samplePacks.set(DRUM_PACK, { state: 'loading', manifest: null, buffers: new Map() });
    const manifestUrl = sampleManifestUrl(DRUM_PACK);
    const promise = fetch(manifestUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`drum manifest failed (${response.status})`);
        return response.json();
      })
      .then(async (manifest) => {
        const concurrency = detectDeviceClass() === 'desktop' ? 4 : 1;
        const decoded = await mapWithConcurrency(manifest.zones, concurrency, async (zone) => {
          const response = await fetch(relativeAssetUrl(manifestUrl, `samples/${zone.file}`), {
            cache: 'force-cache',
            signal: controller.signal,
          });
          if (!response.ok) throw new Error(`drum sample failed (${response.status}): ${zone.file}`);
          return [zone.file, await this.context.decodeAudioData(await response.arrayBuffer())];
        });
        if (this.samplePackGenerations.get(DRUM_PACK) !== generation) return false;
        this.samplePacks.set(DRUM_PACK, { state: 'ready', manifest, buffers: new Map(decoded) });
        return true;
      })
      .catch((error) => {
        if (error?.name === 'AbortError' || this.samplePackGenerations.get(DRUM_PACK) !== generation) {
          return false;
        }
        console.warn('Drum recordings unavailable; synthesis fallback remains active.', error);
        this.samplePacks.set(DRUM_PACK, { state: 'fallback', manifest: null, buffers: new Map() });
        return false;
      })
      .finally(() => {
        if (this.samplePackGenerations.get(DRUM_PACK) === generation) {
          this.sampleLoadPromises.delete(DRUM_PACK);
          this.sampleLoadControllers.delete(DRUM_PACK);
        }
      });
    this.sampleLoadPromises.set(DRUM_PACK, promise);
    return promise;
  }

  findSampleZone(instrument, midi, velocity) {
    const pack = SAMPLE_PACKS[instrument];
    const loaded = pack ? this.samplePacks.get(pack) : null;
    if (!loaded || loaded.state !== 'ready') return null;
    const midiVelocity = Math.round(clamp(velocity, 0, 1) * 127);
    const lowestPitch = Math.min(...loaded.manifest.zones.map((zone) => zone.lowPitch));
    const highestPitch = Math.max(...loaded.manifest.zones.map((zone) => zone.highPitch));
    if (midi < lowestPitch || midi > highestPitch) return null;
    let candidates = loaded.manifest.zones.filter((zone) => (
      midi >= zone.lowPitch && midi <= zone.highPitch
      && midiVelocity >= zone.lowVelocity && midiVelocity <= zone.highVelocity
    ));
    if (!candidates.length) {
      const nearestDistance = Math.min(...loaded.manifest.zones.map((zone) => (
        Math.abs(midi - clamp(midi, zone.lowPitch, zone.highPitch))
        + Math.abs(midiVelocity - clamp(midiVelocity, zone.lowVelocity, zone.highVelocity)) / 127
      )));
      candidates = loaded.manifest.zones.filter((zone) => (
        Math.abs(midi - clamp(midi, zone.lowPitch, zone.highPitch))
        + Math.abs(midiVelocity - clamp(midiVelocity, zone.lowVelocity, zone.highVelocity)) / 127
      ) <= nearestDistance + 0.0001);
    }
    const zone = candidates[this.noiseCounter % Math.max(1, candidates.length)] || null;
    return zone ? { zone, buffer: loaded.buffers.get(zone.file) } : null;
  }

  playSampleAt(note, instrument, velocity, duration, when) {
    const context = this.ensure();
    const midi = typeof note === 'number' ? note : parseNote(note).midi;
    const sample = this.findSampleZone(instrument, midi, velocity);
    if (!sample?.buffer) return null;
    this.noiseCounter += 1;
    const startAt = Math.max(context.currentTime, Number.isFinite(when) ? when : context.currentTime);
    const source = context.createBufferSource();
    source.buffer = sample.buffer;
    source.playbackRate.value = 2 ** ((midi - sample.zone.rootPitch) / 12);
    const profile = SAMPLE_PLAYBACK_PROFILES[instrument] || {
      attack: 0.01, release: 0.36, level: 0.72, pan: 0, tailFloor: 0.2, highpass: 35,
    };
    const naturalDuration = sample.buffer.duration / source.playbackRate.value;
    const requestedDuration = clamp(Number(duration) || 0.65, 0.05, 18);
    const releaseAt = startAt + Math.min(requestedDuration, Math.max(0.08, naturalDuration - 0.08));
    const availableTail = Math.max(profile.tailFloor, Math.min(profile.release, naturalDuration - (releaseAt - startAt)));
    const stopAt = Math.min(startAt + naturalDuration, releaseAt + availableTail);
    const voiceGain = context.createGain();
    const level = clamp(Number(velocity) || 0.78, 0.03, 1.1) * profile.level;
    voiceGain.gain.setValueAtTime(0.0001, startAt);
    voiceGain.gain.linearRampToValueAtTime(level, startAt + profile.attack);
    voiceGain.gain.setTargetAtTime(level * 0.94, startAt + profile.attack, 0.16);
    if (typeof voiceGain.gain.cancelAndHoldAtTime === 'function') {
      voiceGain.gain.cancelAndHoldAtTime(releaseAt);
    } else {
      voiceGain.gain.cancelScheduledValues(releaseAt);
      voiceGain.gain.setValueAtTime(level * 0.9, releaseAt);
    }
    voiceGain.gain.exponentialRampToValueAtTime(0.0001, Math.max(startAt + 0.06, stopAt));
    const highpass = context.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = profile.highpass;
    highpass.Q.value = 0.55;
    source.connect(highpass);
    highpass.connect(voiceGain);
    let output = voiceGain;
    let panner = null;
    if (typeof context.createStereoPanner === 'function') {
      panner = context.createStereoPanner();
      panner.pan.value = profile.pan;
      voiceGain.connect(panner);
      output = panner;
    }
    output.connect(this.input);
    const voice = {
      oscillators: [source],
      vibrato: null,
      noiseSource: null,
      voiceGain,
      stopAt,
      nodes: [source, highpass, voiceGain, panner].filter(Boolean),
    };
    this.voices.add(voice);
    source.onended = () => {
      this.voices.delete(voice);
      voice.nodes.forEach((node) => {
        try { node.disconnect(); } catch { /* already disconnected */ }
      });
    };
    source.start(startAt);
    source.stop(stopAt + 0.025);
    return voice;
  }

  createNoiseBurst(startAt, duration, amount, destination) {
    if (!amount || !this.context) return null;
    const frameCount = Math.max(1, Math.floor(this.context.sampleRate * Math.min(0.08, duration)));
    const buffer = this.context.createBuffer(1, frameCount, this.context.sampleRate);
    const data = buffer.getChannelData(0);
    const random = seededRandom(7331 + this.noiseCounter * 97);
    this.noiseCounter += 1;
    for (let i = 0; i < data.length; i += 1) {
      data[i] = (random() * 2 - 1) * ((1 - i / data.length) ** 2.05);
    }
    const source = this.context.createBufferSource();
    const gain = this.context.createGain();
    gain.gain.setValueAtTime(amount, startAt);
    gain.gain.exponentialRampToValueAtTime(0.0001, startAt + Math.min(0.08, duration));
    source.buffer = buffer;
    source.connect(gain);
    gain.connect(destination);
    source.start(startAt);
    source.stop(startAt + Math.min(0.09, duration + 0.01));
    return source;
  }

  createPluckedBuffer(instrument, midi, duration, variant = 0, detuneCents = 0) {
    const context = this.ensure();
    const profile = PLUCKED_MODELS[instrument];
    const frequency = noteFrequency(midi) * (2 ** (detuneCents / 1200));
    const roundedDuration = Math.ceil(clamp(duration, 0.3, 7) * 2) / 2;
    const key = `${instrument}:${midi}:${roundedDuration}:${variant}:${detuneCents}`;
    if (this.modelBufferCache.has(key)) return this.modelBufferCache.get(key);
    const length = Math.floor(context.sampleRate * (roundedDuration + profile.release + 0.12));
    const buffer = context.createBuffer(1, length, context.sampleRate);
    const data = buffer.getChannelData(0);
    const exactDelay = context.sampleRate / frequency;
    const delay = Math.max(3, Math.floor(exactDelay));
    const fraction = exactDelay - delay;
    const random = seededRandom(1907 + midi * 41 + variant * 977 + Math.round(detuneCents * 10));
    const pickPosition = instrument === 'dobro' ? 0.29 : instrument === 'mandolin' ? 0.17 : 0.2;
    const pickOffset = Math.max(1, Math.floor(delay * pickPosition));
    let smoothNoise = 0;
    for (let index = 0; index < Math.min(delay + 2, length); index += 1) {
      smoothNoise = smoothNoise * 0.28 + (random() * 2 - 1) * 0.72;
      const comb = index >= pickOffset ? data[index - pickOffset] * 0.5 : 0;
      data[index] = (smoothNoise - comb) * profile.brightness;
    }
    for (let index = delay + 2; index < length; index += 1) {
      const first = data[index - delay] * (1 - fraction) + data[index - delay - 1] * fraction;
      const second = data[index - delay - 1] * (1 - fraction) + data[index - delay - 2] * fraction;
      data[index] = profile.damping * ((first + second) * 0.5 + (first - second) * 0.025);
    }
    this.modelBufferCache.set(key, buffer);
    return buffer;
  }

  playPluckedModelAt(note, instrument, velocity, duration, when) {
    const context = this.ensure();
    const profile = PLUCKED_MODELS[instrument];
    if (!profile) return null;
    const midi = typeof note === 'number' ? note : parseNote(note).midi;
    const startAt = Math.max(context.currentTime, Number.isFinite(when) ? when : context.currentTime);
    const requestedDuration = clamp(Number(duration) || 0.7, 0.08, 8);
    const variant = this.noiseCounter % 3;
    this.noiseCounter += 1;
    const voiceGain = context.createGain();
    const body = context.createBiquadFilter();
    body.type = 'peaking';
    body.frequency.value = profile.bodyFrequency;
    body.Q.value = instrument === 'dobro' ? 4.2 : 1.15;
    body.gain.value = profile.bodyGain;
    const highCut = context.createBiquadFilter();
    highCut.type = 'lowpass';
    highCut.frequency.value = instrument === 'mandolin' ? 9800 : instrument === 'dobro' ? 7200 : 7600;
    highCut.Q.value = 0.58;
    const level = clamp(Number(velocity) || 0.78, 0.03, 1) * 0.62;
    const releaseAt = startAt + requestedDuration;
    const stopAt = releaseAt + profile.release;
    voiceGain.gain.setValueAtTime(0.0001, startAt);
    voiceGain.gain.linearRampToValueAtTime(level, startAt + profile.attack);
    voiceGain.gain.exponentialRampToValueAtTime(Math.max(0.0002, level * 0.16), startAt + Math.min(profile.decay, requestedDuration));
    if (typeof voiceGain.gain.cancelAndHoldAtTime === 'function') {
      voiceGain.gain.cancelAndHoldAtTime(releaseAt);
    } else {
      voiceGain.gain.cancelScheduledValues(releaseAt);
      voiceGain.gain.setValueAtTime(Math.max(0.0002, level * 0.12), releaseAt);
    }
    voiceGain.gain.exponentialRampToValueAtTime(0.0001, stopAt);
    body.connect(highCut);
    highCut.connect(voiceGain);
    let output = voiceGain;
    let panner = null;
    if (typeof context.createStereoPanner === 'function') {
      panner = context.createStereoPanner();
      panner.pan.value = profile.pan;
      voiceGain.connect(panner);
      output = panner;
    }
    output.connect(this.input);
    const cents = profile.doubleCourseCents ? [-profile.doubleCourseCents, profile.doubleCourseCents] : [0];
    const sources = cents.map((detune) => {
      const source = context.createBufferSource();
      source.buffer = this.createPluckedBuffer(instrument, midi, requestedDuration, variant, detune);
      const sourceGain = context.createGain();
      sourceGain.gain.value = 1 / Math.sqrt(cents.length);
      source.connect(sourceGain);
      sourceGain.connect(body);
      source.start(startAt);
      source.stop(stopAt + 0.03);
      return { source, sourceGain };
    });
    const voice = {
      oscillators: sources.map((item) => item.source),
      vibrato: null,
      noiseSource: null,
      voiceGain,
      stopAt,
      nodes: [body, highCut, voiceGain, panner, ...sources.flatMap((item) => [item.source, item.sourceGain])].filter(Boolean),
    };
    this.voices.add(voice);
    let ended = 0;
    sources.forEach(({ source }) => {
      source.onended = () => {
        ended += 1;
        if (ended !== sources.length) return;
        this.voices.delete(voice);
        voice.nodes.forEach((node) => {
          try { node.disconnect(); } catch { /* already disconnected */ }
        });
      };
    });
    return voice;
  }

  playDrumAt(note, velocity = 0.82, when = null) {
    const context = this.ensure();
    const startAt = Math.max(context.currentTime, Number.isFinite(when) ? when : context.currentTime);
    const level = clamp(Number(velocity) || 0.82, 0.04, 1.1);
    const midi = typeof note === 'number' ? note : parseNote(note).midi;
    const drumPack = this.samplePacks.get(DRUM_PACK);
    if (drumPack?.state === 'ready') {
      const pad = drumPack.manifest.pads.find((item) => item.defaultMidiNote === midi)
        || drumPack.manifest.pads.find((item) => (
          midi === 51 ? item.defaultMidiNote === 49
            : midi === 43 ? item.defaultMidiNote === 41
              : midi === 45 ? item.defaultMidiNote === 48
                : false
        ));
      const midiVelocity = Math.round(level * 127);
      const choices = pad ? drumPack.manifest.zones.filter((zone) => (
        zone.slot === pad.slot
        && midiVelocity >= zone.lowVelocity
        && midiVelocity <= zone.highVelocity
      )) : [];
      const zone = choices[this.noiseCounter % Math.max(1, choices.length)];
      const buffer = zone ? drumPack.buffers.get(zone.file) : null;
      if (buffer) {
        this.noiseCounter += 1;
        const source = context.createBufferSource();
        const sampleGain = context.createGain();
        sampleGain.gain.value = level * 0.96;
        source.buffer = buffer;
        source.connect(sampleGain);
        sampleGain.connect(this.input);
        const voice = {
          oscillators: [source],
          vibrato: null,
          noiseSource: null,
          voiceGain: sampleGain,
          stopAt: startAt + buffer.duration,
        };
        this.voices.add(voice);
        source.onended = () => {
          this.voices.delete(voice);
          try { source.disconnect(); sampleGain.disconnect(); } catch { /* already disconnected */ }
        };
        source.start(startAt);
        return voice;
      }
    } else {
      this.preloadDrums();
    }
    const voiceGain = context.createGain();
    voiceGain.gain.setValueAtTime(Math.max(0.0001, level * 0.5), startAt);
    voiceGain.gain.exponentialRampToValueAtTime(0.0001, startAt + 0.55);
    voiceGain.connect(this.input);

    const voice = { oscillators: [], vibrato: null, noiseSource: null, voiceGain, stopAt: startAt + 0.6 };

    if (midi === 36) {
      const oscillator = context.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(135, startAt);
      oscillator.frequency.exponentialRampToValueAtTime(46, startAt + 0.16);
      oscillator.connect(voiceGain);
      oscillator.start(startAt);
      oscillator.stop(startAt + 0.5);
      voice.oscillators.push(oscillator);
    } else if (midi === 38) {
      const oscillator = context.createOscillator();
      oscillator.type = 'triangle';
      oscillator.frequency.setValueAtTime(190, startAt);
      oscillator.connect(voiceGain);
      oscillator.start(startAt);
      oscillator.stop(startAt + 0.22);
      voice.oscillators.push(oscillator);
      voice.noiseSource = this.createNoiseBurst(startAt, 0.24, level * 0.7, voiceGain);
    } else if ([42, 46, 49, 51].includes(midi)) {
      const highpass = context.createBiquadFilter();
      highpass.type = 'highpass';
      highpass.frequency.setValueAtTime(midi === 42 ? 6500 : 4300, startAt);
      highpass.Q.value = 0.7;
      highpass.connect(voiceGain);
      voice.noiseSource = this.createNoiseBurst(startAt, midi === 42 ? 0.12 : 0.42, level * 0.9, highpass);
    } else {
      const oscillator = context.createOscillator();
      oscillator.type = 'sine';
      const startFrequency = midi <= 43 ? 150 : midi <= 45 ? 190 : 240;
      oscillator.frequency.setValueAtTime(startFrequency, startAt);
      oscillator.frequency.exponentialRampToValueAtTime(startFrequency * 0.62, startAt + 0.24);
      oscillator.connect(voiceGain);
      oscillator.start(startAt);
      oscillator.stop(startAt + 0.45);
      voice.oscillators.push(oscillator);
    }

    this.voices.add(voice);
    window.setTimeout(() => this.voices.delete(voice), 700);
    return voice;
  }

  playAt(note, instrument, velocity = 0.78, duration = 0.65, when = null) {
    if (instrument === 'drums') return this.playDrumAt(note, velocity, when);
    if (SAMPLE_PACKS[instrument]) {
      const sampleVoice = this.playSampleAt(note, instrument, velocity, duration, when);
      if (sampleVoice) return sampleVoice;
      this.preloadInstrument(instrument);
    }
    if (PLUCKED_MODELS[instrument]) {
      return this.playPluckedModelAt(note, instrument, velocity, duration, when);
    }
    const context = this.ensure();
    const profile = PROFILES[instrument] || PROFILES.fiddle;
    const startAt = Math.max(context.currentTime, Number.isFinite(when) ? when : context.currentTime);
    const safeDuration = clamp(Number(duration) || 0.65, 0.05, 18);
    const level = clamp(Number(velocity) || 0.78, 0.03, 1.1);
    const releaseStart = startAt + safeDuration;
    const stopAt = releaseStart + profile.release + 0.08;
    const frequency = noteFrequency(note);

    const voiceGain = context.createGain();
    const filter = context.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(profile.filter, startAt);
    filter.Q.setValueAtTime(profile.resonance || 1, startAt);
    voiceGain.gain.setValueAtTime(0.0001, startAt);
    const peakLevel = Math.max(0.0002, level * 0.48);
    voiceGain.gain.exponentialRampToValueAtTime(peakLevel, startAt + profile.attack);

    const releaseLevel = profile.decay
      ? Math.max(0.00015, peakLevel * 0.28)
      : Math.max(0.0002, peakLevel * 0.68);
    if (profile.decay) {
      voiceGain.gain.exponentialRampToValueAtTime(releaseLevel, Math.min(releaseStart, startAt + profile.decay));
    } else {
      voiceGain.gain.setTargetAtTime(releaseLevel, startAt + profile.attack, 0.18);
    }

    if (typeof voiceGain.gain.cancelAndHoldAtTime === 'function') {
      voiceGain.gain.cancelAndHoldAtTime(releaseStart);
    } else {
      voiceGain.gain.cancelScheduledValues(releaseStart);
      voiceGain.gain.setValueAtTime(releaseLevel, releaseStart);
    }
    voiceGain.gain.exponentialRampToValueAtTime(0.0001, stopAt);

    filter.connect(voiceGain);
    voiceGain.connect(this.input);

    const oscillators = profile.oscillators.map((definition) => {
      const oscillator = context.createOscillator();
      const oscillatorGain = context.createGain();
      oscillator.type = definition.type;
      oscillator.frequency.setValueAtTime(frequency, startAt);
      oscillator.detune.setValueAtTime(definition.detune || 0, startAt);
      oscillatorGain.gain.value = definition.gain;
      oscillator.connect(oscillatorGain);
      oscillatorGain.connect(filter);
      oscillator.start(startAt);
      oscillator.stop(stopAt);
      return oscillator;
    });

    let vibrato = null;
    if (profile.vibratoRate && profile.vibratoDepth) {
      vibrato = context.createOscillator();
      const vibratoGain = context.createGain();
      vibrato.frequency.value = profile.vibratoRate;
      vibratoGain.gain.value = profile.vibratoDepth;
      vibrato.connect(vibratoGain);
      oscillators.forEach((oscillator) => vibratoGain.connect(oscillator.detune));
      vibrato.start(startAt + Math.min(0.18, safeDuration * 0.25));
      vibrato.stop(stopAt);
    }

    const noiseSource = this.createNoiseBurst(startAt, safeDuration, profile.noise || 0, filter);
    const voice = { oscillators, vibrato, noiseSource, voiceGain, stopAt };
    this.voices.add(voice);
    window.setTimeout(() => this.voices.delete(voice), Math.max(0, (stopAt - context.currentTime) * 1000 + 60));
    return voice;
  }

  play(note, instrument, velocity, duration) {
    return this.playAt(note, instrument, velocity, duration, this.getCurrentTime());
  }

  stopAll(releaseSeconds = 0.04) {
    if (!this.context) return;
    const now = this.context.currentTime;
    this.voices.forEach((voice) => {
      try {
        voice.voiceGain.gain.cancelScheduledValues(now);
        voice.voiceGain.gain.setValueAtTime(Math.max(0.0001, voice.voiceGain.gain.value || 0.02), now);
        voice.voiceGain.gain.exponentialRampToValueAtTime(0.0001, now + releaseSeconds);
        voice.oscillators.forEach((oscillator) => oscillator.stop(now + releaseSeconds + 0.03));
        voice.vibrato?.stop(now + releaseSeconds + 0.03);
        voice.noiseSource?.stop(now + releaseSeconds + 0.03);
      } catch {
        // The voice may already have stopped.
      }
    });
    this.voices.clear();
  }

  getDiagnostics() {
    return Object.fromEntries([...this.samplePacks.entries()].map(([pack, data]) => [
      pack,
      { state: data.state, loadedZones: data.buffers.size },
    ]));
  }
}

export const ensembleAudio = new EnsembleAudioEngine();
