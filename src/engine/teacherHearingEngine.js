const NOTE_NAMES = Object.freeze(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']);

export function midiToNoteName(midi) {
  const rounded = Math.round(Number(midi));
  if (!Number.isFinite(rounded)) return '—';
  return `${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}

export function frequencyToPitch(frequency) {
  const safeFrequency = Number(frequency);
  if (!Number.isFinite(safeFrequency) || safeFrequency <= 0) {
    return { frequency: 0, midi: null, note: '—', cents: 0 };
  }

  const exactMidi = 69 + (12 * Math.log2(safeFrequency / 440));
  const midi = Math.round(exactMidi);
  return {
    frequency: safeFrequency,
    midi,
    note: midiToNoteName(midi),
    cents: Math.round((exactMidi - midi) * 100),
  };
}

export function analyzeSpectralNotes(samples, sampleRate, {
  minimumMidi = 21,
  maximumMidi = 108,
  maximumNotes = 5,
} = {}) {
  if (!samples?.length || !Number.isFinite(sampleRate) || sampleRate <= 0) return [];
  let mean = 0;
  for (let index = 0; index < samples.length; index += 1) mean += samples[index];
  mean /= samples.length;
  const windowScale = Math.max(1, (samples.length - 1) / 2);
  const amplitudes = [];

  for (let midi = minimumMidi; midi <= maximumMidi; midi += 1) {
    const frequency = 440 * (2 ** ((midi - 69) / 12));
    const coefficient = 2 * Math.cos((2 * Math.PI * frequency) / sampleRate);
    let previous = 0;
    let previousPrevious = 0;
    for (let index = 0; index < samples.length; index += 1) {
      const window = 0.5 - (0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, samples.length - 1)));
      const current = ((samples[index] - mean) * window) + (coefficient * previous) - previousPrevious;
      previousPrevious = previous;
      previous = current;
    }
    const power = Math.max(0, (previous * previous) + (previousPrevious * previousPrevious) - (coefficient * previous * previousPrevious));
    amplitudes.push({
      midi,
      note: midiToNoteName(midi),
      frequency,
      amplitude: Math.sqrt(power) / windowScale,
    });
  }

  const strongest = Math.max(0, ...amplitudes.map((entry) => entry.amplitude));
  if (strongest < 0.006) return [];
  const sortedNoise = amplitudes.map((entry) => entry.amplitude).sort((left, right) => left - right);
  const noiseFloor = sortedNoise[Math.floor(sortedNoise.length * 0.55)] || 0;
  const threshold = Math.max(0.006, noiseFloor * 3.2, strongest * 0.105);
  const peaks = amplitudes.filter((entry, index) => {
    if (entry.amplitude < threshold) return false;
    const left = amplitudes[index - 1]?.amplitude || 0;
    const right = amplitudes[index + 1]?.amplitude || 0;
    return entry.amplitude >= left && entry.amplitude >= right;
  });

  return peaks
    .sort((left, right) => right.amplitude - left.amplitude)
    .slice(0, maximumNotes)
    .map((entry) => ({
      ...entry,
      confidence: Math.max(0, Math.min(1, entry.amplitude / strongest)),
      dbfs: 20 * Math.log10(Math.max(entry.amplitude, 0.0000158)),
    }));
}

export function analyzeAudioFrame(samples, sampleRate, {
  minimumRms = 0.008,
  minimumFrequency = 27,
  maximumFrequency = 4300,
  yinThreshold = 0.14,
} = {}) {
  if (!samples?.length || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return { heard: false, rms: 0, dbfs: -96, confidence: 0, notes: [], ...frequencyToPitch(0) };
  }

  let energy = 0;
  let mean = 0;
  for (let index = 0; index < samples.length; index += 1) mean += samples[index];
  mean /= samples.length;
  for (let index = 0; index < samples.length; index += 1) {
    const centered = samples[index] - mean;
    energy += centered * centered;
  }
  const rms = Math.sqrt(energy / samples.length);
  const dbfs = Math.max(-96, 20 * Math.log10(Math.max(rms, 0.0000158)));
  if (rms < minimumRms) {
    return { heard: false, rms, dbfs, confidence: 0, notes: [], ...frequencyToPitch(0) };
  }

  const notes = analyzeSpectralNotes(samples, sampleRate);

  const minimumTau = Math.max(2, Math.floor(sampleRate / maximumFrequency));
  const maximumTau = Math.min(
    Math.floor(sampleRate / minimumFrequency),
    Math.floor(samples.length / 2),
  );
  const stride = samples.length >= 4096 ? 2 : 1;
  const difference = new Float32Array(maximumTau + 1);

  for (let tau = minimumTau; tau <= maximumTau; tau += 1) {
    let sum = 0;
    let count = 0;
    const limit = samples.length - tau;
    for (let index = 0; index < limit; index += stride) {
      const delta = (samples[index] - mean) - (samples[index + tau] - mean);
      sum += delta * delta;
      count += 1;
    }
    difference[tau] = count ? sum / count : 0;
  }

  const cumulative = new Float32Array(maximumTau + 1);
  let runningSum = difference[minimumTau];
  cumulative[minimumTau] = 1;
  for (let tau = minimumTau + 1; tau <= maximumTau; tau += 1) {
    runningSum += difference[tau];
    cumulative[tau] = runningSum > 0
      ? (difference[tau] * (tau - minimumTau)) / runningSum
      : 1;
  }

  let selectedTau = -1;
  let bestTau = minimumTau;
  let bestValue = Number.POSITIVE_INFINITY;
  for (let tau = minimumTau + 2; tau < maximumTau - 1; tau += 1) {
    const value = cumulative[tau];
    if (value < bestValue) {
      bestValue = value;
      bestTau = tau;
    }
    if (value < yinThreshold && value <= cumulative[tau + 1]) {
      selectedTau = tau;
      break;
    }
  }

  if (selectedTau < 0) selectedTau = bestValue < 0.34 ? bestTau : -1;
  if (selectedTau < 0) {
    return { heard: true, rms, dbfs, confidence: 0, notes, ...frequencyToPitch(0) };
  }

  const previous = cumulative[selectedTau - 1] || cumulative[selectedTau];
  const current = cumulative[selectedTau];
  const next = cumulative[selectedTau + 1] || current;
  const denominator = previous - (2 * current) + next;
  const adjustment = Math.abs(denominator) > 1e-7
    ? 0.5 * (previous - next) / denominator
    : 0;
  const refinedTau = selectedTau + Math.max(-0.5, Math.min(0.5, adjustment));
  let pitch = frequencyToPitch(sampleRate / refinedTau);
  const strongestSpectralNote = notes[0];
  if (
    strongestSpectralNote
    && pitch.frequency >= 1000
    && strongestSpectralNote.midi - pitch.midi >= 11
    && strongestSpectralNote.confidence >= 0.72
  ) {
    pitch = frequencyToPitch(strongestSpectralNote.frequency);
  }

  return {
    heard: true,
    rms,
    dbfs,
    confidence: Math.max(0, Math.min(1, 1 - current)),
    notes,
    ...pitch,
  };
}

export async function createTeacherHearingMonitor(stream, onReading, {
  intervalMs = 120,
  fftSize = 4096,
} = {}) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('This browser does not provide Web Audio analysis.');
  if (!stream?.getAudioTracks?.().length) throw new Error('No microphone track is available.');

  const context = new AudioContextClass({ latencyHint: 'interactive' });
  if (context.state === 'suspended') await context.resume();
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = fftSize;
  analyser.smoothingTimeConstant = 0.08;
  source.connect(analyser);

  const samples = new Float32Array(analyser.fftSize);
  let stopped = false;
  let timer = 0;

  const read = () => {
    if (stopped) return;
    analyser.getFloatTimeDomainData(samples);
    onReading(analyzeAudioFrame(samples, context.sampleRate));
    timer = window.setTimeout(read, intervalMs);
  };
  read();

  return {
    context,
    stop: async () => {
      if (stopped) return;
      stopped = true;
      window.clearTimeout(timer);
      source.disconnect();
      analyser.disconnect();
      await context.close().catch(() => {});
    },
  };
}
