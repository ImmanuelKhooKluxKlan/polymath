const ANALYSIS_SAMPLE_RATE = 11025;
const FRAME_SIZE = 1024;
const HOP_SIZE = 512;
const MIN_NOTE_SECONDS = 0.09;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function midiToName(midi) {
  const rounded = Math.round(midi);
  return `${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}

function mixToMono(buffer) {
  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < data.length; index += 1) {
      mono[index] += data[index] / buffer.numberOfChannels;
    }
  }
  return mono;
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const outputLength = Math.max(1, Math.floor(input.length * outputRate / inputRate));
  const output = new Float32Array(outputLength);
  const ratio = inputRate / outputRate;
  for (let index = 0; index < outputLength; index += 1) {
    const source = index * ratio;
    const left = Math.floor(source);
    const mix = source - left;
    output[index] = (input[left] || 0) * (1 - mix) + (input[left + 1] || input[left] || 0) * mix;
  }
  return output;
}

function frameRms(samples, offset) {
  let energy = 0;
  for (let index = 0; index < FRAME_SIZE; index += 2) {
    const sample = samples[offset + index] || 0;
    energy += sample * sample;
  }
  return Math.sqrt(energy / FRAME_SIZE);
}

function estimatePitch(samples, offset, minimumHz, maximumHz) {
  const minimumLag = Math.max(2, Math.floor(ANALYSIS_SAMPLE_RATE / maximumHz));
  const maximumLag = Math.min(
    FRAME_SIZE - 2,
    Math.ceil(ANALYSIS_SAMPLE_RATE / minimumHz),
  );
  let bestLag = 0;
  let bestCorrelation = 0;
  let zeroLagEnergy = 0;
  const correlations = [];

  for (let index = 0; index < FRAME_SIZE; index += 2) {
    const sample = samples[offset + index] || 0;
    zeroLagEnergy += sample * sample;
  }
  if (zeroLagEnergy < 0.00001) return null;

  for (let lag = minimumLag; lag <= maximumLag; lag += 1) {
    let correlation = 0;
    let lagEnergy = 0;
    const count = FRAME_SIZE - lag;
    for (let index = 0; index < count; index += 2) {
      const first = samples[offset + index] || 0;
      const second = samples[offset + index + lag] || 0;
      correlation += first * second;
      lagEnergy += second * second;
    }
    const normalized = correlation / Math.sqrt(Math.max(1e-12, zeroLagEnergy * lagEnergy));
    correlations[lag] = normalized;
    if (normalized > bestCorrelation) {
      bestCorrelation = normalized;
      bestLag = lag;
    }
  }

  const peakThreshold = Math.max(0.62, bestCorrelation * 0.88);
  for (let lag = minimumLag + 1; lag < maximumLag; lag += 1) {
    if (
      correlations[lag] >= peakThreshold
      && correlations[lag] >= correlations[lag - 1]
      && correlations[lag] > correlations[lag + 1]
    ) {
      bestLag = lag;
      bestCorrelation = correlations[lag];
      break;
    }
  }

  if (!bestLag || bestCorrelation < 0.52) return null;
  const frequency = ANALYSIS_SAMPLE_RATE / bestLag;
  const midi = 69 + 12 * Math.log2(frequency / 440);
  return { midi, confidence: clamp((bestCorrelation - 0.5) / 0.45, 0, 1) };
}

function estimateTempo(onsets, duration) {
  if (onsets.length < 4) return { bpm: 120, confidence: 0.2 };
  const candidates = new Map();
  for (let first = 0; first < onsets.length; first += 1) {
    for (let second = first + 1; second < Math.min(onsets.length, first + 8); second += 1) {
      const interval = onsets[second] - onsets[first];
      if (interval < 0.24 || interval > 2.1) continue;
      let bpm = 60 / interval;
      while (bpm < 70) bpm *= 2;
      while (bpm > 180) bpm /= 2;
      const rounded = Math.round(bpm);
      candidates.set(rounded, (candidates.get(rounded) || 0) + 1);
    }
  }
  const ranked = [...candidates.entries()].sort((a, b) => b[1] - a[1]);
  if (!ranked.length) return { bpm: 120, confidence: 0.2 };
  return {
    bpm: ranked[0][0],
    confidence: clamp(ranked[0][1] / Math.max(8, duration / 2), 0.25, 0.9),
  };
}

function segmentPitchFrames(frames, duration, target) {
  const notes = [];
  let active = null;

  function finish(endTime) {
    if (!active) return;
    const noteDuration = endTime - active.time;
    if (noteDuration >= MIN_NOTE_SECONDS && active.frames >= 2) {
      notes.push({
        note: midiToName(active.midiSum / active.frames),
        midi: Math.round(active.midiSum / active.frames),
        time: Number(active.time.toFixed(3)),
        duration: Number(Math.min(8, noteDuration).toFixed(3)),
        velocity: Number(clamp(0.38 + active.energySum / active.frames * 5.5, 0.35, 1).toFixed(3)),
        hand: target === 'bass' ? 'left' : 'right',
        confidence: Number(clamp(active.confidenceSum / active.frames, 0, 1).toFixed(3)),
        source: 'on-device-audio-analysis',
      });
    }
    active = null;
  }

  frames.forEach((frame) => {
    if (!frame.pitch) {
      finish(frame.time);
      return;
    }
    const midi = Math.round(frame.pitch.midi);
    if (!active || Math.abs(midi - Math.round(active.midiSum / active.frames)) > 1) {
      finish(frame.time);
      active = {
        time: frame.time,
        midiSum: frame.pitch.midi,
        confidenceSum: frame.pitch.confidence,
        energySum: frame.rms,
        frames: 1,
      };
      return;
    }
    active.midiSum += frame.pitch.midi;
    active.confidenceSum += frame.pitch.confidence;
    active.energySum += frame.rms;
    active.frames += 1;
  });
  finish(duration);
  return notes;
}

export function transcribePcmSamples(input, inputSampleRate, options = {}) {
  const target = options.target === 'bass' ? 'bass' : 'melody';
  const duration = input.length / inputSampleRate;
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('No usable audio was found.');
  if (duration > 12 * 60) throw new Error('The first version supports recordings up to 12 minutes.');

  const mono = resampleLinear(input, inputSampleRate, ANALYSIS_SAMPLE_RATE);
  const rmsValues = [];
  for (let offset = 0; offset + FRAME_SIZE < mono.length; offset += HOP_SIZE) {
    rmsValues.push(frameRms(mono, offset));
  }
  const sortedRms = [...rmsValues].sort((a, b) => a - b);
  const noiseFloor = sortedRms[Math.floor(sortedRms.length * 0.05)] || 0;
  const medianRms = sortedRms[Math.floor(sortedRms.length * 0.5)] || 0;
  const voicedThreshold = Math.max(0.008, Math.min(noiseFloor * 2.6, medianRms * 0.35));
  const pitchRange = target === 'bass'
    ? { minimum: 38, maximum: 260 }
    : { minimum: 80, maximum: 1050 };
  const frames = [];
  const onsets = [];
  let previousRms = 0;

  for (let frameIndex = 0, offset = 0; offset + FRAME_SIZE < mono.length; frameIndex += 1, offset += HOP_SIZE) {
    const rms = rmsValues[frameIndex];
    const time = offset / ANALYSIS_SAMPLE_RATE;
    if (rms > voicedThreshold && rms > previousRms * 1.42 && rms - previousRms > 0.006) {
      onsets.push(time);
    }
    const pitch = rms > voicedThreshold
      ? estimatePitch(mono, offset, pitchRange.minimum, pitchRange.maximum)
      : null;
    frames.push({ time, rms, pitch });
    previousRms = previousRms * 0.62 + rms * 0.38;
  }

  const tempo = estimateTempo(onsets, duration);
  const notes = segmentPitchFrames(frames, duration, target);
  const averageConfidence = notes.length
    ? notes.reduce((sum, note) => sum + note.confidence, 0) / notes.length
    : 0;

  if (!notes.length) {
    throw new Error('No stable notes were detected. Try a clearer recording or switch between Lead melody and Bass line.');
  }

  return {
    title: options.title?.trim() || options.sourceFileName?.replace(/\.[^.]+$/, '') || 'Audio transcription',
    composer: 'Audio transcription draft',
    bpm: tempo.bpm,
    timeSignature: [4, 4],
    notes,
    duration,
    youtubeUrl: options.youtubeUrl?.trim() || '',
    transcription: {
      version: 1,
      mode: 'on-device-monophonic-draft',
      target,
      sourceFileName: options.sourceFileName || '',
      detectedOnsets: onsets.length,
      tempoConfidence: Number(tempo.confidence.toFixed(3)),
      noteConfidence: Number(averageConfidence.toFixed(3)),
      limitations: [
        'First-version analysis follows one dominant pitch line.',
        'Full-band recordings can create missing or incorrect notes.',
        'YouTube links are stored as references and are not downloaded.',
      ],
    },
  };
}

export async function transcribeMediaFile(file, options = {}) {
  if (!file?.size) throw new Error('Choose a non-empty audio or video file.');
  const audioContext = new AudioContext();
  let decoded;
  try {
    decoded = await audioContext.decodeAudioData(await file.arrayBuffer());
  } catch {
    throw new Error('This browser could not decode the media audio. Try MP3, WAV, M4A, or an MP4 with AAC audio.');
  } finally {
    await audioContext.close().catch(() => {});
  }
  return transcribePcmSamples(
    mixToMono(decoded),
    decoded.sampleRate,
    { ...options, sourceFileName: file.name },
  );
}
