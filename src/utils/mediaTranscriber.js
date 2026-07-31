const ANALYSIS_SAMPLE_RATE = 11025;
const FRAME_SIZE = 1024;
const HOP_SIZE = 512;
const MIN_NOTE_SECONDS = 0.09;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

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

function inferKey(notes) {
  const histogram = Array(12).fill(0);
  notes.forEach((note) => {
    histogram[((note.midi % 12) + 12) % 12] += note.duration * (0.35 + note.confidence);
  });
  const total = histogram.reduce((sum, value) => sum + value, 0);
  if (!total) return null;

  const candidates = [];
  [MAJOR_PROFILE, MINOR_PROFILE].forEach((profile, modeIndex) => {
    for (let tonic = 0; tonic < 12; tonic += 1) {
      let score = 0;
      for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
        score += histogram[pitchClass] * profile[(pitchClass - tonic + 12) % 12];
      }
      candidates.push({ tonic, mode: modeIndex ? 'minor' : 'major', score });
    }
  });
  candidates.sort((first, second) => second.score - first.score);
  const best = candidates[0];
  const runnerUp = candidates[1];
  const intervals = best.mode === 'major' ? [0, 2, 4, 5, 7, 9, 11] : [0, 2, 3, 5, 7, 8, 10];
  return {
    tonic: NOTE_NAMES[best.tonic],
    tonicPitchClass: best.tonic,
    mode: best.mode,
    confidence: clamp((best.score - runnerUp.score) / Math.max(best.score, 0.001) * 5, 0.08, 0.85),
    pitchClasses: intervals.map((interval) => (best.tonic + interval) % 12),
  };
}

function nearestOctave(midi, reference) {
  const choices = [midi - 24, midi - 12, midi, midi + 12, midi + 24];
  return choices.reduce((best, candidate) => (
    Math.abs(candidate - reference) < Math.abs(best - reference) ? candidate : best
  ), midi);
}

function quantizeTime(value, grid, strength) {
  const snapped = Math.round(value / grid) * grid;
  return value + (snapped - value) * strength;
}

function cleanPitchNotes(notes, bpm, target, qualityMode) {
  const rawKey = inferKey(notes);
  if (qualityMode === 'raw') {
    return {
      notes,
      key: rawKey,
      diagnostics: { rawNotes: notes.length, rejected: 0, octaveRepairs: 0, scaleRepairs: 0, merged: 0 },
    };
  }

  const diagnostics = {
    rawNotes: notes.length,
    rejected: 0,
    octaveRepairs: 0,
    scaleRepairs: 0,
    merged: 0,
  };
  const minimumConfidence = qualityMode === 'detailed' ? 0.19 : target === 'bass' ? 0.24 : 0.29;
  const minimumDuration = qualityMode === 'detailed' ? 0.085 : 0.115;
  let filtered = notes.filter((note) => {
    const keep = note.confidence >= minimumConfidence
      && (note.duration >= minimumDuration || note.confidence >= 0.62);
    if (!keep) diagnostics.rejected += 1;
    return keep;
  }).map((note) => ({ ...note }));

  const registerCandidates = filtered
    .filter((note) => note.confidence >= 0.38)
    .map((note) => note.midi)
    .sort((first, second) => first - second);
  const registerCenter = registerCandidates[Math.floor(registerCandidates.length / 2)];
  if (Number.isFinite(registerCenter)) {
    filtered.forEach((note) => {
      const repaired = nearestOctave(note.midi, registerCenter);
      if (Math.abs(note.midi - registerCenter) > 12 && repaired !== note.midi && note.confidence < 0.68) {
        note.midi = repaired;
        note.note = midiToName(repaired);
        diagnostics.octaveRepairs += 1;
      }
    });
  }

  for (let index = 1; index < filtered.length; index += 1) {
    const previous = filtered[index - 1];
    const current = filtered[index];
    const gap = current.time - (previous.time + previous.duration);
    if (gap > 2.5) continue;
    const repaired = nearestOctave(current.midi, previous.midi);
    const originalLeap = Math.abs(current.midi - previous.midi);
    const repairedLeap = Math.abs(repaired - previous.midi);
    if (originalLeap >= 12 && repairedLeap <= 7 && current.confidence < 0.75) {
      current.midi = repaired;
      current.note = midiToName(repaired);
      diagnostics.octaveRepairs += 1;
    } else if (originalLeap >= 12 && repairedLeap <= 7 && previous.confidence < 0.55) {
      const repairedPrevious = nearestOctave(previous.midi, current.midi);
      previous.midi = repairedPrevious;
      previous.note = midiToName(repairedPrevious);
      diagnostics.octaveRepairs += 1;
    }
  }

  const key = rawKey || inferKey(filtered);
  if (key && key.confidence >= 0.12) {
    filtered.forEach((note) => {
      const pitchClass = ((note.midi % 12) + 12) % 12;
      if (key.pitchClasses.includes(pitchClass) || note.confidence >= 0.48) return;
      const down = (pitchClass + 11) % 12;
      const up = (pitchClass + 1) % 12;
      if (key.pitchClasses.includes(down)) note.midi -= 1;
      else if (key.pitchClasses.includes(up)) note.midi += 1;
      else return;
      note.note = midiToName(note.midi);
      diagnostics.scaleRepairs += 1;
    });
  }

  const merged = [];
  filtered.forEach((note) => {
    const previous = merged[merged.length - 1];
    const gap = previous ? note.time - (previous.time + previous.duration) : Infinity;
    const closePitch = previous && Math.abs(note.midi - previous.midi) <= 1;
    const isShortWobble = note.duration < 0.24 || previous?.duration < 0.24;
    if (previous && gap <= 0.13 && (note.midi === previous.midi || (closePitch && isShortWobble))) {
      const previousWeight = previous.duration * previous.confidence;
      const noteWeight = note.duration * note.confidence;
      if (noteWeight > previousWeight && note.midi !== previous.midi) {
        previous.midi = note.midi;
        previous.note = midiToName(note.midi);
      }
      previous.duration = Number((note.time + note.duration - previous.time).toFixed(3));
      previous.confidence = Number(Math.max(previous.confidence, note.confidence).toFixed(3));
      previous.velocity = Number(((previous.velocity + note.velocity) / 2).toFixed(3));
      diagnostics.merged += 1;
    } else {
      merged.push(note);
    }
  });

  const beatSeconds = 60 / Math.max(40, bpm || 120);
  const grid = beatSeconds / 4;
  const quantizeStrength = qualityMode === 'detailed' ? 0.2 : 0.38;
  merged.forEach((note) => {
    const end = note.time + note.duration;
    note.time = Number(Math.max(0, quantizeTime(note.time, grid, quantizeStrength)).toFixed(3));
    const quantizedEnd = quantizeTime(end, grid, quantizeStrength);
    note.duration = Number(Math.max(0.1, quantizedEnd - note.time).toFixed(3));
  });

  diagnostics.outputNotes = merged.length;
  return { notes: merged, key, diagnostics };
}

function buildDrumDraft(onsets, bpm, duration, qualityMode) {
  const beatSeconds = 60 / Math.max(40, bpm || 120);
  const minimumSpacing = Math.max(0.07, beatSeconds / 8);
  const filtered = onsets.filter((time, index) => !index || time - onsets[index - 1] >= minimumSpacing);
  const strength = qualityMode === 'raw' ? 0 : qualityMode === 'detailed' ? 0.35 : 0.68;
  const notes = filtered.map((time, index) => {
    const halfBeat = Math.round(time / (beatSeconds / 2));
    const isSubdivision = Math.abs(time - halfBeat * (beatSeconds / 2)) < beatSeconds * 0.16
      ? halfBeat % 2 === 1
      : index % 3 === 2;
    const beatIndex = Math.round(time / beatSeconds) % 4;
    const note = isSubdivision ? 'F#2' : beatIndex === 1 || beatIndex === 3 ? 'D2' : 'C2';
    const snappedTime = quantizeTime(time, beatSeconds / 4, strength);
    return {
      note,
      midi: note === 'C2' ? 36 : note === 'D2' ? 38 : 42,
      time: Number(snappedTime.toFixed(3)),
      duration: note === 'F#2' ? 0.1 : 0.18,
      velocity: note === 'C2' ? 0.9 : note === 'D2' ? 0.84 : 0.68,
      hand: 'right',
      confidence: 0.48,
      source: 'on-device-rhythm-analysis',
    };
  }).filter((note) => note.time <= duration);
  return notes.filter((note, index) => (
    !index || note.note !== notes[index - 1].note || note.time - notes[index - 1].time >= 0.07
  ));
}

export function transcribePcmSamples(input, inputSampleRate, options = {}) {
  const target = ['bass', 'drums'].includes(options.target) ? options.target : 'melody';
  const qualityMode = ['raw', 'detailed'].includes(options.qualityMode) ? options.qualityMode : 'clean';
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
  const loudRms = sortedRms[Math.floor(sortedRms.length * 0.9)] || medianRms;
  const sourceLoudnessDb = 20 * Math.log10(Math.max(loudRms, 1e-6));
  const sourceNoiseFloorDb = 20 * Math.log10(Math.max(noiseFloor, 1e-6));
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
    const pitch = target !== 'drums' && rms > voicedThreshold
      ? estimatePitch(mono, offset, pitchRange.minimum, pitchRange.maximum)
      : null;
    frames.push({ time, rms, pitch });
    previousRms = previousRms * 0.62 + rms * 0.38;
  }

  const tempo = estimateTempo(onsets, duration);
  const rawNotes = target === 'drums'
    ? buildDrumDraft(onsets, tempo.bpm, duration, 'raw')
    : segmentPitchFrames(frames, duration, target);
  const drumNotes = target === 'drums'
    ? buildDrumDraft(onsets, tempo.bpm, duration, qualityMode)
    : null;
  const cleaned = target === 'drums'
    ? {
      notes: drumNotes,
      key: null,
      diagnostics: {
        rawNotes: rawNotes.length,
        rejected: rawNotes.length - drumNotes.length,
        octaveRepairs: 0,
        scaleRepairs: 0,
        merged: 0,
        outputNotes: drumNotes.length,
      },
    }
    : cleanPitchNotes(rawNotes, tempo.bpm, target, qualityMode);
  const notes = cleaned.notes;
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
      version: 2,
      mode: 'on-device-monophonic-quality-v2',
      target,
      qualityMode,
      sourceFileName: options.sourceFileName || '',
      detectedOnsets: onsets.length,
      tempoConfidence: Number(tempo.confidence.toFixed(3)),
      noteConfidence: Number(averageConfidence.toFixed(3)),
      detectedKey: cleaned.key ? `${cleaned.key.tonic} ${cleaned.key.mode}` : '',
      keyConfidence: Number((cleaned.key?.confidence || 0).toFixed(3)),
      cleanup: cleaned.diagnostics,
      sourceAudio: {
        loudnessDb: Number(sourceLoudnessDb.toFixed(2)),
        noiseFloorDb: Number(sourceNoiseFloorDb.toFixed(2)),
        dynamicRangeDb: Number(Math.max(0, sourceLoudnessDb - sourceNoiseFloorDb).toFixed(2)),
        analysis: 'RMS estimate; not broadcast-standard LUFS',
      },
      limitations: [
        target === 'drums'
          ? 'Rhythm analysis estimates kick, snare, and hi-hat positions from broadband attacks.'
          : 'Analysis follows one dominant pitch line and cannot yet separate simultaneous instruments.',
        'Cleanup reduces harmonic jumps and unstable fragments but cannot reconstruct notes hidden inside a dense full-band mix.',
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
