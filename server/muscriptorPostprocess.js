const fs = require('fs');

// MuScriptor can emit the same pitch several times only a few frames apart.
// A 75 ms window removes that machine-gun artifact while preserving ordinary
// repeated-note rhythms.
const DUPLICATE_ONSET_SECONDS = 0.075;
const SAME_KEY_RELEASE_GAP_SECONDS = 0.018;
const MIN_NOTE_SECONDS = 0.035;
const MAX_PIANO_HOLD_SECONDS = 8;
const VOCAL_MELODY_GAIN = 1.18;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function round(value, places = 4) {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function instrumentPriority(instrument) {
  if (instrument === 'voice') return 0;
  if (instrument === 'acoustic_piano') return 1;
  if (instrument === 'electric_piano') return 2;
  return 3;
}

function normalizeNote(note) {
  const midi = Math.round(Number(note?.midi));
  const time = Math.max(0, Number(note?.time));
  const duration = Math.max(MIN_NOTE_SECONDS, Number(note?.duration));
  if (!Number.isFinite(midi) || midi < 21 || midi > 108) return null;
  if (!Number.isFinite(time) || !Number.isFinite(duration)) return null;
  return {
    ...note,
    midi,
    time: round(time),
    duration: round(duration),
    velocity: clamp(Number(note.velocity) || 0.72, 0.05, 1),
    instrument: String(note.instrument || 'acoustic_piano'),
  };
}

function collapseDuplicateOnsets(notes) {
  const byPitch = new Map();
  notes.forEach((note) => {
    const entries = byPitch.get(note.midi) || [];
    entries.push(note);
    byPitch.set(note.midi, entries);
  });

  const collapsed = [];
  let removed = 0;

  byPitch.forEach((pitchNotes) => {
    pitchNotes.sort((a, b) => (
      a.time - b.time
      || instrumentPriority(a.instrument) - instrumentPriority(b.instrument)
      || b.duration - a.duration
    ));

    let group = [];
    const finishGroup = () => {
      if (!group.length) return;
      const firstTime = Math.min(...group.map((note) => note.time));
      const lastEnd = Math.max(...group.map((note) => note.time + note.duration));
      const preferred = [...group].sort((a, b) => (
        instrumentPriority(a.instrument) - instrumentPriority(b.instrument)
        || a.time - b.time
        || b.duration - a.duration
      ))[0];
      collapsed.push({
        ...preferred,
        time: round(firstTime),
        duration: round(Math.max(MIN_NOTE_SECONDS, lastEnd - firstTime)),
      });
      removed += group.length - 1;
      group = [];
    };

    pitchNotes.forEach((note) => {
      if (!group.length || note.time - group[0].time <= DUPLICATE_ONSET_SECONDS) {
        group.push(note);
      } else {
        finishGroup();
        group.push(note);
      }
    });
    finishGroup();
  });

  return { notes: collapsed, removed };
}

function resolveSameKeyOverlaps(notes) {
  const byPitch = new Map();
  notes.forEach((note) => {
    const entries = byPitch.get(note.midi) || [];
    entries.push(note);
    byPitch.set(note.midi, entries);
  });

  let shortened = 0;
  let capped = 0;

  byPitch.forEach((pitchNotes) => {
    pitchNotes.sort((a, b) => a.time - b.time || b.duration - a.duration);
    pitchNotes.forEach((note, index) => {
      let duration = note.duration;
      if (duration > MAX_PIANO_HOLD_SECONDS) {
        duration = MAX_PIANO_HOLD_SECONDS;
        capped += 1;
      }

      const next = pitchNotes[index + 1];
      if (next) {
        const latestEnd = next.time - SAME_KEY_RELEASE_GAP_SECONDS;
        if (note.time + duration > latestEnd) {
          duration = Math.max(MIN_NOTE_SECONDS, latestEnd - note.time);
          shortened += 1;
        }
      }
      note.duration = round(duration);
    });
  });

  return { notes, shortened, capped };
}

function readWavRmsEnvelope(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  const bytes = fs.readFileSync(filePath);
  if (bytes.length < 44 || bytes.toString('ascii', 0, 4) !== 'RIFF' || bytes.toString('ascii', 8, 12) !== 'WAVE') {
    return null;
  }

  let format = null;
  let dataOffset = 0;
  let dataLength = 0;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = bytes.toString('ascii', offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ' && length >= 16 && start + length <= bytes.length) {
      format = {
        audioFormat: bytes.readUInt16LE(start),
        channels: bytes.readUInt16LE(start + 2),
        sampleRate: bytes.readUInt32LE(start + 4),
        blockAlign: bytes.readUInt16LE(start + 12),
        bitsPerSample: bytes.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      dataOffset = start;
      dataLength = Math.min(length, bytes.length - start);
      break;
    }
    offset = start + length + (length % 2);
  }

  if (!format || !dataOffset || !dataLength || format.channels < 1 || format.sampleRate < 1) return null;
  const supportedPcm = format.audioFormat === 1 && [16, 24, 32].includes(format.bitsPerSample);
  const supportedFloat = format.audioFormat === 3 && format.bitsPerSample === 32;
  if ((!supportedPcm && !supportedFloat) || format.blockAlign < 1) return null;

  const totalFrames = Math.floor(dataLength / format.blockAlign);
  const framesPerBin = Math.max(1, Math.round(format.sampleRate * 0.02));
  const sums = new Float64Array(Math.ceil(totalFrames / framesPerBin));
  const counts = new Uint32Array(sums.length);
  const bytesPerSample = format.bitsPerSample / 8;

  function readSample(sampleOffset) {
    if (format.audioFormat === 3) return bytes.readFloatLE(sampleOffset);
    if (format.bitsPerSample === 16) return bytes.readInt16LE(sampleOffset) / 32768;
    if (format.bitsPerSample === 24) return bytes.readIntLE(sampleOffset, 3) / 8388608;
    return bytes.readInt32LE(sampleOffset) / 2147483648;
  }

  for (let frame = 0; frame < totalFrames; frame += 1) {
    const bin = Math.floor(frame / framesPerBin);
    const frameOffset = dataOffset + frame * format.blockAlign;
    let frameSquare = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const value = readSample(frameOffset + channel * bytesPerSample);
      frameSquare += value * value;
    }
    sums[bin] += frameSquare / format.channels;
    counts[bin] += 1;
  }

  return {
    frameSeconds: framesPerBin / format.sampleRate,
    levels: Array.from(sums, (sum, index) => Math.sqrt(sum / Math.max(1, counts[index]))),
  };
}

function quantile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * clamp(fraction, 0, 1))];
}

function sourceLevelAt(envelope, time) {
  if (!envelope?.levels?.length || !(envelope.frameSeconds > 0)) return 0;
  const start = Math.max(0, Math.floor((time - 0.02) / envelope.frameSeconds));
  const end = Math.min(envelope.levels.length - 1, Math.ceil((time + 0.18) / envelope.frameSeconds));
  let peakRms = 0;
  for (let index = start; index <= end; index += 1) {
    peakRms = Math.max(peakRms, Number(envelope.levels[index]) || 0);
  }
  return peakRms;
}

function applySourceDynamics(notes, envelope) {
  if (!envelope) return false;
  const decibels = notes.map((note) => 20 * Math.log10(Math.max(sourceLevelAt(envelope, note.time), 0.000001)));
  const quiet = quantile(decibels, 0.1);
  const loud = quantile(decibels, 0.9);
  if (!Number.isFinite(quiet) || !Number.isFinite(loud) || loud - quiet < 1) return false;

  notes.forEach((note, index) => {
    const normalized = clamp((decibels[index] - quiet) / (loud - quiet), 0, 1);
    note.velocity = round(0.38 + Math.sqrt(normalized) * 0.56, 3);
  });
  return true;
}

function postProcessMuscriptorResult(payload, options = {}) {
  if (!payload || !Array.isArray(payload.notes) || options.instrument !== 'piano') return payload;
  const excludeVocals = options.playbackMode === 'instrumental';
  const eligibleNotes = payload.notes.filter((note) => (
    !excludeVocals || String(note?.instrument || '').toLowerCase() !== 'voice'
  ));
  const excludedVocalNotes = payload.notes.length - eligibleNotes.length;
  const inputNotes = eligibleNotes.map(normalizeNote).filter(Boolean);
  const collapsed = collapseDuplicateOnsets(inputNotes);
  const resolved = resolveSameKeyOverlaps(collapsed.notes);
  const envelope = options.sourceEnvelope || readWavRmsEnvelope(options.preparedPath);
  const sourceDynamicsApplied = applySourceDynamics(resolved.notes, envelope);
  const notes = resolved.notes.sort((a, b) => (
    a.time - b.time || a.midi - b.midi || a.instrument.localeCompare(b.instrument)
  ));
  let vocalMelodyNotes = 0;
  notes.forEach((note) => {
    if (note.instrument !== 'voice') return;
    note.velocity = round(clamp(note.velocity * VOCAL_MELODY_GAIN, 0.05, 1), 3);
    vocalMelodyNotes += 1;
  });

  return {
    ...payload,
    notes,
    instrumentGroups: [...new Set(notes.map((note) => note.instrument))].sort(),
    performance: {
      ...(payload.performance || {}),
      profile: 'muscriptor-piano-cleanup-v2',
      preserveScoreDurations: true,
      sameKeyRetriggerGapSeconds: SAME_KEY_RELEASE_GAP_SECONDS,
      defaultAutoplayReleaseSeconds: 0.5,
      vocalMelodyRenderedOnPiano: vocalMelodyNotes > 0,
      vocalMelodyGain: VOCAL_MELODY_GAIN,
    },
    transcriptionCleanup: {
      version: 2,
      inputNotes: payload.notes.length,
      outputNotes: notes.length,
      removedDuplicateNotes: collapsed.removed,
      removedRapidRetriggers: collapsed.removed,
      excludedVocalNotes,
      vocalMelodyNotes,
      vocalMelodyGain: VOCAL_MELODY_GAIN,
      shortenedSameKeyOverlaps: resolved.shortened,
      cappedImpossibleDurations: resolved.capped,
      sourceDynamicsApplied,
      duplicateOnsetWindowMs: Math.round(DUPLICATE_ONSET_SECONDS * 1000),
      maximumPianoHoldSeconds: MAX_PIANO_HOLD_SECONDS,
    },
  };
}

module.exports = {
  applySourceDynamics,
  postProcessMuscriptorResult,
  readWavRmsEnvelope,
};
