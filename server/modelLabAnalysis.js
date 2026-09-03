const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function round(value, digits = 3) {
  const factor = 10 ** digits;
  return Math.round((Number(value) || 0) * factor) / factor;
}

function midiToNote(midi) {
  const value = Math.round(Number(midi));
  return `${PITCH_CLASSES[((value % 12) + 12) % 12]}${Math.floor(value / 12) - 1}`;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function normalizeNotes(notes) {
  return (Array.isArray(notes) ? notes : [])
    .map((note, index) => {
      const midi = Math.round(Number(note?.midi));
      const time = Number(note?.time);
      const duration = Number(note?.duration);
      if (!Number.isFinite(midi) || midi < 0 || midi > 127) return null;
      if (!Number.isFinite(time) || time < 0 || !Number.isFinite(duration) || duration <= 0) return null;
      return {
        ...note,
        index,
        midi,
        note: String(note?.note || midiToNote(midi)),
        time,
        duration,
        end: time + duration,
        velocity: clamp(Number(note?.velocity) || 0, 0, 1),
        instrument: String(note?.instrument || 'unknown').trim().toLowerCase() || 'unknown',
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.time - b.time || a.midi - b.midi || a.instrument.localeCompare(b.instrument));
}

function pitchSummary(notes) {
  if (!notes.length) {
    return {
      minimumMidi: null,
      minimumNote: null,
      maximumMidi: null,
      maximumNote: null,
      rangeSemitones: 0,
      rangeOctaves: 0,
      meanMidi: null,
      medianMidi: null,
      uniquePitches: 0,
    };
  }
  const pitches = notes.map((note) => note.midi);
  const minimumMidi = Math.min(...pitches);
  const maximumMidi = Math.max(...pitches);
  return {
    minimumMidi,
    minimumNote: midiToNote(minimumMidi),
    maximumMidi,
    maximumNote: midiToNote(maximumMidi),
    rangeSemitones: maximumMidi - minimumMidi,
    rangeOctaves: round((maximumMidi - minimumMidi) / 12, 2),
    meanMidi: round(pitches.reduce((sum, value) => sum + value, 0) / pitches.length, 2),
    medianMidi: round(median(pitches), 2),
    uniquePitches: new Set(pitches).size,
  };
}

function timingSummary(notes, recordingSeconds) {
  const durations = notes.map((note) => note.duration);
  const starts = notes.map((note) => note.time).sort((a, b) => a - b);
  const onsetClusters = [];
  for (const time of starts) {
    const previous = onsetClusters[onsetClusters.length - 1];
    if (previous && time - previous.start <= 0.03) previous.count += 1;
    else onsetClusters.push({ start: time, count: 1 });
  }

  const events = notes.flatMap((note) => [
    { time: note.time, delta: 1 },
    { time: note.end, delta: -1 },
  ]).sort((a, b) => a.time - b.time || a.delta - b.delta);
  let active = 0;
  let maximumPolyphony = 0;
  for (const event of events) {
    active += event.delta;
    maximumPolyphony = Math.max(maximumPolyphony, active);
  }

  const byVoicePitch = new Map();
  for (const note of notes) {
    const key = `${note.instrument}:${note.midi}`;
    const group = byVoicePitch.get(key) || [];
    group.push(note);
    byVoicePitch.set(key, group);
  }

  let rapidRepeats75ms = 0;
  let nearDuplicates20ms = 0;
  let samePitchOverlaps = 0;
  for (const group of byVoicePitch.values()) {
    group.sort((a, b) => a.time - b.time || b.duration - a.duration);
    for (let index = 1; index < group.length; index += 1) {
      const previous = group[index - 1];
      const current = group[index];
      const onsetGap = current.time - previous.time;
      if (onsetGap <= 0.075) rapidRepeats75ms += 1;
      if (onsetGap <= 0.02) nearDuplicates20ms += 1;
      if (current.time < previous.end) samePitchOverlaps += 1;
    }
  }

  const totalSoundingSeconds = durations.reduce((sum, value) => sum + value, 0);
  return {
    recordingSeconds: round(recordingSeconds),
    noteDensityPerSecond: round(notes.length / Math.max(recordingSeconds, 0.001)),
    totalSoundingSeconds: round(totalSoundingSeconds),
    averagePolyphony: round(totalSoundingSeconds / Math.max(recordingSeconds, 0.001), 2),
    maximumPolyphony,
    uniqueOnsetClusters: onsetClusters.length,
    chordOnsets: onsetClusters.filter((cluster) => cluster.count >= 3).length,
    largestOnsetCluster: Math.max(0, ...onsetClusters.map((cluster) => cluster.count)),
    minimumDurationSeconds: durations.length ? round(Math.min(...durations)) : 0,
    averageDurationSeconds: durations.length
      ? round(durations.reduce((sum, value) => sum + value, 0) / durations.length)
      : 0,
    medianDurationSeconds: round(median(durations)),
    maximumDurationSeconds: durations.length ? round(Math.max(...durations)) : 0,
    notesUnder100ms: durations.filter((value) => value < 0.1).length,
    notesOver2Seconds: durations.filter((value) => value > 2).length,
    rapidRepeats75ms,
    nearDuplicates20ms,
    samePitchOverlaps,
  };
}

function instrumentSummary(notes, recordingSeconds) {
  const groups = new Map();
  for (const note of notes) {
    const group = groups.get(note.instrument) || [];
    group.push(note);
    groups.set(note.instrument, group);
  }

  const totalSounding = notes.reduce((sum, note) => sum + note.duration, 0);
  return [...groups.entries()]
    .map(([instrument, group]) => {
      const soundingSeconds = group.reduce((sum, note) => sum + note.duration, 0);
      const velocities = group.map((note) => note.velocity);
      return {
        instrument,
        notes: group.length,
        noteSharePercent: round((group.length / Math.max(notes.length, 1)) * 100, 1),
        soundingSeconds: round(soundingSeconds),
        soundingSharePercent: round((soundingSeconds / Math.max(totalSounding, 0.001)) * 100, 1),
        notesPerSecond: round(group.length / Math.max(recordingSeconds, 0.001)),
        averageDurationSeconds: round(soundingSeconds / Math.max(group.length, 1)),
        averageVelocity: round(velocities.reduce((sum, value) => sum + value, 0) / Math.max(velocities.length, 1)),
        ...pitchSummary(group),
      };
    })
    .sort((a, b) => b.notes - a.notes || a.instrument.localeCompare(b.instrument));
}

function histogram(notes, selector, labels) {
  const counts = new Map(labels.map((label) => [label, 0]));
  for (const note of notes) {
    const label = selector(note);
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return [...counts.entries()].map(([label, count]) => ({
    label,
    count,
    percent: round((count / Math.max(notes.length, 1)) * 100, 1),
  }));
}

function analyzeTranscription(payload = {}) {
  const sourceNotes = Array.isArray(payload.notes) ? payload.notes : [];
  const notes = normalizeNotes(sourceNotes);
  const recordingSeconds = notes.length ? Math.max(...notes.map((note) => note.end)) : 0;
  const pitches = new Map();
  for (const note of notes) {
    const current = pitches.get(note.midi) || { midi: note.midi, note: midiToNote(note.midi), count: 0 };
    current.count += 1;
    pitches.set(note.midi, current);
  }

  const velocities = notes.map((note) => note.velocity);
  const timing = timingSummary(notes, recordingSeconds);
  const instruments = instrumentSummary(notes, recordingSeconds);
  const warnings = [];
  if (timing.rapidRepeats75ms > 0) warnings.push(`${timing.rapidRepeats75ms} same-instrument pitch repeats occur within 75 ms.`);
  if (timing.samePitchOverlaps > 0) warnings.push(`${timing.samePitchOverlaps} same-instrument pitch events overlap.`);
  if (timing.notesUnder100ms > Math.max(5, notes.length * 0.1)) warnings.push('A large share of notes are shorter than 100 ms; this can sound stuttered.');
  if (instruments.length === 1) warnings.push('Only one instrument group was detected; verify this against what you hear in the recording.');

  return {
    schema: 'polymath-model-lab-analysis-v1',
    generatedAt: new Date().toISOString(),
    model: {
      provider: String(payload.transcriptionProvider || 'Polymath'),
      sourceType: String(payload.sourceType || 'muscriptor-audio-transcription'),
      license: String(payload.modelLicense || 'CC-BY-NC-4.0'),
      rawOutput: true,
      cleanupApplied: false,
    },
    headline: {
      validNotes: notes.length,
      rejectedMalformedNotes: Math.max(0, sourceNotes.length - notes.length),
      detectedInstrumentGroups: instruments.length,
      recordingSeconds: round(recordingSeconds),
      pitchRange: notes.length
        ? `${midiToNote(Math.min(...notes.map((note) => note.midi)))}–${midiToNote(Math.max(...notes.map((note) => note.midi)))}`
        : 'None',
      maximumPolyphony: timing.maximumPolyphony,
      rapidRepeats75ms: timing.rapidRepeats75ms,
    },
    instruments,
    pitch: {
      ...pitchSummary(notes),
      pitchClasses: histogram(notes, (note) => PITCH_CLASSES[note.midi % 12], PITCH_CLASSES),
      topPitches: [...pitches.values()]
        .sort((a, b) => b.count - a.count || a.midi - b.midi)
        .slice(0, 20)
        .map((pitch) => ({
          ...pitch,
          percent: round((pitch.count / Math.max(notes.length, 1)) * 100, 1),
        })),
    },
    timing,
    midi: {
      noteOnEvents: notes.length,
      noteOffEvents: notes.length,
      totalChannelEvents: notes.length * 2,
      inferredTracks: instruments.length,
      uniqueVelocities: new Set(velocities.map((value) => round(value, 3))).size,
      minimumVelocity: velocities.length ? round(Math.min(...velocities)) : 0,
      averageVelocity: velocities.length
        ? round(velocities.reduce((sum, value) => sum + value, 0) / velocities.length)
        : 0,
      maximumVelocity: velocities.length ? round(Math.max(...velocities)) : 0,
      programsAvailable: false,
    },
    warnings,
    limitations: [
      'Instrument labels are model predictions, not verified ground truth.',
      'This Polymath worker does not expose a per-note confidence probability.',
      'The worker currently reports 120 BPM as a placeholder; tempo is not measured in this raw test.',
      'Velocity values are synthetic defaults and must not be interpreted as detected loudness.',
      'MIDI program numbers are not exposed by this checkpoint wrapper; instrument-group labels are shown instead.',
    ],
    notePreview: notes.slice(0, 500).map((note) => ({
      instrument: note.instrument,
      note: note.note,
      midi: note.midi,
      time: round(note.time, 4),
      duration: round(note.duration, 4),
      velocity: round(note.velocity, 3),
    })),
  };
}

module.exports = {
  analyzeTranscription,
  midiToNote,
  normalizeNotes,
};
