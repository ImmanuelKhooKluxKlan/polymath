import { midiToNote, parseNote } from './noteMath.js';

const DEFAULT_ONSET_WINDOW_SECONDS = 0.55;

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function noteMidi(note) {
  if (Number.isFinite(Number(note?.midi))) return Math.round(Number(note.midi));
  try {
    return parseNote(note?.note || note?.name || note?.pitch).midi;
  } catch {
    return null;
  }
}

export function expectedNotesNear(song, songTime, windowSeconds = DEFAULT_ONSET_WINDOW_SECONDS) {
  const notes = Array.isArray(song?.notes) ? song.notes : [];
  const time = finite(songTime);
  const window = Math.max(0.1, finite(windowSeconds, DEFAULT_ONSET_WINDOW_SECONDS));
  return notes
    .map((note, index) => ({
      index,
      midi: noteMidi(note),
      note: note?.note || note?.name || (noteMidi(note) !== null ? midiToNote(noteMidi(note)) : ''),
      time: finite(note?.time),
      duration: Math.max(0.04, finite(note?.duration, 0.25)),
      velocity: Math.max(0, Math.min(1, finite(note?.velocity, 0.75))),
    }))
    .filter((note) => note.midi !== null && Math.abs(note.time - time) <= window)
    .sort((left, right) => Math.abs(left.time - time) - Math.abs(right.time - time));
}

function rounded(value) {
  return Math.round(value * 100) / 100;
}

function correctionText(signal) {
  if (signal.type === 'wrong-note') {
    const expected = signal.expectedNotes.length === 1
      ? signal.expectedNotes[0]
      : signal.expectedNotes.join(' or ');
    return `I heard ${signal.note}. This part expects ${expected}. Try that note again.`;
  }
  if (signal.type === 'hold-short') {
    return `${signal.note} was released ${signal.earlyBySeconds.toFixed(2)} seconds early. Hold it a little longer, then repeat it.`;
  }
  if (signal.type === 'hold-long') {
    return `${signal.note} rang ${signal.lateBySeconds.toFixed(2)} seconds too long. Release it a little sooner, then repeat it.`;
  }
  if (signal.type === 'early') {
    return `${signal.note} arrived ${signal.offsetSeconds.toFixed(2)} seconds early. Wait for the beat, then repeat it.`;
  }
  if (signal.type === 'late') {
    return `${signal.note} arrived ${signal.offsetSeconds.toFixed(2)} seconds late. Prepare the key sooner, then repeat it.`;
  }
  return `${signal.note} matched this part well.`;
}

export function evaluatePerformanceEvent(song, event, options = {}) {
  const midi = Math.round(finite(event?.midi, -1));
  if (midi < 0 || midi > 127) return null;
  const onset = finite(event?.startedSongTime);
  const observedDuration = Math.max(0.04, finite(event?.duration, 0.04));
  const nearby = expectedNotesNear(song, onset, options.onsetWindowSeconds);
  const samePitch = nearby.filter((note) => note.midi === midi);
  const note = midiToNote(midi);
  if (!samePitch.length) {
    if (!nearby.length) return null;
    const signal = {
      type: 'wrong-note',
      note,
      midi,
      expectedNotes: [...new Set(nearby.slice(0, 3).map((entry) => entry.note))],
      startedSongTime: rounded(onset),
      confidence: finite(event?.confidence, 0.5),
      source: event?.source || 'unknown',
    };
    return { ...signal, message: correctionText(signal) };
  }

  const expected = samePitch[0];
  const onsetOffset = onset - expected.time;
  const durationDelta = observedDuration - expected.duration;
  const durationTolerance = Math.max(0.1, expected.duration * 0.25);
  let type = 'correct';
  if (Math.abs(onsetOffset) > 0.2) type = onsetOffset < 0 ? 'early' : 'late';
  else if (durationDelta < -durationTolerance && expected.duration >= 0.18) type = 'hold-short';
  else if (durationDelta > durationTolerance && expected.duration >= 0.18) type = 'hold-long';

  const signal = {
    type,
    note,
    midi,
    expectedTime: rounded(expected.time),
    startedSongTime: rounded(onset),
    expectedDuration: rounded(expected.duration),
    observedDuration: rounded(observedDuration),
    offsetSeconds: rounded(Math.abs(onsetOffset)),
    earlyBySeconds: rounded(Math.max(0, -durationDelta)),
    lateBySeconds: rounded(Math.max(0, durationDelta)),
    confidence: finite(event?.confidence, 0.5),
    source: event?.source || 'unknown',
  };
  return { ...signal, message: correctionText(signal) };
}

export function selectObservedNotes(observation) {
  const visual = Array.isArray(observation?.vision?.pressed) ? observation.vision.pressed : [];
  if (observation?.vision?.calibrated) {
    return visual.slice(0, 8).map((entry) => ({
      midi: Math.round(Number(entry.midi)),
      confidence: Math.max(0, Math.min(1, Number(entry.score) || 0.55)),
      source: 'camera-key-motion',
    })).filter((entry) => Number.isFinite(entry.midi));
  }
  if (Number.isFinite(Number(observation?.hearing?.midi)) && observation?.hearing?.heard !== false) {
    return [{
      midi: Math.round(Number(observation.hearing.midi)),
      confidence: Math.max(0, Math.min(1, Number(observation.hearing.confidence) || 0.5)),
      source: 'microphone-dominant-pitch',
    }];
  }
  const heard = Array.isArray(observation?.hearing?.notes) ? observation.hearing.notes : [];
  return heard.slice(0, 1).map((entry) => {
    try {
      return {
        midi: parseNote(entry.note).midi,
        confidence: Math.max(0, Math.min(1, Number(entry.confidence) || Number(observation?.hearing?.confidence) || 0.5)),
        source: 'microphone-dominant-pitch',
      };
    } catch {
      return null;
    }
  }).filter(Boolean);
}
