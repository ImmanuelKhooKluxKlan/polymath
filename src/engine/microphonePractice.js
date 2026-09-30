import { midiToNote, parseNote } from './noteMath.js';

const MINIMUM_PIANO_MIDI = 21;
const MAXIMUM_PIANO_MIDI = 108;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function finiteMidi(value) {
  const midi = Math.round(Number(value));
  return Number.isFinite(midi) && midi >= MINIMUM_PIANO_MIDI && midi <= MAXIMUM_PIANO_MIDI
    ? midi
    : null;
}

function eventMidi(event) {
  const direct = finiteMidi(event?.midi);
  if (direct !== null) return direct;
  try {
    return finiteMidi(parseNote(event?.note).midi);
  } catch {
    return null;
  }
}

export function buildPracticeOnsetGroups(notes = [], range = null, {
  groupWindowSeconds = 0.055,
} = {}) {
  const start = Number.isFinite(Number(range?.start)) ? Number(range.start) : 0;
  const end = Number.isFinite(Number(range?.end)) ? Number(range.end) : Number.POSITIVE_INFINITY;
  const grouped = [];
  const events = notes
    .map((event, index) => ({
      event,
      index,
      midi: eventMidi(event),
      time: Number(event?.time),
    }))
    .filter(({ midi, time }) => midi !== null && Number.isFinite(time) && time >= start - 0.01 && time < end + 0.01)
    .sort((left, right) => left.time - right.time || left.midi - right.midi);

  events.forEach(({ event, index, midi, time }) => {
    const previous = grouped[grouped.length - 1];
    const group = previous && time - previous.time <= groupWindowSeconds
      ? previous
      : {
        id: `onset-${grouped.length}-${time.toFixed(4)}`,
        time,
        eventIds: [],
        midis: [],
        notes: [],
      };

    if (group !== previous) grouped.push(group);
    group.eventIds.push(event?.id || `event-${index}`);
    if (!group.midis.includes(midi)) {
      group.midis.push(midi);
      group.notes.push(midiToNote(midi));
    }
  });

  return grouped.map((group) => ({
    ...group,
    midis: [...group.midis].sort((left, right) => left - right),
    notes: [...group.midis].sort((left, right) => left - right).map(midiToNote),
  }));
}

export function findPracticeOnsetGroup(groups = [], songTime = 0, {
  earlySeconds = 0.3,
  lateSeconds = 0.42,
  resolvedIds = new Set(),
} = {}) {
  const now = Number(songTime);
  if (!Number.isFinite(now)) return null;

  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  groups.forEach((group) => {
    if (!group || resolvedIds?.has?.(group.id)) return;
    const difference = now - Number(group.time);
    if (difference < -earlySeconds || difference > lateSeconds) return;
    const distance = Math.abs(difference);
    // At an exact midpoint, prefer the note the learner is approaching instead
    // of retrying an already missed note behind the playhead.
    if (distance < bestDistance || (distance === bestDistance && Number(group.time) > Number(best?.time))) {
      best = group;
      bestDistance = distance;
    }
  });
  return best;
}

export function microphoneReadingCandidates(reading = {}) {
  if (!reading?.heard || Number(reading.rms) < 0.0075) return [];
  const candidates = new Map();
  const dominantMidi = finiteMidi(reading.midi);
  const dominantConfidence = clamp(Number(reading.confidence) || 0, 0, 1);
  const dominantCents = Math.abs(Number(reading.cents) || 0);

  if (dominantMidi !== null && dominantConfidence >= 0.62 && dominantCents <= 55) {
    candidates.set(dominantMidi, {
      midi: dominantMidi,
      note: midiToNote(dominantMidi),
      confidence: dominantConfidence,
      source: 'dominant',
    });
  }

  (Array.isArray(reading.notes) ? reading.notes : []).forEach((candidate) => {
    const midi = finiteMidi(candidate?.midi);
    const confidence = clamp(Number(candidate?.confidence) || 0, 0, 1);
    if (midi === null || confidence < 0.24 || Number(candidate?.dbfs ?? -96) < -58) return;
    const previous = candidates.get(midi);
    if (previous?.source === 'dominant') {
      candidates.set(midi, { ...previous, confidence: Math.max(previous.confidence, confidence) });
    } else if (!previous || confidence > previous.confidence) {
      candidates.set(midi, {
        midi,
        note: midiToNote(midi),
        confidence,
        source: 'spectrum',
      });
    }
  });

  return [...candidates.values()]
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, 5);
}

export function classifyPracticeReading(reading = {}, expectedMidis = []) {
  if (!reading?.heard || Number(reading.rms) < 0.0075) {
    return { type: 'silence', observed: [], matches: [], requiredMatches: 1 };
  }

  const observed = microphoneReadingCandidates(reading);
  const expected = [...new Set(expectedMidis.map(finiteMidi).filter((midi) => midi !== null))];
  if (!expected.length || !observed.length) {
    return { type: 'uncertain', observed, matches: [], requiredMatches: 1 };
  }

  const matches = observed.filter((candidate) => expected.includes(candidate.midi));
  const dominant = observed.find((candidate) => candidate.source === 'dominant');
  const confidentlyWrong = dominant?.confidence >= 0.76
    && Number(reading.confidence) >= 0.76
    && Math.abs(Number(reading.cents) || 0) <= 42
    && Number(reading.dbfs ?? -96) >= -52;
  const dominantMatches = Boolean(dominant && expected.includes(dominant.midi));

  if (expected.length === 1) {
    if (dominantMatches) return { type: 'correct', observed, matches: [dominant], requiredMatches: 1 };
    return {
      type: confidentlyWrong ? 'wrong' : 'uncertain',
      observed,
      matches: [],
      requiredMatches: 1,
    };
  }

  if (dominantMatches) return { type: 'correct', observed, matches, requiredMatches: 1 };
  const strongSpectralMatches = matches.filter((candidate) => candidate.confidence >= 0.42);
  if (strongSpectralMatches.length >= 2) {
    return { type: 'correct', observed, matches: strongSpectralMatches, requiredMatches: 2 };
  }
  if (matches.length) return { type: 'partial', observed, matches, requiredMatches: 2 };
  return {
    type: confidentlyWrong ? 'wrong' : 'uncertain',
    observed,
    matches,
    requiredMatches: 2,
  };
}

export function createPracticeReadingStability() {
  return {
    signature: '',
    stableFrames: 0,
    quietFrames: 0,
    emitted: false,
    lastDbfs: -96,
    lowestDbfsSinceEmission: -96,
  };
}

export function advancePracticeReadingStability(previousState, reading = {}, {
  requiredStableFrames = 2,
  retriggerRiseDb = 4.5,
} = {}) {
  const previous = previousState || createPracticeReadingStability();
  const candidates = microphoneReadingCandidates(reading);
  const primary = candidates.find((candidate) => candidate.source === 'dominant') || candidates[0];
  const quiet = !reading?.heard || Number(reading.rms) < 0.0075 || !primary;

  if (quiet) {
    const quietFrames = previous.quietFrames + 1;
    return {
      state: quietFrames >= 1
        ? { ...createPracticeReadingStability(), quietFrames }
        : { ...previous, quietFrames },
      attack: null,
    };
  }

  const signature = String(primary.midi);
  const dbfs = Number.isFinite(Number(reading.dbfs)) ? Number(reading.dbfs) : -96;
  const samePitch = signature === previous.signature;
  let emitted = samePitch ? previous.emitted : false;
  let stableFrames = samePitch ? previous.stableFrames + 1 : 1;
  let lowestDbfsSinceEmission = samePitch && previous.emitted
    ? Math.min(previous.lowestDbfsSinceEmission, dbfs)
    : dbfs;

  if (
    samePitch
    && previous.emitted
    && previous.lowestDbfsSinceEmission <= previous.lastDbfs - 2.5
    && dbfs - previous.lowestDbfsSinceEmission >= retriggerRiseDb
  ) {
    emitted = false;
    stableFrames = requiredStableFrames;
    lowestDbfsSinceEmission = dbfs;
  }

  const shouldEmit = !emitted && stableFrames >= requiredStableFrames;
  const state = {
    signature,
    stableFrames,
    quietFrames: 0,
    emitted: emitted || shouldEmit,
    lastDbfs: dbfs,
    lowestDbfsSinceEmission,
  };

  return {
    state,
    attack: shouldEmit ? { candidates, reading } : null,
  };
}
