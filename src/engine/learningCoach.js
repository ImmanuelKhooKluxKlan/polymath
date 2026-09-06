import { midiToNote, parseNote } from './noteMath.js';

const ONSET_WINDOW_SECONDS = 0.035;
const MATCH_WINDOW_SECONDS = 0.42;
const PROGRESS_VERSION = 2;
const LEGACY_PROGRESS_VERSION = 1;
const MAX_LOCAL_ATTEMPT_HISTORY = 240;

export const PIANO_MASTERY_SKILLS = Object.freeze([
  { id: 'notes', metricKey: 'notes', label: 'Notes', description: 'Find the intended keys without misses or extras.' },
  { id: 'rhythm', metricKey: 'rhythm', label: 'Rhythm', description: 'Land each note at the intended moment.' },
  { id: 'holds', metricKey: 'duration', label: 'Holds', description: 'Release each key at the intended moment.' },
  { id: 'touch', metricKey: 'dynamics', label: 'Touch', description: 'Shape soft and strong notes with MIDI velocity.' },
  { id: 'pedal', metricKey: 'pedal', label: 'Pedal', description: 'Press and release sustain at the intended moments.' },
]);

const PIANO_MASTERY_SKILL_IDS = new Set(PIANO_MASTERY_SKILLS.map((skill) => skill.id));
const PIANO_HAND_MODES = new Set(['left', 'right', 'both']);

export const PIANO_LEARNING_LEVELS = Object.freeze([
  {
    id: 'melody',
    label: 'Melody first',
    shortLabel: 'Melody',
    summary: 'Recognise the song with one clear note at a time.',
    detail: 'Keeps the strongest upper melody and removes accompaniment. Best for a first play-through.',
    handMode: 'right',
    speed: 0.6,
  },
  {
    id: 'beginner',
    label: 'Easy two-hand',
    shortLabel: 'Easy',
    summary: 'Melody plus a simple bass foundation.',
    detail: 'Keeps at most one melody note and one bass note at each moment so both hands can develop without dense chords.',
    handMode: 'both',
    speed: 0.7,
  },
  {
    id: 'intermediate',
    label: 'Full shape, slower',
    shortLabel: 'Medium',
    summary: 'Preserve the musical shape with lighter chords.',
    detail: 'Keeps up to four useful notes per onset, including the outer voices, while reducing overly dense clusters.',
    handMode: 'both',
    speed: 0.85,
  },
  {
    id: 'original',
    label: 'Original arrangement',
    shortLabel: 'Original',
    summary: 'Play every available note at the recorded tempo.',
    detail: 'Preserves the complete arrangement, durations, dynamics, pedal data and hand assignments.',
    handMode: 'both',
    speed: 1,
  },
]);

export const LEARNING_SESSION_GOALS = Object.freeze([
  {
    id: 'quick',
    minutes: 5,
    label: '5-minute focus',
    summary: 'One phrase, three calm repetitions.',
    partSeconds: 8,
  },
  {
    id: 'steady',
    minutes: 10,
    label: '10-minute practice',
    summary: 'Build one section through short, accurate loops.',
    partSeconds: 12,
  },
  {
    id: 'standard',
    minutes: 15,
    label: '15-minute session',
    summary: 'Listen, practise and review one musical section.',
    partSeconds: 15,
  },
  {
    id: 'full',
    minutes: null,
    label: 'Full-song run',
    summary: 'Test your current performance from beginning to end.',
    partSeconds: 30,
  },
]);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function safeMidi(event) {
  const explicit = Number(event?.midi);
  if (Number.isFinite(explicit)) return Math.round(explicit);
  try {
    return parseNote(event?.note).midi;
  } catch {
    return null;
  }
}

function normalizedHand(event, midi) {
  const hand = String(event?.hand || '').toLowerCase();
  const role = String(
    event?.arrangementRole
      || event?.scoreRole
      || event?.learningRole
      || event?.role
      || '',
  ).toLowerCase();
  if (hand === 'left' || role.includes('left') || role.includes('bass')) return 'left';
  if (hand === 'right' || role.includes('right') || role.includes('melody') || role.includes('vocal')) return 'right';
  return midi < 60 ? 'left' : 'right';
}

function normalizeLearningNote(event, index) {
  const midi = safeMidi(event);
  const time = Number(event?.time);
  if (!Number.isFinite(midi) || !Number.isFinite(time)) return null;
  return {
    ...event,
    id: event?.id || `learn-note-${index}`,
    midi,
    time,
    duration: clamp(Number(event?.duration ?? event?.audioDuration) || 0.25, 0.035, 16),
    velocity: clamp(Number(event?.velocity) || 0.72, 0.02, 1.15),
    hand: normalizedHand(event, midi),
  };
}

function groupByOnset(notes) {
  const groups = [];
  notes.forEach((note) => {
    const latest = groups[groups.length - 1];
    if (!latest || Math.abs(note.time - latest.time) > ONSET_WINDOW_SECONDS) {
      groups.push({ time: note.time, notes: [note] });
    } else {
      latest.notes.push(note);
    }
  });
  return groups;
}

function melodyFromGroup(group) {
  const explicitlyMelodic = group.notes.filter((note) => {
    const role = String(
      note.arrangementRole
        || note.scoreRole
        || note.learningRole
        || note.role
        || '',
    ).toLowerCase();
    return role.includes('melody') || role.includes('vocal') || role.includes('top');
  });
  const rightHand = group.notes.filter((note) => note.hand === 'right');
  const pool = explicitlyMelodic.length ? explicitlyMelodic : rightHand.length ? rightHand : group.notes;
  return [...pool].sort((left, right) => right.midi - left.midi)[0];
}

function melodyCandidates(group) {
  const explicit = group.notes.filter((note) => {
    const role = String(
      note.arrangementRole
        || note.scoreRole
        || note.learningRole
        || note.role
        || '',
    ).toLowerCase();
    return role.includes('melody') || role.includes('vocal') || role.includes('top');
  });
  if (explicit.length) return explicit;
  const right = group.notes.filter((note) => note.hand === 'right');
  return right.length ? right : group.notes;
}

function melodyEvidence(note) {
  const role = String(
    note.arrangementRole
      || note.scoreRole
      || note.learningRole
      || note.role
      || '',
  ).toLowerCase();
  const explicit = role.includes('melody') || role.includes('vocal') || role.includes('top');
  const register = note.midi >= 55 && note.midi <= 88 ? 0.9 : -0.6;
  return (explicit ? 5 : 0)
    + (note.hand === 'right' ? 0.8 : 0)
    + register
    + clamp(note.velocity, 0, 1) * 0.7
    + clamp(note.duration / 1.5, 0, 1) * 0.25
    + note.midi / 240;
}

function selectMelodyPath(groups) {
  if (!groups.length) return [];
  const candidates = groups.map((group) => melodyCandidates(group));
  const scores = [];
  const previousIndexes = [];

  candidates.forEach((groupCandidates, groupIndex) => {
    const row = [];
    const backRow = [];
    groupCandidates.forEach((note) => {
      let bestScore = melodyEvidence(note);
      let bestPrevious = -1;
      if (groupIndex > 0) {
        bestScore = Number.NEGATIVE_INFINITY;
        candidates[groupIndex - 1].forEach((previous, previousIndex) => {
          const leap = Math.abs(note.midi - previous.midi);
          const continuityPenalty = Math.min(3.2, leap * 0.105);
          const directionPenalty = leap > 12 ? (leap - 12) * 0.045 : 0;
          const candidateScore = scores[groupIndex - 1][previousIndex]
            + melodyEvidence(note)
            - continuityPenalty
            - directionPenalty;
          if (candidateScore > bestScore) {
            bestScore = candidateScore;
            bestPrevious = previousIndex;
          }
        });
      }
      row.push(bestScore);
      backRow.push(bestPrevious);
    });
    scores.push(row);
    previousIndexes.push(backRow);
  });

  let candidateIndex = scores.at(-1).reduce(
    (best, score, index, row) => (score > row[best] ? index : best),
    0,
  );
  const path = Array(groups.length);
  for (let groupIndex = groups.length - 1; groupIndex >= 0; groupIndex -= 1) {
    path[groupIndex] = candidates[groupIndex][candidateIndex];
    candidateIndex = previousIndexes[groupIndex][candidateIndex];
    if (candidateIndex < 0 && groupIndex > 0) candidateIndex = 0;
  }
  return path;
}

function mergeMachineGunDuplicates(notes, minimumGap = 0.075) {
  const output = [];
  const lastIndexByMidi = new Map();
  notes.forEach((note) => {
    const previousIndex = lastIndexByMidi.get(note.midi);
    const previous = previousIndex === undefined ? null : output[previousIndex];
    if (!previous || note.time - previous.time >= minimumGap) {
      lastIndexByMidi.set(note.midi, output.length);
      output.push({ ...note });
      return;
    }
    const noteEnd = note.time + note.duration;
    previous.duration = Math.max(previous.duration, noteEnd - previous.time);
    previous.scoreDuration = previous.duration;
    previous.visualDuration = previous.duration;
    previous.audioDuration = Math.max(
      Number(previous.audioDuration) || 0,
      Math.min(previous.duration, Math.max(0.055, noteEnd - previous.time - 0.018)),
    );
    previous.velocity = Math.max(previous.velocity, note.velocity);
  });
  return output;
}

function learningNote(note, learningRole, minimumDuration, velocityFloor = null) {
  const duration = Math.max(minimumDuration, note.duration);
  const velocity = velocityFloor === null ? note.velocity : Math.max(velocityFloor, note.velocity);
  return {
    ...note,
    duration,
    scoreDuration: duration,
    visualDuration: duration,
    audioDuration: Math.min(
      duration,
      Math.max(Number(note.audioDuration) || 0, duration * 0.92),
    ),
    releaseSeconds: Math.max(Number(note.releaseSeconds) || 0, 0.52),
    velocity,
    learningRole,
  };
}

function intermediateVoicing(group, melody) {
  const ordered = [...group.notes].sort((left, right) => left.midi - right.midi);
  const bass = ordered.find((note) => note.hand === 'left') || ordered[0];
  const compulsory = [bass, melody].filter(Boolean);
  const selectedIds = new Set(compulsory.map((note) => note.id));
  const selectedPitchClasses = new Set(compulsory.map((note) => note.midi % 12));
  const optional = ordered
    .filter((note) => !selectedIds.has(note.id))
    .sort((left, right) => (
      (right.duration + right.velocity * 0.5) - (left.duration + left.velocity * 0.5)
    ));
  for (const note of optional) {
    if (compulsory.length >= 4) break;
    if (selectedPitchClasses.has(note.midi % 12)) continue;
    compulsory.push(note);
    selectedPitchClasses.add(note.midi % 12);
  }
  return [...new Map(compulsory.map((note) => [note.id, note])).values()]
    .sort((left, right) => left.midi - right.midi);
}

export function learningLevelById(levelId) {
  return PIANO_LEARNING_LEVELS.find((level) => level.id === levelId) || PIANO_LEARNING_LEVELS[1];
}

export function learningSessionById(sessionId) {
  return LEARNING_SESSION_GOALS.find((goal) => goal.id === sessionId) || LEARNING_SESSION_GOALS[1];
}

export function buildLearningArrangement(rawNotes = [], levelId = 'beginner') {
  const notes = rawNotes
    .map(normalizeLearningNote)
    .filter(Boolean)
    .sort((left, right) => left.time - right.time || left.midi - right.midi);
  if (levelId === 'original') return notes;

  const groups = groupByOnset(notes);
  const melodyPath = selectMelodyPath(groups);
  let selected;
  if (levelId === 'melody') {
    selected = groups.map((group, index) => ({
      ...learningNote(melodyPath[index] || melodyFromGroup(group), 'melody', 0.12, 0.78),
      hand: 'right',
    }));
  } else if (levelId === 'beginner') {
    selected = groups.flatMap((group, index) => {
      const left = group.notes.filter((note) => note.hand === 'left').sort((a, b) => a.midi - b.midi)[0];
      const melody = melodyPath[index] || melodyFromGroup(group);
      return [...new Map([left, melody].filter(Boolean).map((note) => [
        note.id,
        learningNote(
          note,
          note.id === melody.id ? 'melody' : 'foundation',
          0.12,
          note.id === melody.id ? 0.76 : null,
        ),
      ])).values()];
    });
  } else {
    selected = groups.flatMap((group, index) => (
      intermediateVoicing(group, melodyPath[index] || melodyFromGroup(group))
        .map((note) => learningNote(
          note,
          note.id === melodyPath[index]?.id ? 'melody' : 'harmony',
          0.09,
          note.id === melodyPath[index]?.id ? 0.74 : null,
        ))
    ));
  }

  return mergeMachineGunDuplicates(
    selected.sort((left, right) => left.time - right.time || left.midi - right.midi),
    levelId === 'melody' ? 0.09 : 0.065,
  );
}

function mean(values, fallback = 0) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : fallback;
}

function metric(score, label, detail, available = true) {
  return { score: available ? Math.round(clamp(score, 0, 100)) : null, label, detail, available };
}

function topMissedNotes(missed) {
  const counts = new Map();
  missed.forEach((note) => counts.set(note.note, (counts.get(note.note) || 0) + 1));
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3)
    .map(([note, count]) => ({ note, count }));
}

function evidenceNote(event) {
  const label = String(event?.note || '').trim();
  if (/^[A-G](?:#|b)?-?\d+$/.test(label)) return label;
  const midi = safeMidi(event);
  return Number.isFinite(midi) ? midiToNote(midi) : null;
}

function measuredPerformanceEvidence({ matches, missed, extras, timingBiasSeconds, dynamicMatches, pedal }) {
  const timing = matches
    .map(({ expected, timingError }) => ({
      note: evidenceNote(expected),
      errorMs: Math.round(timingError * 1000),
      direction: Math.abs(timingError) < 0.035 ? 'centred' : timingError < 0 ? 'early' : 'late',
    }))
    .filter((item) => item.note)
    .sort((left, right) => Math.abs(right.errorMs) - Math.abs(left.errorMs))
    .slice(0, 6);
  const holds = matches
    .map(({ expected, played }) => ({
      note: evidenceNote(expected),
      targetMs: Math.round(Math.max(0.035, Number(expected.duration) || 0.12) * 1000),
      actualMs: Math.round(Math.max(0.035, Number(played.duration) || 0.12) * 1000),
    }))
    .filter((item) => item.note)
    .map((item) => ({
      ...item,
      differenceMs: item.actualMs - item.targetMs,
      direction: item.actualMs < item.targetMs ? 'short' : 'long',
    }))
    .sort((left, right) => {
      const leftRatio = Math.abs(Math.log(Math.max(1, left.actualMs) / Math.max(1, left.targetMs)));
      const rightRatio = Math.abs(Math.log(Math.max(1, right.actualMs) / Math.max(1, right.targetMs)));
      return rightRatio - leftRatio;
    })
    .slice(0, 6);
  const targetVelocity = dynamicMatches.map(({ expected }) => Number(expected.velocity)).filter(Number.isFinite);
  const playedVelocity = dynamicMatches.map(({ played }) => Number(played.velocity)).filter(Number.isFinite);
  const velocityPercent = (values) => Math.round(mean(values) * 100);
  return {
    schema: 'polymath-practice-evidence-v1',
    notes: {
      missed: topMissedNotes(missed),
      extras: extras
        .map((event) => evidenceNote(event))
        .filter(Boolean)
        .slice(0, 8),
    },
    timing: {
      meanBiasMs: Math.round(timingBiasSeconds * 1000),
      worst: timing,
    },
    holds: { worst: holds },
    dynamics: {
      available: dynamicMatches.length > 0,
      targetAveragePercent: targetVelocity.length ? velocityPercent(targetVelocity) : null,
      playedAveragePercent: playedVelocity.length ? velocityPercent(playedVelocity) : null,
      measuredNotes: dynamicMatches.length,
    },
    pedal: pedal?.evidence || { expectedCount: 0, matchedCount: 0, events: [] },
  };
}

function matchAttempt(expected, played) {
  const used = new Set();
  const matches = [];
  const missed = [];
  expected.forEach((target) => {
    let bestIndex = -1;
    let bestDistance = Infinity;
    played.forEach((candidate, index) => {
      if (used.has(index) || candidate.midi !== target.midi) return;
      const distance = Math.abs(candidate.start - target.time);
      if (distance <= MATCH_WINDOW_SECONDS && distance < bestDistance) {
        bestIndex = index;
        bestDistance = distance;
      }
    });
    if (bestIndex < 0) {
      missed.push(target);
      return;
    }
    used.add(bestIndex);
    matches.push({ expected: target, played: played[bestIndex], timingError: played[bestIndex].start - target.time });
  });
  return { matches, missed, extras: played.filter((_, index) => !used.has(index)) };
}

function pedalAccuracy(expectedPedals, playedPedals, range) {
  const orderedPedals = [...(expectedPedals || [])]
    .filter((event) => Number.isFinite(Number(event?.time)))
    .sort((left, right) => Number(left.time) - Number(right.time));
  const stateAtStart = orderedPedals
    .filter((event) => Number(event.time) <= range.start)
    .reduce((state, event) => Boolean(event.down), false);
  const expected = [
    ...(stateAtStart ? [{ time: range.start, down: true, carriedIntoRange: true }] : []),
    ...orderedPedals.filter((event) => Number(event.time) > range.start && Number(event.time) <= range.end),
  ];
  if (!expected.length) return {
    ...metric(0, 'Pedal', 'No pedal changes are required in this section.', false),
    evidence: { expectedCount: 0, matchedCount: 0, events: [] },
  };
  if (!playedPedals?.length) return {
    ...metric(0, 'Pedal', 'No pedal input was detected. Use Space, the pedal control, or MIDI CC64.'),
    evidence: {
      expectedCount: expected.length,
      matchedCount: 0,
      events: expected.slice(0, 6).map((event) => ({
        action: event.down ? 'down' : 'up',
        targetTimeMs: Math.round(Number(event.time) * 1000),
        actualTimeMs: null,
        errorMs: null,
      })),
    },
  };
  const used = new Set();
  const evidenceEvents = [];
  const scores = expected.map((target) => {
    let bestIndex = -1;
    let bestDistance = Infinity;
    playedPedals.forEach((candidate, index) => {
      if (used.has(index) || Boolean(candidate.down) !== Boolean(target.down)) return;
      const distance = Math.abs(candidate.time - target.time);
      if (distance < bestDistance) {
        bestIndex = index;
        bestDistance = distance;
      }
    });
    if (bestIndex < 0) {
      evidenceEvents.push({
        action: target.down ? 'down' : 'up',
        targetTimeMs: Math.round(Number(target.time) * 1000),
        actualTimeMs: null,
        errorMs: null,
      });
      return 0;
    }
    used.add(bestIndex);
    evidenceEvents.push({
      action: target.down ? 'down' : 'up',
      targetTimeMs: Math.round(Number(target.time) * 1000),
      actualTimeMs: Math.round(Number(playedPedals[bestIndex].time) * 1000),
      errorMs: Math.round((Number(playedPedals[bestIndex].time) - Number(target.time)) * 1000),
    });
    return clamp(1 - bestDistance / 0.7, 0, 1);
  });
  return {
    ...metric(mean(scores) * 100, 'Pedal', `${used.size} of ${expected.length} pedal changes matched.`),
    evidence: {
      expectedCount: expected.length,
      matchedCount: used.size,
      events: evidenceEvents.slice(0, 6),
    },
  };
}

export function analyzePracticeAttempt({
  expectedNotes = [],
  playedNotes = [],
  expectedPedals = [],
  playedPedals = [],
  range,
  levelId = 'beginner',
} = {}) {
  const safeRange = {
    start: Math.max(0, Number(range?.start) || 0),
    end: Math.max(Number(range?.end) || 0, Number(range?.start) || 0),
  };
  const expected = expectedNotes
    .map(normalizeLearningNote)
    .filter((note) => note && note.time >= safeRange.start - 0.01 && note.time < safeRange.end + 0.01);
  const played = playedNotes
    .map((note, index) => ({
      ...note,
      id: note.id || `played-${index}`,
      midi: safeMidi(note),
      start: Number(note.start),
      duration: Math.max(0.035, Number(note.duration) || 0.12),
    }))
    .filter((note) => Number.isFinite(note.midi) && Number.isFinite(note.start)
      && note.start >= safeRange.start - MATCH_WINDOW_SECONDS
      && note.start <= safeRange.end + MATCH_WINDOW_SECONDS);
  const { matches, missed, extras } = matchAttempt(expected, played);
  const noteScore = expected.length
    ? clamp(((matches.length - extras.length * 0.3) / expected.length) * 100, 0, 100)
    : 0;
  const rhythmSamples = matches.map(({ timingError }) => clamp(1 - Math.abs(timingError) / MATCH_WINDOW_SECONDS, 0, 1));
  const durationSamples = matches.map(({ expected: target, played: actual }) => {
    const ratio = Math.max(0.05, actual.duration) / Math.max(0.05, target.duration);
    return clamp(1 - Math.abs(Math.log(ratio)) / Math.log(4), 0, 1);
  });
  const dynamicMatches = matches.filter(({ played: actual }) => actual.dynamicCapable);
  const dynamicSamples = dynamicMatches.map(({ expected: target, played: actual }) => (
    clamp(1 - Math.abs(actual.velocity - target.velocity) / 0.55, 0, 1)
  ));
  const timingBiasSeconds = mean(matches.map(({ timingError }) => timingError));
  const pedal = pedalAccuracy(expectedPedals, playedPedals, safeRange);
  const evidence = measuredPerformanceEvidence({
    matches,
    missed,
    extras,
    timingBiasSeconds,
    dynamicMatches,
    pedal,
  });
  const metrics = {
    notes: metric(noteScore, 'Notes', `${matches.length} matched · ${missed.length} missed · ${extras.length} extra`),
    rhythm: metric(mean(rhythmSamples) * 100, 'Rhythm', matches.length
      ? `Average timing difference ${Math.round(mean(matches.map(({ timingError }) => Math.abs(timingError))) * 1000)} ms.`
      : 'Play the highlighted notes to measure rhythm.'),
    duration: metric(mean(durationSamples) * 100, 'Holds', matches.length
      ? 'Compares when each key was released with the intended note length.'
      : 'Key releases were not available to compare.'),
    dynamics: metric(mean(dynamicSamples) * 100, 'Touch', dynamicMatches.length
      ? `Measured from ${dynamicMatches.length} velocity-sensitive MIDI notes.`
      : 'Connect a velocity-sensitive MIDI keyboard to measure soft and strong playing.', dynamicMatches.length > 0),
    pedal,
  };
  const scored = Object.values(metrics).filter((item) => item.available);
  const weights = { Notes: 0.44, Rhythm: 0.3, Holds: 0.18, Touch: 0.05, Pedal: 0.03 };
  const totalWeight = scored.reduce((sum, item) => sum + (weights[item.label] || 0), 0) || 1;
  const overall = Math.round(scored.reduce((sum, item) => sum + item.score * (weights[item.label] || 0), 0) / totalWeight);
  const availableMetrics = Object.values(metrics).filter((item) => item.available);
  const weakestByScore = [...availableMetrics].sort((left, right) => left.score - right.score)[0] || metrics.notes;
  const weakest = missed.length > Math.max(1, matches.length * 0.45) || noteScore < 55
    ? metrics.notes
    : weakestByScore;
  const strongest = [...availableMetrics].sort((left, right) => right.score - left.score)[0] || metrics.notes;
  const timingDirection = Math.abs(timingBiasSeconds) < 0.045 ? 'centred' : timingBiasSeconds < 0 ? 'early' : 'late';
  const missedNotes = topMissedNotes(missed);

  let headline = 'Start with one calm repetition';
  if (overall >= 90) headline = 'Performance-ready section';
  else if (overall >= 75) headline = 'Musical shape is taking form';
  else if (overall >= 55) headline = 'Good foundation—refine one detail';
  else if (matches.length) headline = 'The pattern is beginning to connect';

  let nextAction = 'Listen once, then play only the first few notes.';
  if (weakest.label === 'Notes' && missedNotes.length) nextAction = `Slow down and watch for ${missedNotes.map((item) => item.note).join(', ')}.`;
  else if (weakest.label === 'Rhythm') nextAction = `Repeat at a slower speed; your matched notes tended to land ${timingDirection}.`;
  else if (weakest.label === 'Holds') nextAction = 'Keep each key down until its falling bar fully reaches the line.';
  else if (weakest.label === 'Pedal') nextAction = 'Practise the pedal alone once, then add the notes.';
  else if (overall >= 90) nextAction = 'Move to the next part or raise the tempo slightly.';

  return {
    version: 1,
    createdAt: new Date().toISOString(),
    levelId,
    range: safeRange,
    score: overall,
    headline,
    nextAction,
    timingDirection,
    timingBiasMs: Math.round(timingBiasSeconds * 1000),
    expectedCount: expected.length,
    playedCount: played.length,
    matchedCount: matches.length,
    missedCount: missed.length,
    extraCount: extras.length,
    missedNotes,
    evidence,
    metrics,
    strongest: strongest.label,
    focus: weakest.label,
  };
}

export function emptyLearningProgress() {
  return {
    version: PROGRESS_VERSION,
    totalAttempts: 0,
    practiceSeconds: 0,
    songs: {},
    history: [],
  };
}

export function recommendedLearningLevel(songProgress) {
  if (!songProgress?.attempts) return 'beginner';
  const ordered = PIANO_LEARNING_LEVELS.map((level) => level.id);
  const current = ordered.includes(songProgress.lastLevelId) ? songProgress.lastLevelId : 'beginner';
  const index = ordered.indexOf(current);
  const score = Number(songProgress.lastScore || 0);
  if (score >= 88 && Number(songProgress.attempts) >= 2) {
    return ordered[Math.min(ordered.length - 1, index + 1)];
  }
  if (score < 55) return ordered[Math.max(0, index - 1)];
  return current;
}

export function readLearningProgress(storage, learnerId = 'guest') {
  try {
    const current = JSON.parse(storage?.getItem?.(`polymath-learning-progress-v${PROGRESS_VERSION}:${learnerId}`) || 'null');
    if (current?.version === PROGRESS_VERSION && typeof current.songs === 'object') {
      return normalizeLearningProgress(current);
    }
    const legacy = JSON.parse(storage?.getItem?.(`polymath-learning-progress-v${LEGACY_PROGRESS_VERSION}:${learnerId}`) || 'null');
    if (legacy?.version === LEGACY_PROGRESS_VERSION && typeof legacy.songs === 'object') {
      return normalizeLearningProgress({ ...legacy, version: PROGRESS_VERSION, history: [] });
    }
    return emptyLearningProgress();
  } catch {
    return emptyLearningProgress();
  }
}

export function recordLearningAttempt(progress, songId, report) {
  const current = normalizeLearningProgress(progress);
  const snapshot = learningAttemptSnapshot(songId, report);
  if (current.history.some((attempt) => attempt.id === snapshot.id)) return current;
  const previousSong = current.songs?.[snapshot.songId] || { attempts: 0, bestScore: 0, sections: {} };
  const sectionKey = snapshot.sectionKey;
  const previousSection = previousSong.sections?.[sectionKey] || { attempts: 0, bestScore: 0 };
  const skillScores = { ...(previousSection.skillScores || {}) };
  Object.entries(snapshot.metrics).forEach(([metricKey, item]) => {
    if (!item.available || !Number.isFinite(item.score)) return;
    const previousSkill = skillScores[metricKey] || { attempts: 0, bestScore: 0 };
    skillScores[metricKey] = {
      attempts: Number(previousSkill.attempts || 0) + 1,
      bestScore: Math.max(Number(previousSkill.bestScore || 0), item.score),
      lastScore: item.score,
    };
  });
  return {
    ...current,
    totalAttempts: Number(current.totalAttempts || 0) + 1,
    practiceSeconds: Number(current.practiceSeconds || 0) + snapshot.elapsedSeconds,
    history: [...current.history, snapshot].slice(-MAX_LOCAL_ATTEMPT_HISTORY),
    songs: {
      ...current.songs,
      [snapshot.songId]: {
        ...previousSong,
        attempts: Number(previousSong.attempts || 0) + 1,
        bestScore: Math.max(Number(previousSong.bestScore || 0), snapshot.score),
        lastScore: snapshot.score,
        lastLevelId: snapshot.levelId,
        lastPracticedAt: snapshot.createdAt,
        sections: {
          ...previousSong.sections,
          [sectionKey]: {
            attempts: Number(previousSection.attempts || 0) + 1,
            bestScore: Math.max(Number(previousSection.bestScore || 0), snapshot.score),
            lastScore: snapshot.score,
            lastPracticedAt: snapshot.createdAt,
            range: snapshot.range,
            levelId: snapshot.levelId,
            skillScores,
          },
        },
      },
    },
  };
}

export function writeLearningProgress(storage, learnerId, progress) {
  try {
    storage?.setItem?.(
      `polymath-learning-progress-v${PROGRESS_VERSION}:${learnerId}`,
      JSON.stringify(normalizeLearningProgress(progress)),
    );
    return true;
  } catch {
    return false;
  }
}

function finiteScore(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(clamp(number, 0, 100)) : null;
}

function validDateValue(value) {
  const timestamp = new Date(value || '').getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function normalizedRange(range) {
  const start = Math.max(0, Number(range?.start) || 0);
  const end = Math.max(start, Number(range?.end) || start);
  return { start, end };
}

function normalizeSnapshotMetric(metricKey, item) {
  const score = finiteScore(item?.score);
  return {
    label: PIANO_MASTERY_SKILLS.find((skill) => skill.metricKey === metricKey)?.label || String(item?.label || metricKey),
    score,
    available: item?.available !== false && score !== null,
  };
}

function learningAttemptSnapshot(songId, report = {}) {
  const range = normalizedRange(report.range);
  const createdAt = String(report.createdAt || new Date().toISOString());
  const safeSongId = String(songId || 'unknown-song').slice(0, 180);
  const levelId = learningLevelById(report.levelId).id;
  const metrics = Object.fromEntries(PIANO_MASTERY_SKILLS.map((skill) => [
    skill.metricKey,
    normalizeSnapshotMetric(skill.metricKey, report.metrics?.[skill.metricKey]),
  ]));
  const fallbackIdentity = [safeSongId, createdAt, levelId, range.start.toFixed(3), range.end.toFixed(3), finiteScore(report.score) ?? 0].join(':');
  return {
    id: String(report.attemptId || report.id || fallbackIdentity).slice(0, 220),
    songId: safeSongId,
    createdAt,
    levelId,
    range,
    sectionKey: `${levelId}:${range.start.toFixed(2)}-${range.end.toFixed(2)}`,
    elapsedSeconds: Math.max(0, range.end - range.start),
    score: finiteScore(report.score) ?? 0,
    metrics,
    focus: String(report.focus || '').slice(0, 40),
    strongest: String(report.strongest || '').slice(0, 40),
    timingDirection: ['early', 'late', 'centred'].includes(report.timingDirection) ? report.timingDirection : 'centred',
    timingBiasMs: Math.round(clamp(Number(report.timingBiasMs) || 0, -5000, 5000)),
    speedPercent: report.speedPercent !== null
      && report.speedPercent !== undefined
      && report.speedPercent !== ''
      && Number.isFinite(Number(report.speedPercent))
      ? Math.round(clamp(Number(report.speedPercent), 20, 200))
      : null,
    handMode: PIANO_HAND_MODES.has(report.handMode) ? report.handMode : 'both',
    practiceFocusId: PIANO_MASTERY_SKILL_IDS.has(report.practiceFocusId) ? report.practiceFocusId : '',
    practicePlanSource: ['baseline', 'measured'].includes(report.practicePlanSource)
      ? report.practicePlanSource
      : '',
    practiceTargetScore: finiteScore(report.practiceTargetScore),
    expectedCount: Math.max(0, Math.round(Number(report.expectedCount) || 0)),
    matchedCount: Math.max(0, Math.round(Number(report.matchedCount) || 0)),
    missedCount: Math.max(0, Math.round(Number(report.missedCount) || 0)),
    extraCount: Math.max(0, Math.round(Number(report.extraCount) || 0)),
    missedNotes: Array.isArray(report.missedNotes)
      ? report.missedNotes.slice(0, 5).map((item) => ({
        note: String(item?.note || '').slice(0, 12),
        count: Math.max(1, Math.round(Number(item?.count) || 1)),
      })).filter((item) => item.note)
      : [],
  };
}

function normalizeLearningProgress(progress) {
  if (!progress || typeof progress !== 'object') return emptyLearningProgress();
  const songs = progress.songs && typeof progress.songs === 'object' ? progress.songs : {};
  const history = Array.isArray(progress.history)
    ? progress.history.map((attempt) => learningAttemptSnapshot(attempt.songId, attempt)).slice(-MAX_LOCAL_ATTEMPT_HISTORY)
    : [];
  return {
    version: PROGRESS_VERSION,
    totalAttempts: Math.max(0, Math.round(Number(progress.totalAttempts) || 0)),
    practiceSeconds: Math.max(0, Number(progress.practiceSeconds) || 0),
    songs,
    history,
  };
}

function skillObservationWeight(attempt, skill) {
  if (skill.metricKey === 'notes') return clamp(0.65 + Math.sqrt(attempt.expectedCount || 0) / 8, 0.65, 1.6);
  if (['rhythm', 'duration'].includes(skill.metricKey)) return clamp(0.6 + Math.sqrt(attempt.matchedCount || 0) / 8, 0.6, 1.45);
  return 1;
}

function weightedMean(observations) {
  const totalWeight = observations.reduce((sum, item) => sum + item.weight, 0);
  if (!totalWeight) return null;
  return observations.reduce((sum, item) => sum + item.score * item.weight, 0) / totalWeight;
}

export function learningMasteryProfile(progress, songId) {
  const current = normalizeLearningProgress(progress);
  const safeSongId = String(songId || 'unknown-song');
  const attempts = current.history
    .filter((attempt) => attempt.songId === safeSongId)
    .sort((left, right) => validDateValue(left.createdAt) - validDateValue(right.createdAt));
  const skills = PIANO_MASTERY_SKILLS.map((skill) => {
    const observations = attempts
      .map((attempt, index) => {
        const item = attempt.metrics?.[skill.metricKey];
        if (!item?.available || !Number.isFinite(item.score)) return null;
        const recency = 1 / (1 + Math.max(0, attempts.length - index - 1) * 0.16);
        return { score: item.score, weight: recency * skillObservationWeight(attempt, skill) };
      })
      .filter(Boolean);
    const score = weightedMean(observations);
    const recent = observations.slice(-3).map((item) => item.score);
    const earlier = observations.slice(-6, -3).map((item) => item.score);
    const delta = earlier.length && recent.length ? mean(recent) - mean(earlier) : null;
    return {
      ...skill,
      score: score === null ? null : Math.round(score),
      observations: observations.length,
      confidence: Math.round((1 - Math.exp(-observations.reduce((sum, item) => sum + item.weight, 0) / 4)) * 100),
      trend: delta === null ? 'learning' : delta >= 4 ? 'improving' : delta <= -4 ? 'slipping' : 'steady',
      trendDelta: delta === null ? null : Math.round(delta),
      status: score === null ? 'unmeasured' : score >= 88 ? 'strong' : score >= 70 ? 'building' : 'focus',
    };
  });
  const measured = skills.filter((skill) => skill.score !== null);
  return {
    version: 1,
    songId: safeSongId,
    attempts: Number(current.songs?.[safeSongId]?.attempts || attempts.length),
    lastPracticedAt: current.songs?.[safeSongId]?.lastPracticedAt || attempts.at(-1)?.createdAt || null,
    overall: measured.length ? Math.round(mean(measured.map((skill) => skill.score))) : null,
    measuredSkillCount: measured.length,
    skills,
  };
}

const ADAPTIVE_EXERCISES = Object.freeze({
  notes: {
    title: 'Lock in the notes',
    instruction: 'Hear the phrase once. Then play four notes at a time, slowly, without adding extra keys.',
    successRule: 'Reach 85% note accuracy twice before increasing speed.',
    speedPercent: 60,
    targetScore: 85,
    requiredPasses: 2,
  },
  rhythm: {
    title: 'Place each note in time',
    instruction: 'Listen once, count the pulse aloud, then copy the phrase at a slower speed.',
    successRule: 'Reach 82% rhythm twice with your timing centred.',
    speedPercent: 65,
    targetScore: 82,
    requiredPasses: 2,
  },
  holds: {
    title: 'Let the notes breathe',
    instruction: 'Watch each falling bar to its end. Keep the key down until the bar clears the strike line.',
    successRule: 'Reach 82% holds twice before increasing speed.',
    speedPercent: 65,
    targetScore: 82,
    requiredPasses: 2,
  },
  touch: {
    title: 'Shape the melody',
    instruction: 'Use the MIDI keyboard and play the melody slightly stronger than its accompaniment.',
    successRule: 'Reach 80% touch twice while keeping note accuracy above 80%.',
    speedPercent: 75,
    targetScore: 80,
    requiredPasses: 2,
  },
  pedal: {
    title: 'Clean the pedal changes',
    instruction: 'Practise only the pedal changes once. Then add the notes without changing the pedal timing.',
    successRule: 'Reach 85% pedal timing twice without lowering note accuracy.',
    speedPercent: 70,
    targetScore: 85,
    requiredPasses: 2,
  },
});

function rangesMatch(left, right, tolerance = 0.12) {
  const first = normalizedRange(left);
  const second = normalizedRange(right);
  return Math.abs(first.start - second.start) <= tolerance
    && Math.abs(first.end - second.end) <= tolerance;
}

function attemptSkillScore(attempt, skill) {
  const item = attempt?.metrics?.[skill.metricKey];
  return item?.available && Number.isFinite(item.score) ? item.score : null;
}

function roundedPracticeSpeed(attempt, fallback) {
  const value = Number(attempt?.speedPercent);
  return Math.round(clamp(Number.isFinite(value) ? value : fallback, 20, 100) / 5) * 5;
}

function consecutivePassingAttempts(attempts, skill, targetScore) {
  let passes = 0;
  for (let index = attempts.length - 1; index >= 0; index -= 1) {
    const score = attemptSkillScore(attempts[index], skill);
    if (score === null || score < targetScore) break;
    passes += 1;
  }
  return passes;
}

function adaptiveGoalForSection({ progress, songId, skill, exercise, levelId, range }) {
  const baseSpeed = Math.round(exercise.speedPercent / 5) * 5;
  const comparable = progress.history
    .filter((attempt) => (
      attempt.songId === String(songId)
      && attempt.levelId === levelId
      && rangesMatch(attempt.range, range)
      && attemptSkillScore(attempt, skill) !== null
    ))
    .sort((left, right) => validDateValue(left.createdAt) - validDateValue(right.createdAt));
  const tempoComparable = comparable.filter(
    (attempt) => attempt.speedPercent !== null && attempt.speedPercent !== undefined,
  );
  const speeds = [...new Set(tempoComparable.map((attempt) => roundedPracticeSpeed(attempt, baseSpeed)))];
  const completedSpeeds = speeds.filter((speed) => {
    const atSpeed = tempoComparable.filter((attempt) => roundedPracticeSpeed(attempt, baseSpeed) === speed);
    return consecutivePassingAttempts(atSpeed, skill, exercise.targetScore) >= exercise.requiredPasses;
  });
  const highestCompletedSpeed = completedSpeeds.length ? Math.max(...completedSpeeds) : null;
  const speedPercent = highestCompletedSpeed === null
    ? baseSpeed
    : Math.min(100, Math.max(baseSpeed, highestCompletedSpeed + (highestCompletedSpeed < 80 ? 10 : 5)));
  const attemptsAtSpeed = tempoComparable.filter(
    (attempt) => roundedPracticeSpeed(attempt, baseSpeed) === speedPercent,
  );
  const passes = Math.min(
    exercise.requiredPasses,
    consecutivePassingAttempts(attemptsAtSpeed, skill, exercise.targetScore),
  );
  const latestScore = attemptsAtSpeed.length
    ? attemptSkillScore(attemptsAtSpeed.at(-1), skill)
    : null;
  return {
    metricKey: skill.metricKey,
    targetScore: exercise.targetScore,
    requiredPasses: exercise.requiredPasses,
    passes,
    remainingPasses: Math.max(0, exercise.requiredPasses - passes),
    speedPercent,
    latestScore,
    highestCompletedSpeed,
  };
}

function weakestSectionIndex(songProgress, sections, metricKey) {
  if (!Array.isArray(sections) || !sections.length) return 0;
  const candidates = Object.values(songProgress?.sections || {})
    .map((section) => ({
      range: normalizedRange(section.range),
      score: finiteScore(section.skillScores?.[metricKey]?.lastScore) ?? finiteScore(section.lastScore),
    }))
    .filter((section) => section.score !== null);
  if (!candidates.length) return 0;
  const weakest = [...candidates].sort((left, right) => left.score - right.score)[0];
  return sections.reduce((bestIndex, section, index) => {
    const best = sections[bestIndex];
    const distance = Math.abs(Number(section.start) - weakest.range.start);
    const bestDistance = Math.abs(Number(best.start) - weakest.range.start);
    return distance < bestDistance ? index : bestIndex;
  }, 0);
}

export function buildAdaptivePracticePlan({ progress, songId, sections = [], levelId = 'beginner' } = {}) {
  const current = normalizeLearningProgress(progress);
  const mastery = learningMasteryProfile(current, songId);
  const measured = mastery.skills.filter((skill) => skill.score !== null && skill.observations > 0);
  const focusSkill = [...measured].sort((left, right) => {
    const leftAdjusted = left.score + (100 - left.confidence) * 0.08;
    const rightAdjusted = right.score + (100 - right.confidence) * 0.08;
    return leftAdjusted - rightAdjusted;
  })[0] || null;
  const recommendedLevelId = recommendedLearningLevel(current.songs?.[songId]);
  const baselineSection = sections[0] || { start: 0, end: 8 };
  if (!focusSkill) {
    return {
      version: 1,
      source: 'baseline',
      skillId: 'notes',
      skillLabel: 'Notes',
      title: 'Create your starting point',
      reason: 'One short measured attempt lets Polymath choose the next exercise from evidence.',
      instruction: 'Prepare the piano, hear one short part, then play it once at a comfortable speed.',
      successRule: 'Finish one measured attempt. No score is treated as a judgement.',
      speedPercent: 70,
      recommendedLevelId,
      recommendedSectionIndex: 0,
      recommendedRange: normalizedRange(baselineSection),
      recommendedHand: learningLevelById(levelId).handMode,
      confidence: 0,
      goal: {
        metricKey: 'notes',
        targetScore: null,
        requiredPasses: 1,
        passes: 0,
        remainingPasses: 1,
        speedPercent: 70,
        latestScore: null,
        highestCompletedSpeed: null,
      },
      mastery,
    };
  }
  const exercise = ADAPTIVE_EXERCISES[focusSkill.id];
  const recommendedSectionIndex = weakestSectionIndex(
    current.songs?.[songId],
    sections,
    focusSkill.metricKey,
  );
  const recommendedRange = normalizedRange(sections[recommendedSectionIndex] || baselineSection);
  const goal = adaptiveGoalForSection({
    progress: current,
    songId,
    skill: focusSkill,
    exercise,
    levelId: recommendedLevelId,
    range: recommendedRange,
  });
  const recentAttempt = current.history.filter((attempt) => attempt.songId === String(songId)).at(-1);
  const timingDetail = focusSkill.id === 'rhythm' && recentAttempt?.timingDirection !== 'centred'
    ? ` Your recent matched notes tended to land ${recentAttempt.timingDirection}.`
    : '';
  return {
    version: 1,
    source: 'measured',
    skillId: focusSkill.id,
    skillLabel: focusSkill.label,
    title: exercise.title,
    reason: `${focusSkill.label} is the clearest measured opportunity at ${focusSkill.score}% across ${focusSkill.observations} attempt${focusSkill.observations === 1 ? '' : 's'}.${timingDetail}`,
    instruction: exercise.instruction,
    successRule: exercise.successRule,
    speedPercent: goal.speedPercent,
    recommendedLevelId,
    recommendedSectionIndex,
    recommendedRange,
    recommendedHand: learningLevelById(recommendedLevelId).handMode,
    confidence: focusSkill.confidence,
    goal,
    mastery,
  };
}

function skillFromAttempt(report) {
  const explicit = PIANO_MASTERY_SKILLS.find((skill) => skill.id === report?.practiceFocusId);
  if (explicit) return explicit;
  const focus = String(report?.focus || '').trim().toLowerCase();
  return PIANO_MASTERY_SKILLS.find((skill) => skill.label.toLowerCase() === focus)
    || PIANO_MASTERY_SKILLS[0];
}

export function evaluateAdaptivePracticeOutcome({ progress, songId, report } = {}) {
  if (!report) return null;
  const current = normalizeLearningProgress(progress);
  const snapshot = learningAttemptSnapshot(songId, report);
  const skill = skillFromAttempt(snapshot);
  const exercise = ADAPTIVE_EXERCISES[skill.id];
  const currentScore = attemptSkillScore(snapshot, skill);
  const baseline = snapshot.practicePlanSource === 'baseline';
  const speedPercent = roundedPracticeSpeed(snapshot, exercise.speedPercent);
  const comparable = current.history
    .filter((attempt) => (
      attempt.songId === snapshot.songId
      && attempt.levelId === snapshot.levelId
      && rangesMatch(attempt.range, snapshot.range)
      && attempt.speedPercent !== null
      && attempt.speedPercent !== undefined
      && roundedPracticeSpeed(attempt, exercise.speedPercent) === speedPercent
      && attemptSkillScore(attempt, skill) !== null
    ))
    .sort((left, right) => validDateValue(left.createdAt) - validDateValue(right.createdAt));
  const currentIndex = comparable.findIndex((attempt) => attempt.id === snapshot.id);
  const attemptsThroughCurrent = currentIndex >= 0
    ? comparable.slice(0, currentIndex + 1)
    : [...comparable, snapshot];
  const previous = attemptsThroughCurrent.length > 1
    ? attemptsThroughCurrent.at(-2)
    : null;
  const previousScore = attemptSkillScore(previous, skill);
  const improvement = currentScore !== null && previousScore !== null
    ? currentScore - previousScore
    : null;
  const targetScore = snapshot.practiceTargetScore ?? exercise.targetScore;
  const passes = baseline ? 1 : Math.min(
    exercise.requiredPasses,
    consecutivePassingAttempts(attemptsThroughCurrent, skill, targetScore),
  );
  const achieved = baseline || passes >= exercise.requiredPasses;
  const passedThisAttempt = baseline || (currentScore !== null && currentScore >= targetScore);

  let headline = 'Keep this speed and try once more';
  let nextAction = `Aim for ${targetScore}% ${skill.label.toLowerCase()} at ${speedPercent}% speed.`;
  if (baseline) {
    headline = 'Starting point captured';
    nextAction = 'Polymath can now choose one measured skill and section for your next attempt.';
  } else if (achieved) {
    headline = speedPercent >= 100 ? 'Focus goal mastered' : 'Two clean passes—tempo can rise';
    nextAction = speedPercent >= 100
      ? 'Keep this skill strong while Polymath moves to the next measured opportunity.'
      : 'Your next plan will raise the tempo without changing everything at once.';
  } else if (passedThisAttempt) {
    headline = 'One clean pass—repeat it once';
    nextAction = `Repeat the same section at ${speedPercent}% to prove it is reliable.`;
  }

  return {
    version: 1,
    skillId: skill.id,
    skillLabel: skill.label,
    score: currentScore,
    previousScore,
    improvement,
    speedPercent,
    targetScore: baseline ? null : targetScore,
    requiredPasses: baseline ? 1 : exercise.requiredPasses,
    passes,
    achieved,
    passedThisAttempt,
    status: baseline ? 'baseline' : achieved ? 'achieved' : passedThisAttempt ? 'passed' : 'retry',
    headline,
    nextAction,
  };
}

function newerSong(left, right) {
  return validDateValue(right?.lastPracticedAt) > validDateValue(left?.lastPracticedAt) ? right : left;
}

function mergeSections(leftSections = {}, rightSections = {}) {
  const output = {};
  new Set([...Object.keys(leftSections), ...Object.keys(rightSections)]).forEach((key) => {
    const left = leftSections[key] || {};
    const right = rightSections[key] || {};
    const recent = newerSong(left, right) || {};
    const skillScores = {};
    new Set([...Object.keys(left.skillScores || {}), ...Object.keys(right.skillScores || {})]).forEach((metricKey) => {
      const leftSkill = left.skillScores?.[metricKey] || {};
      const rightSkill = right.skillScores?.[metricKey] || {};
      const last = validDateValue(right.lastPracticedAt) > validDateValue(left.lastPracticedAt) ? rightSkill : leftSkill;
      skillScores[metricKey] = {
        attempts: Math.max(Number(leftSkill.attempts || 0), Number(rightSkill.attempts || 0)),
        bestScore: Math.max(Number(leftSkill.bestScore || 0), Number(rightSkill.bestScore || 0)),
        lastScore: finiteScore(last.lastScore) ?? 0,
      };
    });
    output[key] = {
      ...recent,
      attempts: Math.max(Number(left.attempts || 0), Number(right.attempts || 0)),
      bestScore: Math.max(Number(left.bestScore || 0), Number(right.bestScore || 0)),
      skillScores,
    };
  });
  return output;
}

export function mergeLearningProgress(leftProgress, rightProgress) {
  const left = normalizeLearningProgress(leftProgress);
  const right = normalizeLearningProgress(rightProgress);
  const byId = new Map();
  [...left.history, ...right.history].forEach((attempt) => byId.set(attempt.id, attempt));
  const history = [...byId.values()]
    .sort((a, b) => validDateValue(a.createdAt) - validDateValue(b.createdAt))
    .slice(-MAX_LOCAL_ATTEMPT_HISTORY);
  let rebuilt = emptyLearningProgress();
  history.forEach((attempt) => { rebuilt = recordLearningAttempt(rebuilt, attempt.songId, attempt); });

  const songs = { ...rebuilt.songs };
  new Set([...Object.keys(left.songs), ...Object.keys(right.songs)]).forEach((songId) => {
    const first = left.songs[songId] || {};
    const second = right.songs[songId] || {};
    const calculated = songs[songId] || {};
    const recent = newerSong(newerSong(first, second), calculated) || {};
    songs[songId] = {
      ...recent,
      attempts: Math.max(Number(first.attempts || 0), Number(second.attempts || 0), Number(calculated.attempts || 0)),
      bestScore: Math.max(Number(first.bestScore || 0), Number(second.bestScore || 0), Number(calculated.bestScore || 0)),
      sections: mergeSections(mergeSections(first.sections, second.sections), calculated.sections),
    };
  });
  return {
    version: PROGRESS_VERSION,
    totalAttempts: Math.max(
      Number(left.totalAttempts || 0),
      Number(right.totalAttempts || 0),
      Number(rebuilt.totalAttempts || 0),
      Object.values(songs).reduce((sum, song) => sum + Number(song.attempts || 0), 0),
    ),
    practiceSeconds: Math.max(Number(left.practiceSeconds || 0), Number(right.practiceSeconds || 0), Number(rebuilt.practiceSeconds || 0)),
    songs,
    history,
  };
}

export function learningProgressFromAttempts(attempts = []) {
  return mergeLearningProgress(emptyLearningProgress(), {
    ...emptyLearningProgress(),
    history: Array.isArray(attempts) ? attempts : [],
  });
}
