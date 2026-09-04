import { midiToNote, parseNote } from './noteMath.js';

const ONSET_WINDOW_SECONDS = 0.035;
const MATCH_WINDOW_SECONDS = 0.42;
const PROGRESS_VERSION = 1;

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
  const role = String(event?.scoreRole || event?.role || '').toLowerCase();
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
    const role = String(note.scoreRole || note.role || '').toLowerCase();
    return role.includes('melody') || role.includes('vocal') || role.includes('top');
  });
  const rightHand = group.notes.filter((note) => note.hand === 'right');
  const pool = explicitlyMelodic.length ? explicitlyMelodic : rightHand.length ? rightHand : group.notes;
  return [...pool].sort((left, right) => right.midi - left.midi)[0];
}

function outerVoices(notes, maximum) {
  const ordered = [...notes].sort((left, right) => left.midi - right.midi);
  if (ordered.length <= maximum) return ordered;
  if (maximum <= 2) return [ordered[0], ordered[ordered.length - 1]];
  const selected = [ordered[0], ordered[ordered.length - 1]];
  for (let slot = 1; slot < maximum - 1; slot += 1) {
    const index = Math.round((slot * (ordered.length - 1)) / (maximum - 1));
    selected.push(ordered[index]);
  }
  return [...new Map(selected.map((note) => [note.id, note])).values()]
    .sort((left, right) => left.midi - right.midi)
    .slice(0, maximum);
}

function removeMachineGunDuplicates(notes, minimumGap = 0.075) {
  const lastByMidi = new Map();
  return notes.filter((note) => {
    const lastTime = lastByMidi.get(note.midi);
    lastByMidi.set(note.midi, note.time);
    return lastTime === undefined || note.time - lastTime >= minimumGap;
  });
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
  let selected;
  if (levelId === 'melody') {
    selected = groups.map((group) => ({
      ...melodyFromGroup(group),
      hand: 'right',
      learningRole: 'melody',
      duration: Math.max(0.12, melodyFromGroup(group).duration),
    }));
  } else if (levelId === 'beginner') {
    selected = groups.flatMap((group) => {
      const left = group.notes.filter((note) => note.hand === 'left').sort((a, b) => a.midi - b.midi)[0];
      const melody = melodyFromGroup(group);
      return [...new Map([left, melody].filter(Boolean).map((note) => [note.id, {
        ...note,
        duration: Math.max(0.12, note.duration),
        learningRole: note.id === melody.id ? 'melody' : 'foundation',
      }])).values()];
    });
  } else {
    selected = groups.flatMap((group) => outerVoices(group.notes, 4));
  }

  return removeMachineGunDuplicates(
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
  return { version: PROGRESS_VERSION, totalAttempts: 0, practiceSeconds: 0, songs: {} };
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
    const parsed = JSON.parse(storage?.getItem?.(`polymath-learning-progress-v${PROGRESS_VERSION}:${learnerId}`) || 'null');
    if (!parsed || parsed.version !== PROGRESS_VERSION || typeof parsed.songs !== 'object') return emptyLearningProgress();
    return parsed;
  } catch {
    return emptyLearningProgress();
  }
}

export function recordLearningAttempt(progress, songId, report) {
  const current = progress?.version === PROGRESS_VERSION ? progress : emptyLearningProgress();
  const previousSong = current.songs?.[songId] || { attempts: 0, bestScore: 0, sections: {} };
  const sectionKey = `${report.levelId}:${report.range.start.toFixed(2)}-${report.range.end.toFixed(2)}`;
  const previousSection = previousSong.sections?.[sectionKey] || { attempts: 0, bestScore: 0 };
  const elapsed = Math.max(0, report.range.end - report.range.start);
  return {
    ...current,
    totalAttempts: Number(current.totalAttempts || 0) + 1,
    practiceSeconds: Number(current.practiceSeconds || 0) + elapsed,
    songs: {
      ...current.songs,
      [songId]: {
        ...previousSong,
        attempts: Number(previousSong.attempts || 0) + 1,
        bestScore: Math.max(Number(previousSong.bestScore || 0), report.score),
        lastScore: report.score,
        lastLevelId: report.levelId,
        lastPracticedAt: report.createdAt,
        sections: {
          ...previousSong.sections,
          [sectionKey]: {
            attempts: Number(previousSection.attempts || 0) + 1,
            bestScore: Math.max(Number(previousSection.bestScore || 0), report.score),
            lastScore: report.score,
          },
        },
      },
    },
  };
}

export function writeLearningProgress(storage, learnerId, progress) {
  try {
    storage?.setItem?.(`polymath-learning-progress-v${PROGRESS_VERSION}:${learnerId}`, JSON.stringify(progress));
    return true;
  } catch {
    return false;
  }
}
