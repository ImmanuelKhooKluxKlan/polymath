'use strict';

const LEVEL_IDS = new Set(['melody', 'beginner', 'intermediate', 'original']);
const HAND_MODES = new Set(['left', 'right', 'both']);
const PRACTICE_FOCUS_IDS = new Set(['notes', 'rhythm', 'holds', 'touch', 'pedal']);
const METRICS = Object.freeze({
  notes: 'Notes',
  rhythm: 'Rhythm',
  duration: 'Holds',
  dynamics: 'Touch',
  pedal: 'Pedal',
});

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function text(value, maximum) {
  return String(value || '').trim().slice(0, maximum);
}

function score(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(clamp(number, 0, 100)) : null;
}

function count(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(clamp(number, 0, 100000)) : 0;
}

function learningError(message, code) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function safeCreatedAt(value, now) {
  const timestamp = new Date(value || '').getTime();
  const earliest = new Date('2020-01-01T00:00:00.000Z').getTime();
  if (!Number.isFinite(timestamp) || timestamp < earliest || timestamp > now.getTime() + (10 * 60 * 1000)) {
    return now.toISOString();
  }
  return new Date(timestamp).toISOString();
}

function sanitizeMetrics(metrics) {
  return Object.fromEntries(Object.entries(METRICS).map(([key, label]) => {
    const raw = metrics?.[key];
    const measuredScore = score(raw?.score);
    return [key, {
      label,
      score: measuredScore,
      available: raw?.available !== false && measuredScore !== null,
    }];
  }));
}

function sanitizeMissedNotes(items) {
  if (!Array.isArray(items)) return [];
  return items.slice(0, 5).map((item) => ({
    note: text(item?.note, 12),
    count: Math.max(1, count(item?.count || 1)),
  })).filter((item) => item.note);
}

function sanitizeLearningAttempt(payload, userId, options = {}) {
  const now = options.now instanceof Date ? options.now : new Date();
  const report = payload?.report && typeof payload.report === 'object' ? payload.report : {};
  const clientAttemptId = text(payload?.attemptId || report.attemptId, 120);
  if (!/^[A-Za-z0-9:_-]{8,120}$/.test(clientAttemptId)) {
    throw learningError('A valid learning attempt ID is required.', 'INVALID_LEARNING_ATTEMPT_ID');
  }
  const songId = text(payload?.songId, 180);
  if (!songId) throw learningError('Choose a song before saving learning progress.', 'INVALID_LEARNING_SONG');

  const start = clamp(Number(report.range?.start) || 0, 0, 86400);
  const requestedEnd = Number(report.range?.end);
  const end = clamp(Number.isFinite(requestedEnd) ? requestedEnd : start, start, Math.min(86400, start + 3600));
  const levelId = LEVEL_IDS.has(report.levelId) ? report.levelId : 'beginner';
  const createdAt = safeCreatedAt(report.createdAt, now);
  const attemptScore = score(report.score) ?? 0;
  const internalId = typeof options.idFactory === 'function'
    ? options.idFactory('learning_attempt')
    : `learning_attempt_${clientAttemptId}`;

  return {
    id: internalId,
    userId: text(userId, 160),
    clientAttemptId,
    songId,
    songTitle: text(payload?.songTitle, 180),
    createdAt,
    savedAt: now.toISOString(),
    levelId,
    range: { start, end },
    sectionKey: `${levelId}:${start.toFixed(2)}-${end.toFixed(2)}`,
    elapsedSeconds: Math.max(0, end - start),
    score: attemptScore,
    metrics: sanitizeMetrics(report.metrics),
    focus: text(report.focus, 40),
    strongest: text(report.strongest, 40),
    timingDirection: ['early', 'late', 'centred'].includes(report.timingDirection)
      ? report.timingDirection
      : 'centred',
    timingBiasMs: Math.round(clamp(Number(report.timingBiasMs) || 0, -5000, 5000)),
    speedPercent: report.speedPercent !== null
      && report.speedPercent !== undefined
      && report.speedPercent !== ''
      && Number.isFinite(Number(report.speedPercent))
      ? Math.round(clamp(Number(report.speedPercent), 20, 200))
      : null,
    handMode: HAND_MODES.has(report.handMode) ? report.handMode : 'both',
    practiceFocusId: PRACTICE_FOCUS_IDS.has(report.practiceFocusId) ? report.practiceFocusId : '',
    practicePlanSource: ['baseline', 'measured'].includes(report.practicePlanSource)
      ? report.practicePlanSource
      : '',
    practiceTargetScore: score(report.practiceTargetScore),
    expectedCount: count(report.expectedCount),
    matchedCount: count(report.matchedCount),
    missedCount: count(report.missedCount),
    extraCount: count(report.extraCount),
    missedNotes: sanitizeMissedNotes(report.missedNotes),
  };
}

function publicLearningAttempt(attempt) {
  return {
    id: attempt.clientAttemptId,
    songId: attempt.songId,
    songTitle: attempt.songTitle || '',
    createdAt: attempt.createdAt,
    levelId: attempt.levelId,
    range: attempt.range,
    sectionKey: attempt.sectionKey,
    elapsedSeconds: attempt.elapsedSeconds,
    score: attempt.score,
    metrics: attempt.metrics,
    focus: attempt.focus,
    strongest: attempt.strongest,
    timingDirection: attempt.timingDirection,
    timingBiasMs: attempt.timingBiasMs,
    speedPercent: attempt.speedPercent ?? null,
    handMode: attempt.handMode || 'both',
    practiceFocusId: attempt.practiceFocusId || '',
    practicePlanSource: attempt.practicePlanSource || '',
    practiceTargetScore: attempt.practiceTargetScore ?? null,
    expectedCount: attempt.expectedCount,
    matchedCount: attempt.matchedCount,
    missedCount: attempt.missedCount,
    extraCount: attempt.extraCount,
    missedNotes: attempt.missedNotes,
  };
}

function learningAttemptsForUser(db, userId, maximum = 500) {
  return (Array.isArray(db?.learningAttempts) ? db.learningAttempts : [])
    .filter((attempt) => attempt.userId === userId)
    .sort((left, right) => new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime())
    .slice(-Math.max(1, maximum))
    .map(publicLearningAttempt);
}

function trimUserLearningAttempts(db, userId, maximum = 500) {
  if (!Array.isArray(db.learningAttempts)) db.learningAttempts = [];
  const owned = db.learningAttempts
    .filter((attempt) => attempt.userId === userId)
    .sort((left, right) => new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime());
  const keep = new Set(owned.slice(0, Math.max(1, maximum)).map((attempt) => attempt.id));
  db.learningAttempts = db.learningAttempts.filter((attempt) => attempt.userId !== userId || keep.has(attempt.id));
}

module.exports = {
  learningAttemptsForUser,
  publicLearningAttempt,
  sanitizeLearningAttempt,
  trimUserLearningAttempts,
};
