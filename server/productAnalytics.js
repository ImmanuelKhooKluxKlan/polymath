const crypto = require('node:crypto');

const PRODUCT_EVENT_NAMES = new Set([
  'app_opened',
  'route_viewed',
  'keyboard_prepare_started',
  'keyboard_prepare_completed',
  'learning_preview_opened',
  'learning_upgrade_clicked',
  'learning_attempt_started',
  'learning_attempt_completed',
  'learning_win_shared',
  'transcription_file_selected',
  'transcription_started',
  'transcription_restored',
  'transcription_completed',
  'transcription_failed',
  'transcription_feedback',
  'subscription_page_viewed',
  'checkout_started',
  'checkout_returned',
  'subscription_activated',
]);

// Completion, payment, and quality events are written only by trusted server
// paths. Browsers may report navigation and interaction evidence, but cannot
// manufacture revenue or transcription-success numbers in the admin console.
const PUBLIC_PRODUCT_EVENT_NAMES = new Set([
  'app_opened',
  'route_viewed',
  'keyboard_prepare_started',
  'keyboard_prepare_completed',
  'learning_preview_opened',
  'learning_upgrade_clicked',
  'learning_attempt_started',
  'learning_attempt_completed',
  'learning_win_shared',
  'transcription_file_selected',
  'transcription_restored',
  'subscription_page_viewed',
  'checkout_returned',
]);

const SAFE_PROPERTY_NAMES = new Set([
  'audience', 'deviceClass', 'durationMs', 'durationSeconds', 'execution', 'feedback',
  'freePreview', 'hand', 'inputMode', 'instrument', 'interval', 'level', 'noteCount',
  'outcome', 'page', 'performanceTier', 'plan', 'playbackMode', 'productId', 'qualityScore',
  'refunded', 'restored', 'score', 'signedIn', 'sizeBucket', 'sourceKind', 'tier',
]);

function cleanIdentifier(value, maximum = 100) {
  const cleaned = String(value || '').trim();
  return /^[A-Za-z0-9_-]{8,160}$/.test(cleaned) ? cleaned.slice(0, maximum) : '';
}

function cleanText(value, maximum = 80) {
  return String(value || '').trim().replace(/[\u0000-\u001f\u007f]/g, '').slice(0, maximum);
}

function cleanPath(value) {
  return cleanText(value, 160).replace(/^#/, '').split(/[?#]/)[0].slice(0, 100);
}

function safeProperties(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  const output = {};
  for (const [key, candidate] of Object.entries(value)) {
    if (!SAFE_PROPERTY_NAMES.has(key)) continue;
    if (typeof candidate === 'boolean') output[key] = candidate;
    else if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      output[key] = Math.max(-1_000_000_000, Math.min(1_000_000_000, candidate));
    } else if (typeof candidate === 'string') output[key] = cleanText(candidate);
  }
  return output;
}

function safeOccurredAt(value, now = Date.now()) {
  const parsed = Date.parse(String(value || ''));
  const oldest = now - (7 * 24 * 60 * 60 * 1000);
  const newest = now + (5 * 60 * 1000);
  return new Date(Number.isFinite(parsed) && parsed >= oldest && parsed <= newest ? parsed : now).toISOString();
}

function sanitizeProductEventBatch(value, {
  userId = '',
  now = Date.now(),
  allowedEventNames = PRODUCT_EVENT_NAMES,
} = {}) {
  const entries = Array.isArray(value) ? value.slice(0, 20) : [];
  return entries.flatMap((candidate) => {
    const eventName = cleanText(candidate?.eventName, 60).toLowerCase();
    if (!PRODUCT_EVENT_NAMES.has(eventName) || !allowedEventNames.has(eventName)) return [];
    const eventId = cleanIdentifier(candidate?.eventId, 100) || `event_${crypto.randomUUID()}`;
    const anonymousId = cleanIdentifier(candidate?.anonymousId, 100);
    const sessionId = cleanIdentifier(candidate?.sessionId, 100);
    if (!userId && !anonymousId && !sessionId) return [];
    return [{
      eventId,
      eventName,
      occurredAt: safeOccurredAt(candidate?.occurredAt, now),
      userId: cleanText(userId, 100),
      anonymousId,
      sessionId,
      path: cleanPath(candidate?.path),
      release: cleanText(candidate?.release, 80),
      properties: safeProperties(candidate?.properties),
    }];
  });
}

function actorFor(event) {
  return event.userId || event.anonymousId || event.sessionId || '';
}

function percent(numerator, denominator) {
  return denominator > 0 ? Math.round((numerator / denominator) * 1000) / 10 : null;
}

function summaryFromCounts({ counts = [], daily = [], feedback = [], days = 30, returningActors = 0, signedActors = 0 } = {}) {
  const byName = new Map(counts.map((entry) => [entry.eventName, {
    events: Number(entry.events || 0),
    actors: Number(entry.actors || 0),
    averageScore: entry.averageScore === null || entry.averageScore === undefined
      ? null
      : Math.round(Number(entry.averageScore) * 10) / 10,
    averageDurationSeconds: entry.averageDurationSeconds === null || entry.averageDurationSeconds === undefined
      ? null
      : Math.round(Number(entry.averageDurationSeconds) * 10) / 10,
  }]));
  const count = (name) => byName.get(name)?.actors || 0;
  const stages = [
    ['visited', 'Visited', count('app_opened')],
    ['lesson_started', 'Started a measured lesson', count('learning_attempt_started')],
    ['lesson_completed', 'Earned a score', count('learning_attempt_completed')],
    ['shared', 'Shared a win', count('learning_win_shared')],
    ['checkout', 'Started checkout', count('checkout_started')],
    ['paid', 'Activated a subscription', count('subscription_activated')],
  ].map(([id, label, actors], index, all) => ({
    id,
    label,
    actors,
    fromVisitPercent: percent(actors, all[0][2]),
    fromPreviousPercent: index === 0 ? 100 : percent(actors, all[index - 1][2]),
  }));
  const transcriptionStarted = count('transcription_started');
  const transcriptionCompleted = count('transcription_completed');
  const transcriptionFailed = count('transcription_failed');
  const feedbackByValue = new Map(feedback.map((entry) => [entry.feedback, Number(entry.actors || 0)]));
  const accurateFeedback = feedbackByValue.get('accurate') || 0;
  const needsWorkFeedback = feedbackByValue.get('needs-work') || 0;
  return {
    windowDays: days,
    generatedAt: new Date().toISOString(),
    stages,
    events: Object.fromEntries(byName),
    daily,
    learning: {
      averageScore: byName.get('learning_attempt_completed')?.averageScore ?? null,
      completedAttempts: byName.get('learning_attempt_completed')?.events || 0,
    },
    transcription: {
      startedActors: transcriptionStarted,
      completedActors: transcriptionCompleted,
      failedActors: transcriptionFailed,
      completionPercent: percent(transcriptionCompleted, transcriptionStarted),
      failurePercent: percent(transcriptionFailed, transcriptionStarted),
      averageDurationSeconds: byName.get('transcription_completed')?.averageDurationSeconds ?? null,
      feedbackActors: count('transcription_feedback'),
      accurateFeedbackActors: accurateFeedback,
      needsWorkFeedbackActors: needsWorkFeedback,
      playablePercent: percent(accurateFeedback, accurateFeedback + needsWorkFeedback),
    },
    returnSignal: {
      signedActors,
      returningActors,
      returningPercent: percent(returningActors, signedActors),
      definition: 'Signed-in people active on at least two separate UTC days in this window.',
    },
    privacy: 'No source audio, filenames, song titles, messages, IP addresses, email addresses, or phone numbers are stored in product events.',
  };
}

function summarizeProductEvents(events = [], days = 30) {
  const actorSets = new Map();
  const totals = new Map();
  const scores = new Map();
  const durations = new Map();
  const dailyMap = new Map();
  const signedDays = new Map();
  const feedbackActors = new Map();
  for (const event of events) {
    const actor = actorFor(event);
    if (!actor) continue;
    if (!actorSets.has(event.eventName)) actorSets.set(event.eventName, new Set());
    actorSets.get(event.eventName).add(actor);
    totals.set(event.eventName, (totals.get(event.eventName) || 0) + 1);
    if (Number.isFinite(event.properties?.score)) {
      if (!scores.has(event.eventName)) scores.set(event.eventName, []);
      scores.get(event.eventName).push(event.properties.score);
    }
    if (Number.isFinite(event.properties?.durationSeconds)) {
      if (!durations.has(event.eventName)) durations.set(event.eventName, []);
      durations.get(event.eventName).push(event.properties.durationSeconds);
    }
    const day = String(event.occurredAt || '').slice(0, 10);
    dailyMap.set(day, (dailyMap.get(day) || 0) + 1);
    if (event.userId && day) {
      if (!signedDays.has(event.userId)) signedDays.set(event.userId, new Set());
      signedDays.get(event.userId).add(day);
    }
    if (event.eventName === 'transcription_feedback' && event.properties?.feedback) {
      const value = event.properties.feedback;
      if (!feedbackActors.has(value)) feedbackActors.set(value, new Set());
      feedbackActors.get(value).add(actor);
    }
  }
  const mean = (values = []) => values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
  const counts = [...totals].map(([eventName, total]) => ({
    eventName,
    events: total,
    actors: actorSets.get(eventName)?.size || 0,
    averageScore: mean(scores.get(eventName)),
    averageDurationSeconds: mean(durations.get(eventName)),
  }));
  const daily = [...dailyMap].sort(([left], [right]) => left.localeCompare(right))
    .map(([day, eventCount]) => ({ day, eventCount }));
  const returningActors = [...signedDays.values()].filter((seen) => seen.size >= 2).length;
  const feedback = [...feedbackActors].map(([value, actors]) => ({ feedback: value, actors: actors.size }));
  return summaryFromCounts({
    counts,
    daily,
    feedback,
    days,
    returningActors,
    signedActors: signedDays.size,
  });
}

module.exports = {
  PRODUCT_EVENT_NAMES,
  PUBLIC_PRODUCT_EVENT_NAMES,
  sanitizeProductEventBatch,
  safeProperties,
  summarizeProductEvents,
  summaryFromCounts,
};
