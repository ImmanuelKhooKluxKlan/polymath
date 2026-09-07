const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAILY_ATTEMPT_GOAL = 1;
const STREAK_MILESTONES = Object.freeze([3, 7, 14, 30, 60, 100]);

function safeTimeZone(timeZone) {
  const requested = String(timeZone || '').trim();
  if (requested) {
    try {
      new Intl.DateTimeFormat('en-US', { timeZone: requested }).format(new Date());
      return requested;
    } catch {
      // Fall through to the device zone when a stored zone is no longer valid.
    }
  }
  return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
}

function localDayKey(value, timeZone) {
  const date = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(date.getTime())) return '';
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function shiftDayKey(dayKey, offset) {
  const [year, month, day] = String(dayKey).split('-').map(Number);
  if (![year, month, day].every(Number.isFinite)) return '';
  return new Date(Date.UTC(year, month - 1, day) + (offset * DAY_MS))
    .toISOString()
    .slice(0, 10);
}

function weekdayLabel(dayKey) {
  const [year, month, day] = dayKey.split('-').map(Number);
  return new Intl.DateTimeFormat('en-US', { weekday: 'short', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, month - 1, day)))
    .slice(0, 1);
}

function safeHistory(progress) {
  return Array.isArray(progress?.history)
    ? progress.history.filter((attempt) => localDayKey(attempt?.createdAt, 'UTC'))
    : [];
}

export function buildLearningMomentum(progress, options = {}) {
  const now = options.now instanceof Date ? options.now : new Date(options.now || Date.now());
  const timeZone = safeTimeZone(options.timeZone);
  const dailyGoal = Math.max(1, Math.min(5, Math.round(Number(options.dailyAttemptGoal) || DEFAULT_DAILY_ATTEMPT_GOAL)));
  const todayKey = localDayKey(now, timeZone);
  const attempts = safeHistory(progress).map((attempt) => ({
    ...attempt,
    dayKey: localDayKey(attempt.createdAt, timeZone),
  })).filter((attempt) => attempt.dayKey);
  const attemptsByDay = new Map();
  attempts.forEach((attempt) => {
    const existing = attemptsByDay.get(attempt.dayKey) || [];
    existing.push(attempt);
    attemptsByDay.set(attempt.dayKey, existing);
  });
  const activeDays = new Set(attemptsByDay.keys());
  const todayAttempts = attemptsByDay.get(todayKey) || [];
  const todayComplete = todayAttempts.length >= dailyGoal;
  const yesterdayKey = shiftDayKey(todayKey, -1);
  const anchorKey = todayComplete ? todayKey : yesterdayKey;
  let streakDays = 0;
  while (activeDays.has(shiftDayKey(anchorKey, -streakDays))) streakDays += 1;

  const week = Array.from({ length: 7 }, (_, index) => {
    const key = shiftDayKey(todayKey, index - 6);
    const count = (attemptsByDay.get(key) || []).length;
    return {
      key,
      label: weekdayLabel(key),
      count,
      active: count > 0,
      today: key === todayKey,
    };
  });
  const nextMilestone = STREAK_MILESTONES.find((milestone) => milestone > streakDays)
    || Math.ceil((streakDays + 1) / 100) * 100;

  return {
    timeZone,
    todayKey,
    dailyGoal,
    todayAttempts: todayAttempts.length,
    todayComplete,
    remainingToday: Math.max(0, dailyGoal - todayAttempts.length),
    streakDays,
    streakAtRisk: !todayComplete && streakDays > 0,
    activeDaysThisWeek: week.filter((day) => day.active).length,
    attemptsThisWeek: week.reduce((total, day) => total + day.count, 0),
    week,
    nextMilestone,
    daysToMilestone: Math.max(0, nextMilestone - streakDays),
  };
}
