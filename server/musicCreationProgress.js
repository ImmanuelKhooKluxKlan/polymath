'use strict';

const TERMINAL_STATUSES = new Set(['COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED']);
const DEFAULT_WINDOWS_SECONDS = Object.freeze({
  draft: Object.freeze([35, 90]),
  revise: Object.freeze([25, 75]),
});

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function timestamp(value) {
  const parsed = new Date(value || 0).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
}

function percentile(sorted, fraction) {
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * (position - lower));
}

function completedDurationSeconds(job) {
  if (!job || !TERMINAL_STATUSES.has(String(job.status || '').toUpperCase())) return 0;
  const started = timestamp(job.createdAt);
  const finished = timestamp(job.completedAt || job.updatedAt);
  const duration = (finished - started) / 1000;
  return Number.isFinite(duration) && duration >= 5 && duration <= 300 ? duration : 0;
}

function durationWindow(job, jobs = []) {
  const kind = String(job?.kind || 'draft').toLowerCase() === 'revise' ? 'revise' : 'draft';
  const fallback = DEFAULT_WINDOWS_SECONDS[kind];
  const samples = jobs
    .filter((candidate) => (
      candidate !== job
      && String(candidate?.kind || 'draft').toLowerCase() === kind
      && String(candidate?.status || '').toUpperCase() === 'COMPLETED'
      && candidate?.blueprint
    ))
    .map(completedDurationSeconds)
    .filter(Boolean)
    .slice(-30)
    .sort((left, right) => left - right);

  if (samples.length < 3) {
    return { low: fallback[0], high: fallback[1], learned: false, sampleCount: samples.length };
  }

  const lowerQuartile = percentile(samples, 0.25);
  const upperQuartile = percentile(samples, 0.75);
  return {
    low: clamp(Math.floor(lowerQuartile * 0.85), 15, 180),
    high: clamp(Math.ceil(Math.max(upperQuartile * 1.2, lowerQuartile + 15)), 30, 240),
    learned: true,
    sampleCount: samples.length,
  };
}

function roundSeconds(value) {
  const safe = Math.max(0, Number(value) || 0);
  if (safe < 10) return Math.ceil(safe);
  return Math.ceil(safe / 5) * 5;
}

function describeMusicCreationProgress(job, jobs = [], now = Date.now()) {
  const status = String(job?.status || 'IN_QUEUE').toUpperCase();
  const createdAt = timestamp(job?.createdAt) || now;
  const elapsedSeconds = Math.max(0, Math.floor((now - createdAt) / 1000));
  const window = durationWindow(job, jobs);
  const midpoint = (window.low + window.high) / 2;
  const terminal = TERMINAL_STATUSES.has(status);
  const queued = status === 'IN_QUEUE';
  let percent;

  if (status === 'COMPLETED') percent = 100;
  else if (terminal) percent = 100;
  else if (queued) percent = clamp(5 + ((elapsedSeconds / Math.max(midpoint, 1)) * 14), 5, 19);
  else percent = clamp(24 + ((elapsedSeconds / Math.max(midpoint, 1)) * 66), 24, 94);

  const remainingLowSeconds = roundSeconds(Math.max(0, window.low - elapsedSeconds));
  const remainingHighSeconds = roundSeconds(Math.max(0, window.high - elapsedSeconds));

  return {
    percent: Math.round(percent),
    elapsedSeconds,
    remainingLowSeconds,
    remainingHighSeconds,
    overEstimate: !terminal && elapsedSeconds >= window.high,
    learnedEstimate: window.learned,
    sampleCount: window.sampleCount,
  };
}

module.exports = {
  completedDurationSeconds,
  describeMusicCreationProgress,
  durationWindow,
};
