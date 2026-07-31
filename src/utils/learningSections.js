function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function eventEnd(event) {
  return Number(event?.time || 0) + Number(event?.duration || event?.audioDuration || 0.25);
}

function eventSignature(event) {
  if (event?.chord) return `chord:${event.chord}`;
  if (Number.isFinite(event?.midi)) return `midi:${event.midi % 12}`;
  if (event?.note) return `note:${String(event.note).replace(/-?\d+$/, '')}`;
  if (Array.isArray(event?.frets)) return `frets:${event.frets.join(',')}`;
  return 'event';
}

function sequenceAt(events, index, size = 4) {
  return events.slice(index, index + size).map(eventSignature).join('|');
}

function sectionName(index, count) {
  if (index === 0) return 'Intro';
  if (index === count - 1) return 'Ending';
  return `Part ${index + 1}`;
}

export function formatLearningTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

export function analyzeLearningSections(rawEvents = [], rawDuration = 0, preferredSeconds = 15) {
  const events = [...rawEvents]
    .filter((event) => Number.isFinite(Number(event?.time)))
    .sort((left, right) => Number(left.time) - Number(right.time));
  const duration = Math.max(Number(rawDuration) || 0, ...events.map(eventEnd));
  if (!events.length || duration <= 0) return [];

  const recommendation = clamp(Number(preferredSeconds) || 15, 5, 60);
  const naturalCount = Math.max(1, Math.round(duration / Math.min(recommendation, 15)));
  const desiredCount = duration >= 55 ? clamp(naturalCount, 7, 10) : clamp(naturalCount, 1, 7);
  if (desiredCount === 1) {
    return [{ id: 'section-1', name: 'Full song', start: 0, end: duration, duration, reason: 'complete song' }];
  }

  const occurrences = new Map();
  events.forEach((event, index) => {
    const signature = sequenceAt(events, index);
    if (!occurrences.has(signature)) occurrences.set(signature, []);
    occurrences.get(signature).push(index);
  });

  const candidates = events.slice(1).map((event, index) => {
    const eventIndex = index + 1;
    const time = Number(event.time);
    const previousEnd = eventEnd(events[eventIndex - 1]);
    const gap = Math.max(0, time - previousEnd);
    const repeatedPhrase = (occurrences.get(sequenceAt(events, eventIndex)) || []).length > 1;
    const repeatedKeyStyle = eventSignature(events[eventIndex]) === eventSignature(events[0])
      || eventSignature(events[eventIndex]) === eventSignature(events[Math.max(0, eventIndex - 4)]);
    return {
      time,
      score: gap * 8 + (gap >= 0.6 ? 8 : 0) + (repeatedPhrase ? 5 : 0) + (repeatedKeyStyle ? 1.5 : 0),
      reason: repeatedPhrase ? 'repeated musical pattern' : gap >= 0.45 ? 'musical pause' : 'phrase change',
    };
  });

  const boundaries = [0];
  for (let part = 1; part < desiredCount; part += 1) {
    const ideal = (duration * part) / desiredCount;
    const previous = boundaries[boundaries.length - 1];
    const remainingParts = desiredCount - part;
    const latest = duration - remainingParts * Math.max(3, recommendation * 0.45);
    const viable = candidates.filter((candidate) => candidate.time > previous + 2.5 && candidate.time < latest);
    const pool = viable.length ? viable : candidates.filter((candidate) => candidate.time > previous + 2.5);
    if (!pool.length) break;
    const chosen = [...pool].sort((left, right) => {
      const leftValue = Math.abs(left.time - ideal) - left.score * 0.7;
      const rightValue = Math.abs(right.time - ideal) - right.score * 0.7;
      return leftValue - rightValue;
    })[0];
    boundaries.push(chosen.time);
  }
  boundaries.push(duration);

  return boundaries.slice(0, -1).map((start, index) => {
    const end = boundaries[index + 1];
    const nearest = candidates.reduce((best, candidate) => (
      Math.abs(candidate.time - end) < Math.abs((best?.time ?? Infinity) - end) ? candidate : best
    ), null);
    return {
      id: `section-${index + 1}`,
      name: sectionName(index, boundaries.length - 1),
      start,
      end,
      duration: end - start,
      reason: nearest?.reason || 'balanced practice length',
      exceedsRecommendation: end - start > recommendation,
    };
  });
}
