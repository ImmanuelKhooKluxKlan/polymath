import { getSongDuration, normalizeSong } from './scheduler.js';

const DEFAULT_BASE_URL = 'https://polymathmusician67.com/';

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function safeSlug(value) {
  const slug = String(value || '').trim().toLowerCase();
  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) ? slug.slice(0, 64) : '';
}

function safeReferral(value) {
  const code = String(value || '').trim().toUpperCase();
  return /^[A-Z0-9_-]{3,32}$/.test(code) ? code : '';
}

function eventTime(event) {
  return Number(event?.time ?? event?.start ?? 0);
}

function eventDuration(event) {
  return Math.max(0.035, Number(
    event?.duration ?? event?.visualDuration ?? event?.audioDuration ?? 0.45,
  ) || 0.45);
}

function clippedEventDuration(value, fallback, eventStart, clippedStart, previewEnd) {
  const candidate = Number(value);
  const sourceDuration = Number.isFinite(candidate) && candidate > 0 ? candidate : fallback;
  const sourceEnd = eventStart + sourceDuration;
  return Math.max(0.035, Math.min(previewEnd, sourceEnd) - clippedStart);
}

export function campaignPlaybackRange(song) {
  const duration = Math.max(0, Number(song?.duration || 0), getSongDuration(song));
  return {
    id: `campaign-${song?.campaignId || 'challenge'}`,
    name: 'Artist challenge',
    start: 0,
    end: duration,
    duration,
  };
}

export function prepareCampaignSong(rawSong, campaign) {
  const previewStart = Math.max(0, Number(campaign?.preview?.startSeconds || 0));
  const previewDuration = clamp(Number(campaign?.preview?.durationSeconds || 20), 10, 45);
  const previewEnd = previewStart + previewDuration;
  const notes = (Array.isArray(rawSong?.notes) ? rawSong.notes : []).flatMap((note, index) => {
    const start = eventTime(note);
    const duration = eventDuration(note);
    const end = start + duration;
    if (!Number.isFinite(start) || start >= previewEnd || end <= previewStart) return [];
    const clippedStart = Math.max(previewStart, start);
    const clippedEnd = Math.min(previewEnd, end);
    const clippedDuration = Math.max(0.035, clippedEnd - clippedStart);
    return [{
      ...note,
      id: `campaign-note-${index}-${String(note.id || note.note || note.midi || 'event')}`,
      time: clippedStart - previewStart,
      duration: clippedDuration,
      visualDuration: clippedEventDuration(
        note.visualDuration,
        duration,
        start,
        clippedStart,
        previewEnd,
      ),
      audioDuration: clippedEventDuration(
        note.audioDuration,
        duration,
        start,
        clippedStart,
        previewEnd,
      ),
    }];
  });
  if (!notes.length) throw new Error('This campaign preview does not contain playable notes. Ask the administrator to verify its start time.');

  const sourcePedals = Array.isArray(rawSong?.pedals) ? rawSong.pedals : [];
  const priorPedal = sourcePedals
    .filter((pedal) => Number(pedal.time) <= previewStart)
    .sort((left, right) => Number(left.time) - Number(right.time))
    .at(-1);
  const pedals = [
    ...(priorPedal?.down ? [{ ...priorPedal, time: 0, source: priorPedal.source || 'campaign-boundary' }] : []),
    ...sourcePedals
      .filter((pedal) => Number(pedal.time) > previewStart && Number(pedal.time) < previewEnd)
      .map((pedal) => ({ ...pedal, time: Number(pedal.time) - previewStart })),
  ];

  const normalized = normalizeSong({
    ...rawSong,
    title: campaign?.title || rawSong?.title || 'Artist challenge',
    artist: campaign?.artist || rawSong?.artist || rawSong?.composer || '',
    composer: campaign?.artist || rawSong?.composer || rawSong?.artist || '',
    notes,
    pedals,
    duration: previewDuration,
    libraryId: `campaign:${campaign?.id || safeSlug(campaign?.slug)}`,
    libraryType: 'artist-campaign',
    campaignId: campaign?.id || '',
    campaignSlug: safeSlug(campaign?.slug),
    campaignPreviewStart: previewStart,
    campaignPreviewDuration: previewDuration,
    performance: {
      ...(rawSong?.performance || {}),
      preserveScoreDurations: true,
      preserveScoreTiming: true,
    },
  });
  return {
    ...normalized,
    duration: previewDuration,
    campaignId: campaign?.id || '',
    campaignSlug: safeSlug(campaign?.slug),
    campaignPreviewStart: previewStart,
    campaignPreviewDuration: previewDuration,
  };
}

export function campaignShareUrl(baseUrl, campaign, score = null) {
  let url;
  try {
    url = new URL(baseUrl || DEFAULT_BASE_URL, DEFAULT_BASE_URL);
  } catch {
    url = new URL(DEFAULT_BASE_URL);
  }
  const slug = safeSlug(campaign?.slug);
  if (!slug) return url.toString();
  const referral = safeReferral(campaign?.referralCode);
  const safeScore = Number.isFinite(Number(score))
    ? Math.round(clamp(Number(score), 0, 100))
    : null;
  const isLocal = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  url.search = '';
  if (isLocal) {
    const params = new URLSearchParams({ try: 'learn', campaign: slug });
    if (safeScore !== null) params.set('score', String(safeScore));
    if (referral) params.set('ref', referral);
    url.pathname = url.pathname || '/';
    url.hash = `studio?${params.toString()}`;
    return url.toString();
  }
  url.pathname = `/c/${encodeURIComponent(slug)}`;
  url.hash = '';
  if (safeScore !== null) url.searchParams.set('score', String(safeScore));
  if (referral) url.searchParams.set('ref', referral);
  return url.toString();
}
