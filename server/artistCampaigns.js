'use strict';

const { Midi } = require('@tonejs/midi');

const CAMPAIGN_STATUSES = new Set(['draft', 'published', 'paused', 'archived']);
const MAX_TEXT = Object.freeze({
  artist: 100,
  title: 140,
  hook: 180,
  description: 500,
  rightsBasis: 240,
  rightsHolder: 120,
  verificationNotes: 400,
  referralCode: 32,
});
const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const NOTE_BASES = Object.freeze({ C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 });
const MAX_EXCERPT_NOTES = 20_000;

function boundedText(value, maximum) {
  return String(value ?? '')
    .trim()
    .replace(/[\u0000-\u001f\u007f]/g, '')
    .replace(/\s+/g, ' ')
    .slice(0, maximum);
}

function booleanValue(value, fallback = false) {
  if (typeof value === 'boolean') return value;
  if (value === undefined || value === null || value === '') return fallback;
  return ['true', '1', 'yes', 'on'].includes(String(value).trim().toLowerCase());
}

function boundedNumber(value, fallback, minimum, maximum) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(maximum, Math.max(minimum, number));
}

function campaignSlug(value, fallback = '') {
  const source = boundedText(value || fallback, 100)
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
    .replace(/-+$/g, '');
  return source;
}

function referralCode(value, slug = '') {
  const normalized = boundedText(value, MAX_TEXT.referralCode)
    .toUpperCase()
    .replace(/[^A-Z0-9_-]/g, '')
    .slice(0, MAX_TEXT.referralCode);
  if (normalized.length >= 3) return normalized;
  return `ARTIST-${String(slug || 'SONG').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 18) || 'SONG'}`;
}

function safeHttpUrl(value) {
  const text = boundedText(value, 500);
  if (!text) return '';
  try {
    const url = new URL(text);
    return ['http:', 'https:'].includes(url.protocol) ? url.toString().slice(0, 500) : '';
  } catch {
    return '';
  }
}

function invalidExcerpt(message) {
  return Object.assign(new Error(message), { code: 'CAMPAIGN_EXCERPT_INVALID', status: 400 });
}

function midiName(value) {
  const midi = Math.round(Number(value));
  if (!Number.isFinite(midi) || midi < 0 || midi > 127) return '';
  return `${PITCH_CLASSES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function noteMidi(event = {}) {
  for (const candidate of [event.midi, event.pitch, event.noteNumber]) {
    const numeric = Number(candidate);
    if (Number.isFinite(numeric) && numeric >= 0 && numeric <= 127) return Math.round(numeric);
  }
  const rawName = String(event.note ?? event.name ?? event.pitch ?? '').trim();
  if (/^\d{1,3}$/.test(rawName)) {
    const numeric = Number(rawName);
    return numeric >= 0 && numeric <= 127 ? numeric : null;
  }
  const match = rawName.match(/^([A-Ga-g])([#b]?)(-?\d{1,2})$/);
  if (!match) return null;
  const base = NOTE_BASES[match[1].toUpperCase()];
  const accidental = match[2] === '#' ? 1 : match[2] === 'b' ? -1 : 0;
  const numeric = (Number(match[3]) + 1) * 12 + base + accidental;
  return numeric >= 0 && numeric <= 127 ? numeric : null;
}

function eventTime(event = {}) {
  return Number(event.time ?? event.start ?? event.startTime);
}

function eventDuration(event = {}) {
  const direct = Number(event.duration ?? event.scoreDuration ?? event.visualDuration ?? event.audioDuration);
  if (Number.isFinite(direct) && direct > 0) return direct;
  const start = eventTime(event);
  const end = Number(event.end ?? event.endTime);
  return Number.isFinite(start) && Number.isFinite(end) && end > start ? end - start : 0.45;
}

function clippedDuration(value, fallback, eventStart, clippedStart, previewEnd) {
  const candidate = Number(value);
  const duration = Number.isFinite(candidate) && candidate > 0 ? candidate : fallback;
  return Math.max(0.035, Math.min(previewEnd, eventStart + duration) - clippedStart);
}

function safeTimeSignature(value) {
  const numerator = Number(Array.isArray(value) ? value[0] : value?.numerator);
  const denominator = Number(Array.isArray(value) ? value[1] : value?.denominator);
  if (!Number.isInteger(numerator) || numerator < 1 || numerator > 32
      || !Number.isInteger(denominator) || denominator < 1 || denominator > 32) return undefined;
  return { numerator, denominator };
}

function jsonCampaignSource(bytes) {
  let parsed;
  try {
    parsed = JSON.parse(bytes.toString('utf8'));
  } catch {
    throw invalidExcerpt('The campaign JSON could not be parsed.');
  }
  const notes = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.notes)
      ? parsed.notes
      : Array.isArray(parsed?.events)
        ? parsed.events
        : [];
  return {
    bpm: Number(parsed?.bpm ?? parsed?.tempo),
    timeSignature: safeTimeSignature(parsed?.timeSignature),
    instrument: boundedText(parsed?.instrument || 'piano', 64),
    notes,
    pedals: Array.isArray(parsed?.pedals)
      ? parsed.pedals
      : (Array.isArray(parsed?.pedalEvents) ? parsed.pedalEvents : []),
  };
}

function midiCampaignSource(bytes) {
  let midi;
  try {
    midi = new Midi(bytes);
  } catch {
    throw invalidExcerpt('The campaign MIDI could not be parsed.');
  }
  const notes = [];
  const pedals = [];
  midi.tracks.forEach((track, trackIndex) => {
    const instrument = boundedText(track.instrument?.name || 'piano', 64);
    track.notes.forEach((note) => notes.push({
      midi: note.midi,
      time: note.time,
      duration: note.duration,
      velocity: note.velocity,
      channel: track.channel,
      track: trackIndex,
      instrument,
    }));
    const sustain = track.controlChanges?.[64] || [];
    sustain.forEach((change) => pedals.push({
      time: change.time,
      down: Number(change.value) >= 0.5,
      value: Math.round(Math.max(0, Math.min(1, Number(change.value) || 0)) * 127),
    }));
  });
  return {
    bpm: Number(midi.header.tempos?.[0]?.bpm),
    timeSignature: safeTimeSignature(midi.header.timeSignatures?.[0]?.timeSignature),
    instrument: 'piano',
    notes,
    pedals,
  };
}

function publicNote(event, index, previewStart, previewEnd) {
  const start = eventTime(event);
  const duration = eventDuration(event);
  const end = start + duration;
  const midi = noteMidi(event);
  if (!Number.isFinite(start) || !Number.isFinite(duration) || duration <= 0
      || midi === null || start >= previewEnd || end <= previewStart) return null;
  const clippedStart = Math.max(previewStart, start);
  const clippedEnd = Math.min(previewEnd, end);
  let velocity = Number(event.velocity);
  if (!Number.isFinite(velocity)) velocity = 0.78;
  if (velocity > 1) velocity /= 127;
  const result = {
    id: `campaign-note-${index}`,
    note: midiName(midi),
    midi,
    time: Number((clippedStart - previewStart).toFixed(4)),
    duration: Number(Math.max(0.035, clippedEnd - clippedStart).toFixed(4)),
    visualDuration: Number(clippedDuration(
      event.visualDuration,
      duration,
      start,
      clippedStart,
      previewEnd,
    ).toFixed(4)),
    audioDuration: Number(clippedDuration(
      event.audioDuration,
      duration,
      start,
      clippedStart,
      previewEnd,
    ).toFixed(4)),
    velocity: Number(Math.max(0.02, Math.min(1, velocity)).toFixed(4)),
  };
  const hand = String(event.hand || '').trim().toLowerCase();
  if (['left', 'right', 'both'].includes(hand)) result.hand = hand;
  for (const key of ['instrument', 'scoreRole', 'voice', 'articulation']) {
    const value = boundedText(event[key], 64);
    if (value) result[key] = value;
  }
  for (const key of ['channel', 'track', 'program']) {
    const value = Number(event[key]);
    if (Number.isInteger(value) && value >= 0 && value <= 255) result[key] = value;
  }
  return result;
}

function publicPedals(events, previewStart, previewEnd) {
  const normalized = events.flatMap((event) => {
    const time = eventTime(event);
    if (!Number.isFinite(time)) return [];
    const numericValue = Number(event.value);
    return [{
      time,
      down: typeof event.down === 'boolean' ? event.down : numericValue >= 0.5,
      value: Number.isFinite(numericValue)
        ? Math.round(Math.max(0, Math.min(numericValue > 1 ? 127 : 1, numericValue)) * (numericValue > 1 ? 1 : 127))
        : (event.down ? 127 : 0),
    }];
  }).sort((left, right) => left.time - right.time);
  const prior = normalized.filter((pedal) => pedal.time <= previewStart).at(-1);
  const clipped = normalized
    .filter((pedal) => pedal.time > previewStart && pedal.time < previewEnd)
    .map((pedal) => ({ ...pedal, time: Number((pedal.time - previewStart).toFixed(4)) }));
  return [...(prior?.down ? [{ ...prior, time: 0 }] : []), ...clipped];
}

function buildCampaignExcerpt(bytes, format, campaign = {}) {
  const source = format === 'MIDI'
    ? midiCampaignSource(bytes)
    : format === 'JSON'
      ? jsonCampaignSource(bytes)
      : null;
  if (!source) throw invalidExcerpt('Campaign excerpts support ready-to-play JSON or MIDI only.');
  const previewStart = Math.max(0, Number(campaign.previewStartSeconds) || 0);
  const previewDuration = Math.max(10, Math.min(45, Number(campaign.previewDurationSeconds) || 20));
  const previewEnd = previewStart + previewDuration;
  const notes = source.notes
    .map((event, index) => publicNote(event, index, previewStart, previewEnd))
    .filter(Boolean)
    .sort((left, right) => left.time - right.time || left.midi - right.midi);
  if (!notes.length) throw invalidExcerpt('The selected campaign window contains no playable notes. Change the preview start time or upload another file.');
  if (notes.length > MAX_EXCERPT_NOTES) {
    throw invalidExcerpt(`The selected campaign window exceeds ${MAX_EXCERPT_NOTES.toLocaleString('en-US')} notes.`);
  }
  const bpm = Number.isFinite(source.bpm) ? Math.max(20, Math.min(400, source.bpm)) : 120;
  const output = {
    readyToPlayFormat: 'polymath-artist-campaign-v1',
    sourceType: 'artist-campaign-excerpt',
    title: boundedText(campaign.title, MAX_TEXT.title) || 'Artist challenge',
    artist: boundedText(campaign.artist, MAX_TEXT.artist),
    composer: boundedText(campaign.artist, MAX_TEXT.artist),
    instrument: source.instrument || 'piano',
    bpm: Number(bpm.toFixed(3)),
    ...(source.timeSignature ? { timeSignature: source.timeSignature } : {}),
    duration: previewDuration,
    notes,
    pedals: publicPedals(source.pedals, previewStart, previewEnd),
    performance: {
      profile: 'polymath-artist-challenge-v1',
      preserveScoreDurations: true,
      preserveScoreTiming: true,
    },
    campaignId: boundedText(campaign.id, 100),
    campaignSlug: campaignSlug(campaign.slug),
  };
  const buffer = Buffer.from(JSON.stringify(output));
  return { buffer, noteCount: notes.length, durationSeconds: previewDuration };
}

function optionalIsoDate(value, fallback = '') {
  const text = String(value ?? '').trim();
  if (!text) return fallback || '';
  const timestamp = Date.parse(text);
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : fallback || '';
}

function normalizeCampaignInput(input = {}, current = {}) {
  const artist = boundedText(input.artist ?? current.artist, MAX_TEXT.artist);
  const title = boundedText(input.title ?? current.title, MAX_TEXT.title);
  const slug = campaignSlug(input.slug ?? current.slug, `${artist}-${title}`);
  const statusCandidate = String(input.status ?? current.status ?? 'draft').trim().toLowerCase();
  const previewStartSeconds = boundedNumber(
    input.previewStartSeconds ?? current.previewStartSeconds,
    Number(current.previewStartSeconds) || 0,
    0,
    60 * 60,
  );
  const previewDurationSeconds = boundedNumber(
    input.previewDurationSeconds ?? current.previewDurationSeconds,
    Number(current.previewDurationSeconds) || 20,
    10,
    45,
  );
  return {
    artist,
    title,
    slug,
    hook: boundedText(
      input.hook ?? current.hook,
      MAX_TEXT.hook,
    ) || `Can you play ${title || 'this song'}?`,
    description: boundedText(input.description ?? current.description, MAX_TEXT.description),
    artistUrl: safeHttpUrl(input.artistUrl ?? current.artistUrl),
    status: CAMPAIGN_STATUSES.has(statusCandidate) ? statusCandidate : 'draft',
    referralCode: referralCode(input.referralCode ?? current.referralCode, slug),
    affiliatePercent: boundedNumber(
      input.affiliatePercent ?? current.affiliatePercent,
      Number(current.affiliatePercent) || 20,
      0,
      50,
    ),
    previewStartSeconds,
    previewDurationSeconds,
    challengeScore: Math.round(boundedNumber(
      input.challengeScore ?? current.challengeScore,
      Number(current.challengeScore) || 0,
      0,
      100,
    )),
    qaScore: Math.round(boundedNumber(
      input.qaScore ?? current.qaScore,
      Number(current.qaScore) || 0,
      0,
      100,
    )),
    rightsConfirmed: booleanValue(input.rightsConfirmed, Boolean(current.rightsConfirmed)),
    artistApproved: booleanValue(input.artistApproved, Boolean(current.artistApproved)),
    humanVerified: booleanValue(input.humanVerified, Boolean(current.humanVerified)),
    rightsHolder: boundedText(input.rightsHolder ?? current.rightsHolder, MAX_TEXT.rightsHolder),
    rightsBasis: boundedText(input.rightsBasis ?? current.rightsBasis, MAX_TEXT.rightsBasis),
    verificationNotes: boundedText(
      input.verificationNotes ?? current.verificationNotes,
      MAX_TEXT.verificationNotes,
    ),
    launchesAt: optionalIsoDate(input.launchesAt, current.launchesAt),
    endsAt: optionalIsoDate(input.endsAt, current.endsAt),
  };
}

function campaignPublishProblems(campaign = {}) {
  const problems = [];
  if (!campaign.artist) problems.push('Artist name is required.');
  if (!campaign.title) problems.push('Song title is required.');
  if (!campaign.slug) problems.push('A public campaign slug is required.');
  if (!campaign.songAssetKey || !campaign.songFilename) problems.push('A playable JSON or MIDI challenge file is required.');
  if (campaign.songAssetKey && (!campaign.publicSongAssetKey || Number(campaign.excerptNoteCount || 0) < 1)) {
    problems.push('The approved public excerpt must contain playable notes.');
  }
  if (!campaign.rightsConfirmed) problems.push('Music rights must be confirmed.');
  if (!campaign.rightsHolder) problems.push('Record the rights holder or approving artist.');
  if (!campaign.rightsBasis) problems.push('Record the permission or licence basis.');
  if (!campaign.artistApproved) problems.push('The artist must approve the public challenge.');
  if (!campaign.humanVerified) problems.push('A human pianist must verify the challenge.');
  if (!campaign.verificationNotes) problems.push('Record what the human verifier checked.');
  if (Number(campaign.qaScore || 0) < 80) problems.push('QA score must be at least 80/100.');
  const start = Number(campaign.previewStartSeconds);
  const duration = Number(campaign.previewDurationSeconds);
  if (!Number.isFinite(start) || start < 0) problems.push('Preview start time is invalid.');
  if (!Number.isFinite(duration) || duration < 10 || duration > 45) {
    problems.push('Public challenges must be between 10 and 45 seconds.');
  }
  if (campaign.launchesAt && campaign.endsAt
      && Date.parse(campaign.endsAt) <= Date.parse(campaign.launchesAt)) {
    problems.push('Campaign end must be later than its launch.');
  }
  return problems;
}

function campaignIsLive(campaign, now = Date.now()) {
  if (!campaign || campaign.status !== 'published' || campaignPublishProblems(campaign).length) return false;
  const launchesAt = Date.parse(campaign.launchesAt || '');
  const endsAt = Date.parse(campaign.endsAt || '');
  if (Number.isFinite(launchesAt) && launchesAt > now) return false;
  if (Number.isFinite(endsAt) && endsAt <= now) return false;
  return true;
}

function publicArtistCampaign(campaign, { now = Date.now() } = {}) {
  const slug = campaign.slug;
  return {
    id: campaign.id,
    slug,
    artist: campaign.artist,
    title: campaign.title,
    hook: campaign.hook,
    description: campaign.description || '',
    artistUrl: campaign.artistUrl || '',
    challengeScore: Number(campaign.challengeScore || 0),
    preview: {
      // The public artifact is generated as a bounded excerpt and shifted to
      // zero. The private source offset never needs to reach the browser.
      startSeconds: 0,
      durationSeconds: Number(campaign.previewDurationSeconds || 20),
    },
    verification: {
      humanVerified: Boolean(campaign.humanVerified),
      qaScore: Number(campaign.qaScore || 0),
      label: campaign.humanVerified ? 'Human verified' : '',
    },
    referralCode: campaign.referralCode || '',
    launchesAt: campaign.launchesAt || null,
    endsAt: campaign.endsAt || null,
    live: campaignIsLive(campaign, now),
    songFilename: `${slug || 'artist'}-challenge.json`,
    songFormat: 'JSON',
    songUrl: `/api/artist-campaigns/${encodeURIComponent(slug)}/song`,
    coverUrl: campaign.coverAssetKey
      ? `/api/artist-campaigns/${encodeURIComponent(slug)}/cover`
      : '',
    sharePath: `/c/${encodeURIComponent(slug)}`,
  };
}

function adminArtistCampaign(campaign, options = {}) {
  return {
    ...publicArtistCampaign(campaign, options),
    status: campaign.status,
    affiliatePercent: Number(campaign.affiliatePercent || 0),
    previewStartSeconds: Number(campaign.previewStartSeconds || 0),
    previewDurationSeconds: Number(campaign.previewDurationSeconds || 20),
    rightsConfirmed: Boolean(campaign.rightsConfirmed),
    artistApproved: Boolean(campaign.artistApproved),
    rightsHolder: campaign.rightsHolder || '',
    rightsBasis: campaign.rightsBasis || '',
    verificationNotes: campaign.verificationNotes || '',
    uploadedSongFilename: campaign.songFilename || '',
    uploadedSongFormat: campaign.songFormat || '',
    songSize: Number(campaign.songSize || 0),
    excerptNoteCount: Number(campaign.excerptNoteCount || 0),
    publicSongSize: Number(campaign.publicSongSize || 0),
    coverContentType: campaign.coverContentType || '',
    publishProblems: campaignPublishProblems(campaign),
    createdAt: campaign.createdAt,
    updatedAt: campaign.updatedAt,
    publishedAt: campaign.publishedAt || null,
  };
}

function campaignAttribution(db, input = {}, { now = Date.now() } = {}) {
  const campaigns = Array.isArray(db?.artistCampaigns) ? db.artistCampaigns : [];
  const requestedId = boundedText(input.campaignId, 100);
  const requestedSlug = campaignSlug(input.campaignSlug || input.campaign);
  const campaign = campaigns.find((candidate) => (
    (requestedId && candidate.id === requestedId)
    || (requestedSlug && candidate.slug === requestedSlug)
  ));
  if (!campaign || !campaignIsLive(campaign, now)) return {};
  const rawReferral = boundedText(input.referralCode, MAX_TEXT.referralCode);
  const requestedReferral = rawReferral ? referralCode(rawReferral, '') : '';
  return {
    campaignId: campaign.id,
    campaignSlug: campaign.slug,
    referralCode: requestedReferral && requestedReferral === campaign.referralCode
      ? campaign.referralCode
      : '',
  };
}

module.exports = {
  CAMPAIGN_STATUSES,
  adminArtistCampaign,
  buildCampaignExcerpt,
  campaignAttribution,
  campaignIsLive,
  campaignPublishProblems,
  campaignSlug,
  normalizeCampaignInput,
  publicArtistCampaign,
  referralCode,
};
