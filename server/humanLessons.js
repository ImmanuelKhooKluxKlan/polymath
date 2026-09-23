const crypto = require('crypto');

const LESSON_DURATION_STEP_MINUTES = 10;
const LESSON_COST_PER_STEP_MCOINS = 0.2;
const BREAKOUT_ROOM_COST_MCOINS = 0.5;
const MINIMUM_CHAT_TRANSFER_MCOINS = 30;
const MAXIMUM_LESSON_DURATION_MINUTES = 12 * 60;
const MAXIMUM_CHAT_TRANSFER_MCOINS = 1_000_000;
const PARTICIPANT_ACTIVE_WINDOW_MS = 30_000;
const SIGNAL_RETENTION_MS = 5 * 60_000;

const ACCESS_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

class HumanLessonError extends Error {
  constructor(message, status = 400, code = 'HUMAN_LESSON_INVALID') {
    super(message);
    this.name = 'HumanLessonError';
    this.status = status;
    this.code = code;
  }
}

function roundMcoins(value) {
  return Number((Number(value) || 0).toFixed(2));
}

function normalizeLessonDuration(value) {
  const minutes = Number(value);
  if (!Number.isInteger(minutes)
      || minutes < LESSON_DURATION_STEP_MINUTES
      || minutes > MAXIMUM_LESSON_DURATION_MINUTES
      || minutes % LESSON_DURATION_STEP_MINUTES !== 0) {
    throw new HumanLessonError(
      `Lesson duration must be from ${LESSON_DURATION_STEP_MINUTES} to ${MAXIMUM_LESSON_DURATION_MINUTES} minutes in ${LESSON_DURATION_STEP_MINUTES}-minute intervals.`,
      400,
      'INVALID_LESSON_DURATION',
    );
  }
  return minutes;
}

function lessonCallCost(durationMinutes) {
  const minutes = normalizeLessonDuration(durationMinutes);
  return roundMcoins((minutes / LESSON_DURATION_STEP_MINUTES) * LESSON_COST_PER_STEP_MCOINS);
}

function normalizeTransferAmount(value) {
  const amount = roundMcoins(value);
  if (!Number.isFinite(Number(value))
      || amount < MINIMUM_CHAT_TRANSFER_MCOINS
      || amount > MAXIMUM_CHAT_TRANSFER_MCOINS) {
    throw new HumanLessonError(
      `Transfers must be from ${MINIMUM_CHAT_TRANSFER_MCOINS.toLocaleString()} to ${MAXIMUM_CHAT_TRANSFER_MCOINS.toLocaleString()} Mcoins.`,
      400,
      'INVALID_TRANSFER_AMOUNT',
    );
  }
  return amount;
}

function randomCharacters(length) {
  let value = '';
  while (value.length < length) {
    const byte = crypto.randomBytes(1)[0];
    if (byte >= 256 - (256 % ACCESS_CODE_ALPHABET.length)) continue;
    value += ACCESS_CODE_ALPHABET[byte % ACCESS_CODE_ALPHABET.length];
  }
  return value;
}

function createMeetingCode(existingCodes = new Set()) {
  for (let attempt = 0; attempt < 25; attempt += 1) {
    const code = `PM-${randomCharacters(4)}-${randomCharacters(4)}`;
    if (!existingCodes.has(code)) return code;
  }
  throw new HumanLessonError('A private meeting ID could not be generated. Try again.', 503, 'MEETING_ID_UNAVAILABLE');
}

function createAccessCode() {
  return `${randomCharacters(4)}-${randomCharacters(4)}`;
}

function hashAccessCode(accessCode, salt = crypto.randomBytes(16).toString('hex')) {
  const normalized = String(accessCode || '').trim().toUpperCase();
  return {
    salt,
    hash: crypto.scryptSync(normalized, salt, 64).toString('hex'),
  };
}

function accessCodeMatches(accessCode, salt, expectedHash) {
  if (!salt || !expectedHash) return false;
  const attempted = hashAccessCode(accessCode, salt).hash;
  const attemptedBytes = Buffer.from(attempted, 'hex');
  const expectedBytes = Buffer.from(expectedHash, 'hex');
  return attemptedBytes.length === expectedBytes.length
    && crypto.timingSafeEqual(attemptedBytes, expectedBytes);
}

function credentialEncryptionKey(secret) {
  const material = String(secret || '').trim();
  if (!material) throw new HumanLessonError('Private lesson credentials are not configured.', 503, 'LESSON_SECRET_MISSING');
  return crypto.createHash('sha256').update(material).digest();
}

function sealAccessCode(accessCode, secret) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', credentialEncryptionKey(secret), iv);
  const ciphertext = Buffer.concat([cipher.update(String(accessCode), 'utf8'), cipher.final()]);
  return ['v1', iv.toString('base64url'), cipher.getAuthTag().toString('base64url'), ciphertext.toString('base64url')].join('.');
}

function openAccessCode(sealed, secret) {
  try {
    const [version, iv, tag, ciphertext] = String(sealed || '').split('.');
    if (version !== 'v1' || !iv || !tag || !ciphertext) throw new Error('Invalid credential envelope.');
    const decipher = crypto.createDecipheriv(
      'aes-256-gcm',
      credentialEncryptionKey(secret),
      Buffer.from(iv, 'base64url'),
    );
    decipher.setAuthTag(Buffer.from(tag, 'base64url'));
    return Buffer.concat([
      decipher.update(Buffer.from(ciphertext, 'base64url')),
      decipher.final(),
    ]).toString('utf8');
  } catch (error) {
    if (error instanceof HumanLessonError) throw error;
    throw new HumanLessonError('This lesson access code can no longer be opened. Create a new lesson.', 409, 'LESSON_CREDENTIAL_UNREADABLE');
  }
}

function cleanLessonTitle(value) {
  return String(value || '')
    .replace(/[<>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 100) || 'Private music lesson';
}

function cleanRoomName(value, fallback = 'Breakout room') {
  return String(value || '')
    .replace(/[<>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 60) || fallback;
}

function normalizeScheduledFor(value, now = Date.now()) {
  if (!value) return new Date(now).toISOString();
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) {
    throw new HumanLessonError('Choose a valid lesson date and time.', 400, 'INVALID_LESSON_TIME');
  }
  const latest = now + (366 * 24 * 60 * 60 * 1000);
  if (timestamp < now - (10 * 60 * 1000) || timestamp > latest) {
    throw new HumanLessonError('Lessons can be scheduled from now through the next 12 months.', 400, 'INVALID_LESSON_TIME');
  }
  return new Date(timestamp).toISOString();
}

function effectiveMeetingStatus(meeting, now = Date.now()) {
  if (!meeting) return 'missing';
  if (['ended', 'cancelled'].includes(meeting.status)) return meeting.status;
  const expiresAt = new Date(meeting.expiresAt || 0).getTime();
  if (expiresAt && expiresAt <= now) return 'ended';
  return meeting.status || 'ready';
}

function meetingExpiresAt(meeting, parentMeeting = null) {
  if (parentMeeting?.expiresAt) return parentMeeting.expiresAt;
  if (!meeting?.startedAt) return null;
  return new Date(
    new Date(meeting.startedAt).getTime() + (Number(meeting.durationMinutes || 0) * 60_000),
  ).toISOString();
}

function participantIsActive(participant, now = Date.now()) {
  if (!participant || participant.leftAt) return false;
  return now - new Date(participant.lastSeenAt || participant.joinedAt || 0).getTime()
    <= PARTICIPANT_ACTIVE_WINDOW_MS;
}

function trimLessonSignals(signals, now = Date.now()) {
  const cutoff = now - SIGNAL_RETENTION_MS;
  return (Array.isArray(signals) ? signals : []).filter(
    (signal) => new Date(signal.createdAt || 0).getTime() >= cutoff,
  );
}

function normalizeClientRequestId(value) {
  return String(value || '').trim().replace(/[^a-zA-Z0-9:_-]/g, '').slice(0, 100);
}

module.exports = {
  ACCESS_CODE_ALPHABET,
  BREAKOUT_ROOM_COST_MCOINS,
  HumanLessonError,
  LESSON_COST_PER_STEP_MCOINS,
  LESSON_DURATION_STEP_MINUTES,
  MAXIMUM_CHAT_TRANSFER_MCOINS,
  MAXIMUM_LESSON_DURATION_MINUTES,
  MINIMUM_CHAT_TRANSFER_MCOINS,
  PARTICIPANT_ACTIVE_WINDOW_MS,
  SIGNAL_RETENTION_MS,
  accessCodeMatches,
  cleanLessonTitle,
  cleanRoomName,
  createAccessCode,
  createMeetingCode,
  effectiveMeetingStatus,
  hashAccessCode,
  lessonCallCost,
  meetingExpiresAt,
  normalizeClientRequestId,
  normalizeLessonDuration,
  normalizeScheduledFor,
  normalizeTransferAmount,
  openAccessCode,
  participantIsActive,
  roundMcoins,
  sealAccessCode,
  trimLessonSignals,
};
