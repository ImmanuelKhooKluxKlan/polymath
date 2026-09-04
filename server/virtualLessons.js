'use strict';

const LESSON_RATE_MCOINS_PER_HOUR = 10;
const LESSON_DURATION_STEP_MINUTES = 30;
const LESSON_MIN_DURATION_MINUTES = 30;
const LESSON_MAX_DURATION_MINUTES = 12 * 60;
const LESSON_DEFAULT_DURATION_MINUTES = 60;
const DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS = 5;
const CONVERSATION_MODES = Object.freeze(['music-coach', 'adult-companion']);
const COMPANION_STYLES = Object.freeze(['gentle', 'playful', 'confident']);
const MAX_SESSION_MESSAGES = 80;
const MAX_SESSION_MESSAGE_CHARS = 1600;
const MAX_DEMONSTRATION_SECONDS = 30;

function cleanText(value, maximum = MAX_SESSION_MESSAGE_CHARS) {
  return String(value || '').trim().replace(/\s+/g, ' ').slice(0, maximum);
}

function roundMcoins(value) {
  return Number((Math.round(Number(value) * 100) / 100).toFixed(2));
}

function normalizeLessonBlockPrice(value = DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS) {
  const configured = Number(value);
  return roundMcoins(Number.isFinite(configured)
    ? Math.min(1000000000, Math.max(0, configured))
    : DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS);
}

function normalizeLessonDuration(durationMinutes) {
  const requested = Number(durationMinutes);
  if (!Number.isFinite(requested)) return null;
  const rounded = Math.round(requested / LESSON_DURATION_STEP_MINUTES) * LESSON_DURATION_STEP_MINUTES;
  return Math.min(LESSON_MAX_DURATION_MINUTES, Math.max(LESSON_MIN_DURATION_MINUTES, rounded));
}

function lessonQuote(
  durationMinutes,
  pricePer30MinutesMcoins = DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
) {
  const requestedDurationMinutes = Number(durationMinutes);
  const minutes = normalizeLessonDuration(requestedDurationMinutes);
  if (!minutes) return null;
  const pricePerBlockMcoins = normalizeLessonBlockPrice(pricePer30MinutesMcoins);
  const priceMcoins = roundMcoins((minutes / LESSON_DURATION_STEP_MINUTES) * pricePerBlockMcoins);
  return Object.freeze({
    requestedDurationMinutes,
    durationMinutes: minutes,
    rounded: requestedDurationMinutes !== minutes,
    durationStepMinutes: LESSON_DURATION_STEP_MINUTES,
    pricePer30MinutesMcoins: pricePerBlockMcoins,
    priceMcoins,
    priceUsd: priceMcoins,
  });
}

function lessonCatalog(pricePer30MinutesMcoins = DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS) {
  const blockPrice = normalizeLessonBlockPrice(pricePer30MinutesMcoins);
  return {
    rateMcoinsPerHour: roundMcoins(blockPrice * 2),
    pricePer30MinutesMcoins: blockPrice,
    durationStepMinutes: LESSON_DURATION_STEP_MINUTES,
    minimumDurationMinutes: LESSON_MIN_DURATION_MINUTES,
    maximumDurationMinutes: LESSON_MAX_DURATION_MINUTES,
    defaultDurationMinutes: LESSON_DEFAULT_DURATION_MINUTES,
    mcoinsPerUsd: 1,
    memoryPolicy: 'session-only',
    conversationModes: [
      {
        id: 'music-coach',
        label: 'Music teacher',
        description: 'Music-first teaching with normal conversation welcome.',
      },
      {
        id: 'adult-companion',
        label: 'Flirty companion',
        description: 'Optional 18+ AI companion roleplay for eligible characters.',
        requiresAdultConfirmation: true,
      },
    ],
  };
}

function normalizeConversationMode(value) {
  const normalized = cleanText(value, 40).toLowerCase();
  return CONVERSATION_MODES.includes(normalized) ? normalized : 'music-coach';
}

function sanitizeConversationPreferences(value = {}) {
  const companionStyle = cleanText(value?.companionStyle, 30).toLowerCase();
  return {
    preferredName: cleanText(value?.preferredName, 60),
    companionStyle: COMPANION_STYLES.includes(companionStyle) ? companionStyle : 'playful',
  };
}

function sanitizeTeacher(teacher) {
  const requestedMinimumAge = Number(teacher?.minimumAge);
  const minimumAge = Number.isFinite(requestedMinimumAge)
    ? Math.min(99, Math.max(0, Math.floor(requestedMinimumAge)))
    : (teacher?.requiresAdultConfirmation ? 18 : 0);
  const requestedPrice = teacher?.pricePer30MinutesMcoins;
  const pricePer30MinutesMcoins = requestedPrice === null
    || requestedPrice === undefined
    || String(requestedPrice).trim() === ''
    ? null
    : normalizeLessonBlockPrice(requestedPrice);
  return {
    id: cleanText(teacher?.id, 64).replace(/[^a-z0-9_-]/gi, '') || 'aria',
    name: cleanText(teacher?.name, 80) || 'Aria',
    title: cleanText(teacher?.title, 120) || 'Polymath piano teacher',
    style: cleanText(teacher?.style, 280) || 'Clear, patient, and precise',
    voice: cleanText(teacher?.voice, 100) || 'Natural and expressive',
    voiceType: ['feminine', 'masculine'].includes(cleanText(teacher?.voiceType, 20).toLowerCase())
      ? cleanText(teacher?.voiceType, 20).toLowerCase()
      : 'neutral',
    minimumAge,
    requiresAdultConfirmation: minimumAge >= 18,
    adultCompanionEnabled: Boolean(teacher?.adultCompanionEnabled),
    pricePer30MinutesMcoins,
  };
}

function normalizeClientRequestId(value) {
  const candidate = cleanText(value, 96);
  return /^[a-z0-9_-]{16,96}$/i.test(candidate) ? candidate : '';
}

function sessionIsActive(session, now = new Date()) {
  return Boolean(
    session
    && session.status === 'active'
    && Number.isFinite(new Date(session.expiresAt).getTime())
    && new Date(session.expiresAt).getTime() > now.getTime(),
  );
}

function expireVirtualLessons(db, now = new Date()) {
  if (!Array.isArray(db?.virtualLessonSessions)) return false;
  let changed = false;
  db.virtualLessonSessions.forEach((session) => {
    if (session.status !== 'active') return;
    if (new Date(session.expiresAt).getTime() > now.getTime()) return;
    session.status = 'expired';
    session.endedAt = session.expiresAt || now.toISOString();
    session.messages = [];
    session.memory = null;
    session.memoryClearedAt = now.toISOString();
    changed = true;
  });
  return changed;
}

function activeVirtualLesson(db, userId, now = new Date()) {
  expireVirtualLessons(db, now);
  return (db?.virtualLessonSessions || [])
    .filter((session) => session.userId === userId && sessionIsActive(session, now))
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))[0] || null;
}

function publicVirtualLesson(session, now = new Date()) {
  if (!session) return null;
  const active = sessionIsActive(session, now);
  const remainingSeconds = active
    ? Math.max(0, Math.ceil((new Date(session.expiresAt).getTime() - now.getTime()) / 1000))
    : 0;
  return {
    id: session.id,
    status: active ? 'active' : session.status,
    durationMinutes: Number(session.durationMinutes),
    priceMcoins: Number(session.priceMcoins),
    teacher: sanitizeTeacher(session.teacher),
    conversationMode: normalizeConversationMode(session.conversationMode),
    conversationPreferences: sanitizeConversationPreferences(session.conversationPreferences),
    adultCompanionConfirmed: normalizeConversationMode(session.conversationMode) === 'adult-companion'
      && Boolean(session.adultConfirmedAt)
      && Boolean(session.companionConsentAt),
    startedAt: session.startedAt,
    expiresAt: session.expiresAt,
    remainingSeconds,
    messages: active
      ? (session.messages || []).map((message) => ({
        id: message.id,
        role: message.role,
        text: cleanText(message.text),
        createdAt: message.createdAt,
      }))
      : [],
    memory: active && session.memory ? {
      goal: cleanText(session.memory.goal, 240),
      preferences: Array.isArray(session.memory.preferences)
        ? session.memory.preferences.map((item) => cleanText(item, 180)).filter(Boolean).slice(-4)
        : [],
      lastSong: cleanText(session.memory.lastSong, 160),
      lastFocus: cleanText(session.memory.lastFocus, 120),
      preferredName: cleanText(session.memory.preferredName, 60),
      instruments: Array.isArray(session.memory.instruments)
        ? session.memory.instruments.map((item) => cleanText(item, 80)).filter(Boolean).slice(-4)
        : [],
      genres: Array.isArray(session.memory.genres)
        ? session.memory.genres.map((item) => cleanText(item, 80)).filter(Boolean).slice(-4)
        : [],
      interests: Array.isArray(session.memory.interests)
        ? session.memory.interests.map((item) => cleanText(item, 120)).filter(Boolean).slice(-4)
        : [],
    } : null,
  };
}

function createVirtualLesson({
  id,
  userId,
  clientRequestId,
  durationMinutes,
  priceMcoins,
  teacher,
  conversationMode = 'music-coach',
  conversationPreferences = {},
  adultConfirmed = false,
  companionConsent = false,
  now = new Date(),
  studentName = '',
}) {
  const quote = lessonQuote(durationMinutes);
  if (!quote) throw new Error('Enter a valid virtual lesson duration.');
  const mode = normalizeConversationMode(conversationMode);
  if (mode === 'adult-companion' && (!adultConfirmed || !companionConsent)) {
    throw new Error('Adult companion mode requires both 18+ confirmation and explicit companion consent.');
  }
  const startedAt = now.toISOString();
  const preferences = sanitizeConversationPreferences(conversationPreferences);
  return {
    id,
    userId,
    clientRequestId: normalizeClientRequestId(clientRequestId),
    status: 'active',
    durationMinutes: quote.durationMinutes,
    priceMcoins: roundMcoins(priceMcoins),
    teacher: sanitizeTeacher(teacher),
    conversationMode: mode,
    conversationPreferences: preferences,
    adultConfirmedAt: mode === 'adult-companion' ? startedAt : null,
    companionConsentAt: mode === 'adult-companion' ? startedAt : null,
    startedAt,
    expiresAt: new Date(now.getTime() + quote.durationMinutes * 60 * 1000).toISOString(),
    createdAt: startedAt,
    lastActivityAt: startedAt,
    messages: [],
    memory: {
      studentName: cleanText(studentName, 80),
      preferredName: preferences.preferredName,
      goal: '',
      preferences: [],
      instruments: [],
      genres: [],
      interests: [],
      lastSong: '',
      lastFocus: '',
      lastDemonstration: null,
    },
  };
}

function appendSessionMessage(session, message, now = new Date()) {
  if (!sessionIsActive(session, now)) throw new Error('This virtual lesson has ended.');
  const text = cleanText(message?.text);
  if (!text) throw new Error('Say or type a message first.');
  const role = message?.role === 'assistant' ? 'assistant' : 'user';
  session.messages = [
    ...(session.messages || []),
    {
      id: cleanText(message?.id, 96),
      role,
      text,
      createdAt: message?.createdAt || now.toISOString(),
    },
  ].slice(-MAX_SESSION_MESSAGES);
  session.lastActivityAt = message?.createdAt || now.toISOString();
  return session.messages[session.messages.length - 1];
}

function boundedSeconds(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : fallback;
}

function clockSeconds(token, unit = '') {
  const value = String(token || '').trim();
  if (/^\d{1,3}:\d{1,2}(?:\.\d+)?$/.test(value)) {
    const [minutes, seconds] = value.split(':').map(Number);
    return minutes * 60 + seconds;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return /^m(?:in(?:ute)?s?)?$/i.test(unit) ? number * 60 : number;
}

function clampDemonstration(start, end, songDuration) {
  const maximum = boundedSeconds(songDuration, Number.POSITIVE_INFINITY);
  const safeStart = Math.min(maximum, boundedSeconds(start));
  const safeEnd = Math.min(maximum, Math.max(safeStart + 0.5, boundedSeconds(end, safeStart + 5)));
  return {
    startSeconds: Number(safeStart.toFixed(3)),
    endSeconds: Number(Math.min(safeEnd, safeStart + MAX_DEMONSTRATION_SECONDS).toFixed(3)),
  };
}

function requestedHand(text) {
  if (/\bleft\s+hand\b/i.test(text)) return 'left';
  if (/\bright\s+hand\b/i.test(text)) return 'right';
  return 'both';
}

function parseTeacherDemonstration(message, lessonContext = {}, previousAction = null) {
  const text = cleanText(message).toLowerCase();
  const wantsDemonstration = /\b(show|demonstrate|play|watch|move|moving|hands?)\b/.test(text)
    && /\b(hand|hands|play|playing|move|notes?|keys?|seconds?|secs?|minutes?|mins?|part|passage|again|slowly|slower)\b/.test(text);
  if (!wantsDemonstration) return null;

  const songDuration = boundedSeconds(lessonContext?.duration ?? lessonContext?.songDuration, Number.POSITIVE_INFINITY);
  const currentTime = boundedSeconds(lessonContext?.currentTime);
  let start = null;
  let end = null;

  const range = text.match(/\b(?:from|between)\s+(\d{1,3}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)?\s*(?:to|until|and|-)\s*(\d{1,3}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)?/i);
  if (range) {
    start = clockSeconds(range[1], range[2]);
    end = clockSeconds(range[3], range[4] || range[2]);
  }

  if (start === null) {
    const first = text.match(/\bfirst\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b/i);
    if (first) {
      start = 0;
      end = clockSeconds(first[1], first[2]);
    }
  }

  if (start === null) {
    const next = text.match(/\bnext\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b/i);
    if (next) {
      start = currentTime;
      end = currentTime + clockSeconds(next[1], next[2]);
    }
  }

  if (start === null) {
    const at = text.match(/\bat\s+(\d{1,3}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)?\b/i);
    if (at) {
      start = clockSeconds(at[1], at[2]);
      end = start + 5;
    }
  }

  if (start === null && /\b(again|repeat that|slower)\b/.test(text) && previousAction) {
    start = boundedSeconds(previousAction.startSeconds);
    end = boundedSeconds(previousAction.endSeconds, start + 5);
  }

  if (start === null) {
    const activeRange = lessonContext?.activeRange;
    start = boundedSeconds(activeRange?.start, currentTime);
    end = boundedSeconds(activeRange?.end, start + 5);
  }

  const bounded = clampDemonstration(start, end, songDuration);
  return {
    type: 'demonstrate_range',
    ...bounded,
    hand: requestedHand(text),
    speed: /\b(slow|slowly|slower)\b/.test(text) ? 0.55 : null,
  };
}

function rememberUnique(items, value, maximum = 4) {
  const cleanValue = cleanText(value, 180);
  if (!cleanValue) return Array.isArray(items) ? items.slice(-maximum) : [];
  return [...(Array.isArray(items) ? items : []).filter((item) => item !== cleanValue), cleanValue].slice(-maximum);
}

function updateSessionMemory(session, {
  studentName = '',
  userMessage = '',
  lessonContext = {},
  practiceReport = null,
  action = null,
} = {}) {
  const memory = session.memory && typeof session.memory === 'object' ? session.memory : {};
  memory.studentName = cleanText(studentName || memory.studentName, 80);
  const message = cleanText(userMessage, 600);
  const preferredName = message.match(/\b(?:call me|you can call me|my name is)\s+([a-z][a-z '-]{0,40}?)(?=[.!?,]|$)/i);
  if (preferredName) memory.preferredName = cleanText(preferredName[1], 60).replace(/[.!?]+$/, '');
  const goal = message.match(/\b(?:my goal is|i want to|i need to|help me)\s+(.{3,220})/i);
  if (goal) memory.goal = cleanText(goal[1], 240);
  const preference = message.match(/\b(?:i prefer|i like|i struggle with|i find)\s+(.{3,170})/i);
  if (preference) memory.preferences = rememberUnique(memory.preferences, preference[0]);
  const instrument = message.match(/\b(?:i play|i am learning|i'm learning|i practise|i practice)\s+(?:the\s+)?([a-z][a-z -]{1,40})/i);
  if (instrument) memory.instruments = rememberUnique(memory.instruments, instrument[1], 4);
  const genre = message.match(/\b(?:my favourite genre is|my favorite genre is|i like playing)\s+([a-z][a-z &'-]{1,50})/i);
  if (genre) memory.genres = rememberUnique(memory.genres, genre[1], 4);
  const interest = message.match(/\b(?:outside music i like|i'm also into|i am also into|i love talking about)\s+(.{2,110})/i);
  if (interest) memory.interests = rememberUnique(memory.interests, interest[1].replace(/[.!?]+$/, ''), 4);
  const song = cleanText(lessonContext?.title, 160);
  if (song) memory.lastSong = song;
  const focus = cleanText(practiceReport?.focus, 120);
  if (focus) memory.lastFocus = focus;
  if (action?.type === 'demonstrate_range') memory.lastDemonstration = { ...action };
  session.memory = memory;
  return memory;
}

function endVirtualLesson(session, now = new Date(), reason = 'student-ended') {
  session.status = reason === 'expired' ? 'expired' : 'ended';
  session.endedAt = now.toISOString();
  session.endReason = cleanText(reason, 80);
  session.messages = [];
  session.memory = null;
  session.memoryClearedAt = now.toISOString();
  return session;
}

module.exports = {
  COMPANION_STYLES,
  CONVERSATION_MODES,
  DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
  LESSON_DEFAULT_DURATION_MINUTES,
  LESSON_DURATION_STEP_MINUTES,
  LESSON_MAX_DURATION_MINUTES,
  LESSON_MIN_DURATION_MINUTES,
  LESSON_RATE_MCOINS_PER_HOUR,
  MAX_SESSION_MESSAGES,
  activeVirtualLesson,
  appendSessionMessage,
  createVirtualLesson,
  endVirtualLesson,
  expireVirtualLessons,
  lessonCatalog,
  lessonQuote,
  normalizeClientRequestId,
  normalizeConversationMode,
  normalizeLessonBlockPrice,
  normalizeLessonDuration,
  parseTeacherDemonstration,
  publicVirtualLesson,
  sanitizeTeacher,
  sanitizeConversationPreferences,
  sessionIsActive,
  updateSessionMemory,
};
