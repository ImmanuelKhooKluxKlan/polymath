const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
require('dotenv').config({
  path: path.join(__dirname, '.env'),
});

const app = express();
const PORT = Number(process.env.PORT || 3000);
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || 'http://localhost:5173';

const PAYPAL_ENV = String(process.env.PAYPAL_ENV || 'live').trim().toLowerCase();
const PAYPAL_API_BASE = PAYPAL_ENV === 'sandbox'
  ? 'https://api-m.sandbox.paypal.com'
  : 'https://api-m.paypal.com';

const DATA_DIR = path.join(__dirname, "data");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const DB_PATH = path.join(DATA_DIR, "database.json");

const WITHDRAWAL_FEE_RATE = 0;
const MARKETPLACE_FEE_RATE = 0.10;
const MCOINS_PER_USD = 10;
const TRANSLATION_MCOIN_COST = 30;
const FREE_TRANSLATION_LIMIT = 1;
const PRO_TRANSLATION_LIMIT = 20;
const TRANSLATION_INITIAL_ESTIMATE_MS = 20 * 60 * 1000;
const TRANSLATION_EXTENSION_MS = 5 * 60 * 1000;
const MAX_PDF_BYTES = 10 * 1024 * 1024;
const WELCOME_MCOINS = Math.max(0, Math.floor(Number(process.env.WELCOME_MCOINS || 0)));
const MARKETPLACE_MAX_BYTES = 8 * 1024 * 1024;

const OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses';
const OPENAI_TRANSCRIPTIONS_URL = 'https://api.openai.com/v1/audio/transcriptions';
const OPENAI_LYRIC_MODEL = String(process.env.OPENAI_LYRIC_MODEL || 'whisper-1').trim();
const MAX_AUDIO_ANALYSIS_BYTES = 24 * 1024 * 1024;
const OPENAI_MODEL = String(process.env.OPENAI_MODEL || 'gpt-5.6').trim();
const OPENAI_PDF_DETAIL = ['low', 'high', 'auto'].includes(
  String(process.env.OPENAI_PDF_DETAIL || 'high').trim().toLowerCase(),
)
  ? String(process.env.OPENAI_PDF_DETAIL || 'high').trim().toLowerCase()
  : 'high';
const OPENAI_MAX_OUTPUT_TOKENS = Math.min(
  100000,
  Math.max(4000, Math.floor(Number(process.env.OPENAI_MAX_OUTPUT_TOKENS || 30000))),
);
const OPENAI_TIMEOUT_MS = Math.min(
  30 * 60 * 1000,
  Math.max(60 * 1000, Math.floor(Number(process.env.OPENAI_TIMEOUT_MS || 20 * 60 * 1000))),
);
const ADMIN_EMAILS = new Set(
  String(process.env.ADMIN_EMAILS || '')
    .split(',')
    .map((email) => email.trim().toLowerCase())
    .filter(Boolean),
);

const INSTRUMENTS = {
  piano: { label: 'Piano', cover: '🎹' },
  guitar: { label: 'Acoustic Guitar', cover: '🎸' },
  fiddle: { label: 'Five-string Fiddle', cover: '🎻' },
  banjo: { label: 'Five-string Banjo', cover: '🪕' },
  mandolin: { label: 'Mandolin', cover: '🎶' },
  dobro: { label: 'Dobro / Resonator Guitar', cover: '◉' },
  'upright-bass': { label: 'Upright Double Bass', cover: '🎼' },
  ukulele: { label: 'Ukulele', cover: '🏝️' },
  'electric-guitar': { label: 'Electric Guitar', cover: '⚡' },
  drums: { label: 'Drum Set', cover: '🥁' },
  synth: { label: 'Synth Keyboard', cover: '🎛️' },
  violin: { label: 'Violin', cover: '🎻' },
  cello: { label: 'Cello', cover: '🎻' },
  flute: { label: 'Flute', cover: '♫' },
  saxophone: { label: 'Saxophone', cover: '🎷' },
  trumpet: { label: 'Trumpet', cover: '🎺' },
  clarinet: { label: 'Clarinet', cover: '♬' },
};

const PRODUCTS = {
  'polymath-pro': {
    id: 'polymath-pro',
    name: 'Polymath Musician Pro',
    price: '19.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'MONTH',
    mcoins: 0,
  },
  'mcoins-50': {
    id: 'mcoins-50',
    name: '50 Mcoins',
    price: '5.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 50,
  },
  'mcoins-100': {
    id: 'mcoins-100',
    name: '100 Mcoins',
    price: '10.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 100,
  },
  'mcoins-300': {
    id: 'mcoins-300',
    name: '300 Mcoins',
    price: '30.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 300,
  },
};

function ensureStorage() {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
  if (!fs.existsSync(DB_PATH)) {
    const now = new Date().toISOString();
    const seed = {
      users: [
        {
          id: 'platform',
          name: 'Polymath Musician',
          email: 'platform@polymath.invalid',
          passwordHash: '',
          passwordSalt: '',
          mcoins: 0,
          withdrawableMcoins: 0,
          pro: true,
          proStatus: 'ACTIVE',
          paypalSubscriptionId: null,
          createdAt: now,
        },
      ],
      sessions: [],
      listings: [
        {
          id: 'demo-piano-1',
          sellerId: 'platform',
          artist: 'Traditional',
          title: 'Be Thou My Vision — Learning Edition',
          instrument: 'piano',
          format: 'JSON',
          priceMcoins: 140,
          description: 'Right/left-hand practice JSON with velocity and sustain-pedal events.',
          cover: '🎹',
          assetPath: 'demo-piano.json',
          filename: 'be-thou-my-vision-learning.json',
          demo: true,
          createdAt: now,
        },
        {
          id: 'demo-guitar-1',
          sellerId: 'platform',
          artist: 'Traditional',
          title: 'Come Thou Fount — Guitar Chords',
          instrument: 'guitar',
          format: 'JSON',
          priceMcoins: 90,
          description: 'Colour-coded chord changes and strumming practice.',
          cover: '🎸',
          assetPath: 'demo-guitar.json',
          filename: 'come-thou-fount-guitar.json',
          demo: true,
          createdAt: now,
        },

      ],
      purchases: [],
      messages: [],
      withdrawals: [],
      paymentOrders: [],
      subscriptions: [],
      webhookEvents: [],
      scoreTranslationJobs: [],
      bands: [],
      bandMemberships: [],
      ledger: [],
    };
    fs.writeFileSync(DB_PATH, JSON.stringify(seed, null, 2));
  }

  const pianoDemoPath = path.join(UPLOAD_DIR, 'demo-piano.json');
  if (!fs.existsSync(pianoDemoPath)) {
    fs.writeFileSync(pianoDemoPath, JSON.stringify({
      title: 'Be Thou My Vision — Learning Edition',
      composer: 'Traditional',
      bpm: 82,
      pedals: [
        { time: 0, down: true },
        { time: 3.8, down: false },
      ],
      notes: [
        { note: 'C4', time: 0, duration: 0.7, velocity: 0.72, hand: 'right' },
        { note: 'E4', time: 0.8, duration: 0.7, velocity: 0.76, hand: 'right' },
        { note: 'G4', time: 1.6, duration: 0.7, velocity: 0.8, hand: 'right' },
        { note: 'C5', time: 2.4, duration: 1.2, velocity: 0.84, hand: 'right' },
        { note: 'C3', time: 0, duration: 3.6, velocity: 0.62, hand: 'left' }
      ]
    }, null, 2));
  }

  const guitarDemoPath = path.join(UPLOAD_DIR, 'demo-guitar.json');
  if (!fs.existsSync(guitarDemoPath)) {
    fs.writeFileSync(guitarDemoPath, JSON.stringify({
      title: 'Come Thou Fount — Guitar Chords',
      bpm: 78,
      events: [
        { time: 0, chord: 'C', duration: 2, direction: 'down' },
        { time: 2, chord: 'G', duration: 2, direction: 'down' },
        { time: 4, chord: 'Am', duration: 2, direction: 'down' },
        { time: 6, chord: 'F', duration: 2, direction: 'down' }
      ]
    }, null, 2));
  }
}

function normalizeDb(db) {
  const normalized = db && typeof db === 'object' ? db : {};
  const arrays = [
    'users',
    'sessions',
    'listings',
    'purchases',
    'messages',
    'withdrawals',
    'paymentOrders',
    'subscriptions',
    'webhookEvents',
    'scoreTranslationJobs',
    'bands',
    'bandMemberships',
    'ledger',
  ];
  arrays.forEach((key) => {
    if (!Array.isArray(normalized[key])) normalized[key] = [];
  });
  normalized.users.forEach((user) => {
    user.mcoins = Number(user.mcoins || 0);
    user.withdrawableMcoins = Number(user.withdrawableMcoins || 0);
    if (!user.proStatus) user.proStatus = user.pro ? 'ACTIVE' : 'INACTIVE';
    if (!Object.prototype.hasOwnProperty.call(user, 'paypalSubscriptionId')) {
      user.paypalSubscriptionId = null;
    }
    user.pro = user.proStatus === 'ACTIVE' || user.pro === true;
  });
  return normalized;
}

function readDb() {
  ensureStorage();
  return normalizeDb(JSON.parse(fs.readFileSync(DB_PATH, 'utf8')));
}

function writeDb(db) {
  const temp = `${DB_PATH}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(db, null, 2));
  fs.renameSync(temp, DB_PATH);
}

function id(prefix) {
  return `${prefix}_${crypto.randomUUID()}`;
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return { salt, hash };
}

function normalizePhone(phone = '') {
  return String(phone).replace(/\D/g, '');
}

function currentTranslationPeriod(now = new Date()) {
  return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}`;
}

function nextTranslationResetAt(now = new Date()) {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1)).toISOString();
}

function ensureTranslationUsage(user, now = new Date()) {
  const period = currentTranslationPeriod(now);
  if (!user.translationUsage || user.translationUsage.period !== period) {
    user.translationUsage = {
      period,
      freeUsed: 0,
      proUsed: 0,
    };
  }
  user.translationUsage.freeUsed = Math.max(0, Number(user.translationUsage.freeUsed || 0));
  user.translationUsage.proUsed = Math.max(0, Number(user.translationUsage.proUsed || 0));
  return user.translationUsage;
}

function translationAllowance(user, now = new Date()) {
  const usage = ensureTranslationUsage(user, now);
  const isPro = Boolean(user.pro);
  const limit = isPro ? PRO_TRANSLATION_LIMIT : FREE_TRANSLATION_LIMIT;
  const used = isPro ? usage.proUsed : usage.freeUsed;
  return {
    plan: isPro ? 'pro' : 'free',
    limit,
    used: Math.min(limit, used),
    remaining: Math.max(0, limit - used),
    resetAt: nextTranslationResetAt(now),
  };
}

function deductTranslationAllowance(user) {
  const allowance = translationAllowance(user);
  if (allowance.remaining <= 0) return null;
  if (allowance.plan === 'pro') user.translationUsage.proUsed += 1;
  else user.translationUsage.freeUsed += 1;
  return allowance.plan;
}

function restoreTranslationAllowance(user, bucket) {
  ensureTranslationUsage(user);
  if (bucket === 'pro') user.translationUsage.proUsed = Math.max(0, user.translationUsage.proUsed - 1);
  if (bucket === 'free') user.translationUsage.freeUsed = Math.max(0, user.translationUsage.freeUsed - 1);
}

function safeUser(user) {
  return {
    user_id: user.id,
    name: user.name,
    email: user.email,
    phone: user.phone || '',
    mcoins: user.mcoins,
    withdrawableMcoins: Number(user.withdrawableMcoins || 0),
    pro: Boolean(user.pro),
    proStatus: user.proStatus || (user.pro ? 'ACTIVE' : 'INACTIVE'),
    paypalSubscriptionId: user.paypalSubscriptionId || null,
    translationAllowance: translationAllowance(user),
    admin: ADMIN_EMAILS.has(String(user.email || '').toLowerCase()),
    mustChangePassword: Boolean(user.mustChangePassword),
    createdAt: user.createdAt,
  };
}

function bearerToken(req) {
  const header = req.get('authorization') || '';
  return header.startsWith('Bearer ') ? header.slice(7).trim() : '';
}

function authUser(req, db) {
  const token = bearerToken(req);
  if (!token) return null;
  const session = db.sessions.find((item) => item.token === token);
  if (!session || new Date(session.expiresAt).getTime() < Date.now()) return null;
  return db.users.find((user) => user.id === session.userId) || null;
}

function requireAuth(req, res, next) {
  const db = readDb();
  const user = authUser(req, db);
  if (!user) return res.status(401).json({ error: 'Please sign in first.' });
  req.db = db;
  req.user = user;
  next();
}

function requireAdmin(req, res, next) {
  if (ADMIN_EMAILS.size === 0) {
    return res.status(503).json({ error: 'Admin access is not configured. Set ADMIN_EMAILS on the backend.' });
  }
  if (!ADMIN_EMAILS.has(String(req.user.email || '').toLowerCase())) {
    return res.status(403).json({ error: 'Administrator access is required.' });
  }
  next();
}

function adminPurchaseRows(db) {
  const rows = [];

  db.paymentOrders
    .filter((order) => order.status === 'COMPLETED')
    .forEach((order) => {
      const user = db.users.find((candidate) => candidate.id === order.userId);
      const product = PRODUCTS[order.productId];
      const amount = Number(order.amount ?? product?.price);
      if (!user || !Number.isFinite(amount)) return;
      rows.push({
        id: order.orderId,
        userId: user.id,
        name: user.name,
        email: user.email,
        purchase: product?.name || order.productId || 'PayPal purchase',
        amount,
        currency: order.currency || product?.currency || 'USD',
        status: order.status,
        purchasedAt: order.completedAt || order.createdAt,
      });
    });

  db.purchases.forEach((purchase) => {
    const user = db.users.find((candidate) => candidate.id === purchase.buyerId);
    const listing = db.listings.find((candidate) => candidate.id === purchase.listingId);
    const amount = Number(purchase.amount ?? purchase.amountMcoins ?? purchase.grossMcoins);
    if (!user || !Number.isFinite(amount)) return;
    rows.push({
      id: purchase.id,
      userId: user.id,
      name: user.name,
      email: user.email,
      purchase: listing?.title || purchase.listingId || 'Marketplace purchase',
      amount,
      currency: purchase.currency || 'MCOINS',
      status: 'COMPLETED',
      purchasedAt: purchase.createdAt,
    });
  });

  return rows.sort((a, b) => String(b.purchasedAt).localeCompare(String(a.purchasedAt)));
}

function addLedger(db, userId, amount, type, detail) {
  db.ledger.push({
    id: id('ledger'),
    userId,
    amount,
    type,
    detail,
    createdAt: new Date().toISOString(),
  });
}

function sanitizeFilename(filename = 'asset') {
  return filename.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 120) || 'asset';
}

function validateMarketplaceAsset(format, filename, bytes) {
  const lowerName = String(filename || '').toLowerCase();
  const extensionRules = {
    JSON: ['.json'],
    PDF: ['.pdf'],
    MIDI: ['.mid', '.midi'],
    MUSICXML: ['.musicxml', '.xml'],
  };
  if (!extensionRules[format]?.some((extension) => lowerName.endsWith(extension))) {
    throw new Error(`The selected filename does not match the ${format} file type.`);
  }
  if (!bytes.length || bytes.length > MARKETPLACE_MAX_BYTES) {
    throw new Error('File must be smaller than 8 MB.');
  }
  if (format === 'PDF' && bytes.subarray(0, 5).toString('ascii') !== '%PDF-') {
    throw new Error('Invalid PDF music sheet. The selected file is not a valid PDF.');
  }
  if (format === 'MIDI' && bytes.subarray(0, 4).toString('ascii') !== 'MThd') {
    throw new Error('Invalid MIDI file. A standard MIDI header was not found.');
  }
  if (format === 'JSON') {
    try {
      const parsed = JSON.parse(bytes.toString('utf8'));
      if (!parsed || (typeof parsed !== 'object')) throw new Error('invalid');
    } catch {
      throw new Error('Invalid ready-to-play JSON file.');
    }
  }
  if (format === 'MUSICXML') {
    const sample = bytes.subarray(0, Math.min(bytes.length, 250000)).toString('utf8').toLowerCase();
    if (!sample.includes('<score-partwise') && !sample.includes('<score-timewise')) {
      throw new Error('Invalid MusicXML score. A score-partwise or score-timewise document was not found.');
    }
  }
}

function decodeYouTubeText(value = '') {
  return String(value)
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCodePoint(parseInt(code, 16)));
}

function publicListing(listing, db, viewerId = null) {
  const seller = db.users.find((user) => user.id === listing.sellerId);
  const purchased = Boolean(viewerId && db.purchases.some(
    (purchase) => purchase.listingId === listing.id && purchase.buyerId === viewerId,
  ));
  return {
    ...listing,
    assetPath: undefined,
    seller: seller ? { user_id: seller.id, name: seller.name } : { user_id: listing.sellerId, name: 'Seller' },
    purchased,
    owned: viewerId === listing.sellerId,
  };
}

function bandMembership(db, bandId, userId) {
  return db.bandMemberships.find(
    (membership) => membership.bandId === bandId && membership.userId === userId,
  ) || null;
}

function safeBand(band, db, viewerId = null) {
  const host = db.users.find((user) => user.id === band.hostId);
  const membership = viewerId ? bandMembership(db, band.id, viewerId) : null;
  const isHost = viewerId === band.hostId;
  const canAccessParts = Boolean(membership || isHost);
  const members = db.bandMemberships
    .filter((item) => item.bandId === band.id)
    .map((item) => {
      const user = db.users.find((candidate) => candidate.id === item.userId);
      return {
        userId: item.userId,
        name: user?.name || 'Band member',
        role: item.role,
        joinedAt: item.joinedAt,
      };
    });
  return {
    id: band.id,
    name: band.name,
    description: band.description || '',
    host: { userId: band.hostId, name: host?.name || 'Band host' },
    accessMode: band.accessMode,
    entryFeeMcoins: Number(band.entryFeeMcoins || 0),
    memberCount: members.length,
    members: canAccessParts ? members : undefined,
    instruments: (Array.isArray(band.instruments) ? band.instruments : []).map((part) => (
      canAccessParts ? part : {
        id: part.id,
        instrument: part.instrument,
        name: part.name,
        hasScore: Boolean(part.score),
      }
    )),
    generalScore: canAccessParts ? (band.generalScore || null) : undefined,
    isHost,
    joined: Boolean(membership || isHost),
    role: isHost ? 'host' : membership?.role || null,
    inviteCode: isHost ? band.inviteCode : undefined,
    createdAt: band.createdAt,
  };
}

function validBandInstrument(value) {
  return Object.prototype.hasOwnProperty.call(INSTRUMENTS, value);
}

function lyricTokens(text) {
  return String(text || '')
    .match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || [];
}

function normalizedLyricToken(token) {
  return String(token || '').toLowerCase().replace(/[’]/g, "'").replace(/[^\p{L}\p{N}']/gu, '');
}

function lyricEditDistance(first, second) {
  const previous = Array.from({ length: second.length + 1 }, (_, index) => index);
  for (let row = 1; row <= first.length; row += 1) {
    const current = [row];
    for (let column = 1; column <= second.length; column += 1) {
      current[column] = Math.min(
        current[column - 1] + 1,
        previous[column] + 1,
        previous[column - 1] + (first[row - 1] === second[column - 1] ? 0 : 1),
      );
    }
    previous.splice(0, previous.length, ...current);
  }
  return previous[second.length];
}

function retimeExpectedLyrics(expected, start, end) {
  const weights = expected.map((word) => Math.max(1, normalizedLyricToken(word).length));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let cursor = start;
  return expected.map((word, index) => {
    const wordEnd = index === expected.length - 1
      ? end
      : cursor + (end - start) * weights[index] / total;
    const result = {
      word,
      start: Number(cursor.toFixed(3)),
      end: Number(wordEnd.toFixed(3)),
      correctedByLyricsHint: true,
    };
    cursor = wordEnd;
    return result;
  });
}

function applyLyricsHint(words, hint) {
  const expected = lyricTokens(hint);
  if (expected.length < 2 || !words.length) return { words, correctedWords: 0, matchConfidence: 0 };
  const recognized = words.map((word) => normalizedLyricToken(word.word));
  const expectedNormalized = expected.map(normalizedLyricToken);
  const minimumWindow = Math.max(2, expected.length - Math.ceil(expected.length * 0.2));
  const maximumWindow = Math.min(words.length, expected.length + Math.ceil(expected.length * 0.2));
  const candidates = [];

  for (let length = minimumWindow; length <= maximumWindow; length += 1) {
    for (let start = 0; start + length <= words.length; start += 1) {
      const candidate = recognized.slice(start, start + length);
      const distance = lyricEditDistance(candidate, expectedNormalized);
      const confidence = 1 - distance / Math.max(candidate.length, expectedNormalized.length);
      if (confidence >= 0.6) candidates.push({ start, length, confidence });
    }
  }
  candidates.sort((first, second) => second.confidence - first.confidence || first.start - second.start);
  const selected = [];
  candidates.forEach((candidate) => {
    const overlaps = selected.some((item) => (
      candidate.start < item.start + item.length && item.start < candidate.start + candidate.length
    ));
    if (!overlaps) selected.push(candidate);
  });
  selected.sort((first, second) => first.start - second.start);
  if (!selected.length) return { words, correctedWords: 0, matchConfidence: 0 };

  const corrected = [];
  let cursor = 0;
  let correctedWords = 0;
  selected.forEach((match) => {
    corrected.push(...words.slice(cursor, match.start));
    corrected.push(...retimeExpectedLyrics(
      expected,
      words[match.start].start,
      words[match.start + match.length - 1].end,
    ));
    correctedWords += expected.reduce((count, word, index) => (
      count + (normalizedLyricToken(word) !== recognized[match.start + index] ? 1 : 0)
    ), 0);
    cursor = match.start + match.length;
  });
  corrected.push(...words.slice(cursor));
  return {
    words: corrected,
    correctedWords,
    matchConfidence: selected.reduce((sum, match) => sum + match.confidence, 0) / selected.length,
  };
}

app.use(cors({
  origin(origin, callback) {
    if (!origin || origin === CLIENT_ORIGIN || /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin)) {
      callback(null, true);
      return;
    }
    callback(new Error('Origin is not allowed by Polymath Musician CORS policy.'));
  },
  credentials: false,
}));
app.use(express.json({ limit: '34mb' }));

app.get('/', (req, res) => res.send('Polymath Musician backend is running'));
app.get('/api/test', (req, res) => res.json({
  message: 'Backend is working',
  environment: PAYPAL_ENV,
  openaiConfigured: Boolean(String(process.env.OPENAI_API_KEY || '').trim()),
  openaiModel: OPENAI_MODEL,
}));

app.post('/api/audio/lyrics', requireAuth, async (req, res) => {
  const apiKey = String(process.env.OPENAI_API_KEY || '').trim();
  if (!apiKey) return res.status(503).json({ error: 'Model lyric analysis is not configured on the server.' });
  const filename = path.basename(String(req.body.filename || 'recording.wav')).replace(/[^a-z0-9_. -]/gi, '_');
  const content = Buffer.from(String(req.body.contentBase64 || ''), 'base64');
  if (!content.length || content.length > MAX_AUDIO_ANALYSIS_BYTES) {
    return res.status(400).json({ error: 'The prepared analysis audio must be between 1 byte and 24 MB.' });
  }
  if (!/^RIFF/.test(content.subarray(0, 4).toString('ascii'))) {
    return res.status(400).json({ error: 'The prepared lyric-analysis file is not a valid WAV file.' });
  }

  try {
    const form = new FormData();
    form.append('file', new Blob([content], { type: 'audio/wav' }), filename);
    form.append('model', OPENAI_LYRIC_MODEL);
    form.append('response_format', 'verbose_json');
    form.append('timestamp_granularities[]', 'word');
    form.append('language', /^[a-z]{2}$/i.test(String(req.body.language || '')) ? String(req.body.language).toLowerCase() : 'en');
    const lyricsHint = String(req.body.lyricsHint || '').trim().slice(0, 4000);
    if (lyricsHint) form.append('prompt', lyricsHint);
    const response = await fetch(OPENAI_TRANSCRIPTIONS_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: form,
      signal: AbortSignal.timeout(Math.min(OPENAI_TIMEOUT_MS, 10 * 60 * 1000)),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = String(data?.error?.message || `OpenAI transcription failed (${response.status}).`);
      return res.status(response.status === 401 ? 503 : 502).json({ error: message });
    }
    const rawWords = (data.words || []).map((word) => ({
      word: String(word.word || '').trim(),
      start: Number(word.start || 0),
      end: Number(word.end || 0),
    })).filter((word) => word.word && word.end > word.start);
    const alignment = lyricsHint
      ? applyLyricsHint(rawWords, lyricsHint)
      : { words: rawWords, correctedWords: 0, matchConfidence: 0 };
    return res.json({
      text: alignment.words.map((word) => word.word).join(' '),
      rawText: String(data.text || '').trim(),
      language: String(data.language || ''),
      duration: Number(data.duration || 0),
      words: alignment.words,
      lyricHintAlignment: {
        applied: alignment.correctedWords > 0,
        correctedWords: alignment.correctedWords,
        matchConfidence: Number(alignment.matchConfidence.toFixed(3)),
      },
      provider: 'OpenAI',
      model: OPENAI_LYRIC_MODEL,
    });
  } catch (error) {
    return res.status(502).json({ error: error.message || 'Model lyric analysis failed.' });
  }
});

app.get('/api/catalog', (req, res) => {
  res.json({
    products: Object.values(PRODUCTS),
    withdrawalFeeRate: WITHDRAWAL_FEE_RATE,
    withdrawalFeeLabel: 'No additional Polymath Musician withdrawal fee',
    marketplaceFeeRate: MARKETPLACE_FEE_RATE,
    mcoinsPerUsd: MCOINS_PER_USD,
    translationMcoinCost: TRANSLATION_MCOIN_COST,
  });
});

app.post('/api/auth/register', (req, res) => {
  const name = String(req.body.name || '').trim();
  const email = String(req.body.email || '').trim().toLowerCase();
  const phone = String(req.body.phone || '').trim();
  const password = String(req.body.password || '');
  if (name.length < 2) return res.status(400).json({ error: 'Name must contain at least 2 characters.' });
  if (!/^\S+@\S+\.\S+$/.test(email)) return res.status(400).json({ error: 'Enter a valid email.' });
  if (!/^\+?[0-9 ()-]{7,24}$/.test(phone)) return res.status(400).json({ error: 'Enter a valid phone number, including country code where possible.' });
  if (password.length < 8) return res.status(400).json({ error: 'Password must contain at least 8 characters.' });

  const db = readDb();
  if (db.users.some((user) => user.email === email)) return res.status(409).json({ error: 'An account already exists for this email.' });
  const normalizedPhone = normalizePhone(phone);
  if (db.users.some((user) => normalizePhone(user.phone) === normalizedPhone)) {
    return res.status(409).json({ error: 'An account already exists for this phone number.' });
  }
  const { salt, hash } = hashPassword(password);
  const user = {
    id: id('user'),
    name,
    email,
    phone,
    passwordHash: hash,
    passwordSalt: salt,
    mcoins: WELCOME_MCOINS,
    withdrawableMcoins: 0,
    pro: false,
    proStatus: 'INACTIVE',
    paypalSubscriptionId: null,
    translationUsage: {
      period: currentTranslationPeriod(),
      freeUsed: 0,
      proUsed: 0,
    },
    createdAt: new Date().toISOString(),
  };
  const token = crypto.randomBytes(32).toString('hex');
  db.users.push(user);
  db.sessions.push({ token, userId: user.id, expiresAt: new Date(Date.now() + 30 * 86400000).toISOString() });
  if (WELCOME_MCOINS > 0) addLedger(db, user.id, WELCOME_MCOINS, 'welcome_bonus', 'Configured welcome balance');
  writeDb(db);
  res.status(201).json({ token, user: safeUser(user) });
});

app.post('/api/auth/login', (req, res) => {
  const identifier = String(req.body.identifier || req.body.email || '').trim();
  const email = identifier.toLowerCase();
  const phone = normalizePhone(identifier);
  const password = String(req.body.password || '');
  const db = readDb();
  const user = db.users.find((candidate) => (
    String(candidate.email || '').toLowerCase() === email
    || (phone.length >= 7 && normalizePhone(candidate.phone) === phone)
  ));
  if (!user || !user.passwordHash) return res.status(401).json({ error: 'Email/phone or password is incorrect.' });
  const { hash } = hashPassword(password, user.passwordSalt);
  const matches = crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(user.passwordHash, 'hex'));
  if (!matches) return res.status(401).json({ error: 'Email/phone or password is incorrect.' });
  const token = crypto.randomBytes(32).toString('hex');
  db.sessions = db.sessions.filter((session) => new Date(session.expiresAt).getTime() >= Date.now());
  db.sessions.push({ token, userId: user.id, expiresAt: new Date(Date.now() + 30 * 86400000).toISOString() });
  writeDb(db);
  res.json({ token, user: safeUser(user) });
});

app.get('/api/auth/me', requireAuth, (req, res) => {
  res.json({ user: safeUser(req.user) });
});

app.post('/api/auth/change-password', requireAuth, (req, res) => {
  const password = String(req.body.password || '');
  if (password.length < 12) return res.status(400).json({ error: 'Your new password must contain at least 12 characters.' });
  const { salt, hash } = hashPassword(password);
  req.user.passwordHash = hash;
  req.user.passwordSalt = salt;
  req.user.mustChangePassword = false;
  const currentToken = bearerToken(req);
  req.db.sessions = req.db.sessions.filter((session) => session.userId !== req.user.id || session.token === currentToken);
  writeDb(req.db);
  res.json({ user: safeUser(req.user) });
});

app.post('/api/auth/logout', requireAuth, (req, res) => {
  const token = bearerToken(req);
  req.db.sessions = req.db.sessions.filter((session) => session.token !== token);
  writeDb(req.db);
  res.json({ ok: true });
});

app.get('/api/listings', (req, res) => {
  const db = readDb();
  const viewer = authUser(req, db);
  const instrument = String(req.query.instrument || '').toLowerCase();
  const artist = String(req.query.artist || '').toLowerCase();
  const query = String(req.query.query || '').toLowerCase();
  const format = String(req.query.format || '').toLowerCase();
  const listings = db.listings
    .filter((listing) => !instrument || listing.instrument.toLowerCase() === instrument)
    .filter((listing) => !artist || listing.artist.toLowerCase().includes(artist))
    .filter((listing) => !format || listing.format.toLowerCase() === format)
    .filter((listing) => !query || `${listing.artist} ${listing.title}`.toLowerCase().includes(query))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .map((listing) => publicListing(listing, db, viewer?.id));
  res.json({ listings });
});

app.get('/api/library', requireAuth, (req, res) => {
  const purchasedSongs = req.db.purchases
    .filter((purchase) => purchase.buyerId === req.user.id)
    .map((purchase) => {
      const listing = req.db.listings.find((item) => item.id === purchase.listingId);
      if (!listing) return null;
      return {
        ...publicListing(listing, req.db, req.user.id),
        purchasedAt: purchase.createdAt,
        purchaseAmountMcoins: Number(purchase.amountMcoins ?? purchase.amount ?? listing.priceMcoins),
      };
    })
    .filter(Boolean)
    .sort((a, b) => String(b.purchasedAt).localeCompare(String(a.purchasedAt)));
  const sellingSongs = req.db.listings
    .filter((listing) => listing.sellerId === req.user.id)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .map((listing) => publicListing(listing, req.db, req.user.id));
  res.json({ purchasedSongs, sellingSongs });
});

app.post('/api/listings', requireAuth, (req, res) => {
  const artist = String(req.body.artist || '').trim();
  const title = String(req.body.title || '').trim();
  const instrument = String(req.body.instrument || '').trim().toLowerCase();
  const format = String(req.body.format || '').trim().toUpperCase();
  const description = String(req.body.description || '').trim().slice(0, 800);
  const priceMcoins = Math.floor(Number(req.body.priceMcoins));
  const filename = sanitizeFilename(req.body.filename || `${title}.${format.toLowerCase()}`);
  const contentBase64 = String(req.body.contentBase64 || '');
  const rightsConfirmed = req.body.rightsConfirmed === true;
  const feeConfirmed = req.body.feeConfirmed === true;

  if (!artist || !title) return res.status(400).json({ error: 'Artist and song title are required.' });
  if (!rightsConfirmed) return res.status(400).json({ error: 'Confirm that you own the rights or have permission to sell this file.' });
  if (!feeConfirmed) return res.status(400).json({ error: 'Confirm the 10% marketplace fee before publishing.' });
  if (!INSTRUMENTS[instrument]) return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  if (!['JSON', 'PDF', 'MIDI', 'MUSICXML'].includes(format)) return res.status(400).json({ error: 'Unsupported listing format.' });
  if (!Number.isFinite(priceMcoins) || priceMcoins < 10 || priceMcoins > 100000 || priceMcoins % 10 !== 0) return res.status(400).json({ error: 'Price must be between 10 and 100,000 Mcoins in 10-Mcoin increments so the 10% fee is exact.' });
  if (!contentBase64) return res.status(400).json({ error: 'Attach the song file before publishing.' });

  let bytes;
  try {
    bytes = Buffer.from(contentBase64, 'base64');
  } catch {
    return res.status(400).json({ error: 'The attached file could not be decoded.' });
  }
  try {
    validateMarketplaceAsset(format, filename, bytes);
  } catch (error) {
    return res.status(400).json({ error: error.message });
  }

  const listingId = id('listing');
  const storedName = `${listingId}-${filename}`;
  fs.writeFileSync(path.join(UPLOAD_DIR, storedName), bytes);
  const listing = {
    id: listingId,
    sellerId: req.user.id,
    artist,
    title,
    instrument,
    format,
    priceMcoins,
    description,
    cover: INSTRUMENTS[instrument].cover,
    filename,
    assetPath: storedName,
    demo: false,
    rightsConfirmed: true,
    feeConfirmed: true,
    marketplaceFeeRate: MARKETPLACE_FEE_RATE,
    createdAt: new Date().toISOString(),
  };
  req.db.listings.push(listing);
  writeDb(req.db);
  res.status(201).json({ listing: publicListing(listing, req.db, req.user.id) });
});

app.put('/api/listings/:listingId', requireAuth, (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  if (listing.sellerId !== req.user.id) {
    return res.status(403).json({ error: 'Only the seller can amend this listing.' });
  }

  const artist = String(req.body.artist ?? listing.artist).trim();
  const title = String(req.body.title ?? listing.title).trim();
  const instrument = String(req.body.instrument ?? listing.instrument).trim().toLowerCase();
  const description = String(req.body.description ?? listing.description).trim().slice(0, 800);
  const priceMcoins = Math.floor(Number(req.body.priceMcoins ?? listing.priceMcoins));
  if (!artist || !title) return res.status(400).json({ error: 'Artist and song title are required.' });
  if (!INSTRUMENTS[instrument]) return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  if (!Number.isFinite(priceMcoins) || priceMcoins < 10 || priceMcoins > 100000 || priceMcoins % 10 !== 0) {
    return res.status(400).json({ error: 'Price must be between 10 and 100,000 Mcoins in 10-Mcoin increments.' });
  }

  Object.assign(listing, {
    artist,
    title,
    instrument,
    description,
    priceMcoins,
    cover: INSTRUMENTS[instrument].cover,
    updatedAt: new Date().toISOString(),
  });
  writeDb(req.db);
  res.json({ listing: publicListing(listing, req.db, req.user.id) });
});

app.post('/api/listings/:listingId/purchase', requireAuth, (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  if (listing.sellerId === req.user.id) return res.status(400).json({ error: 'You already own this listing.' });
  const existing = req.db.purchases.find((purchase) => purchase.listingId === listing.id && purchase.buyerId === req.user.id);
  if (existing) return res.json({ purchase: existing, user: safeUser(req.user) });
  if (req.user.mcoins < listing.priceMcoins) return res.status(402).json({ error: 'Not enough Mcoins.' });

  const seller = req.db.users.find((user) => user.id === listing.sellerId);
  const platform = req.db.users.find((user) => user.id === 'platform');
  const platformFeeMcoins = listing.priceMcoins / 10;
  const sellerEarningsMcoins = listing.priceMcoins - platformFeeMcoins;

  req.user.mcoins -= listing.priceMcoins;
  if (seller) {
    seller.mcoins += sellerEarningsMcoins;
    seller.withdrawableMcoins = Number(seller.withdrawableMcoins || 0) + sellerEarningsMcoins;
  }
  if (platform) {
    platform.mcoins += platformFeeMcoins;
  }

  const purchase = {
    id: id('purchase'),
    listingId: listing.id,
    buyerId: req.user.id,
    sellerId: listing.sellerId,
    amount: listing.priceMcoins,
    currency: 'MCOINS',
    amountMcoins: listing.priceMcoins,
    grossMcoins: listing.priceMcoins,
    platformFeeMcoins,
    platformFeeRate: MARKETPLACE_FEE_RATE,
    sellerEarningsMcoins,
    format: listing.format,
    instrument: listing.instrument,
    createdAt: new Date().toISOString(),
  };
  req.db.purchases.push(purchase);
  addLedger(req.db, req.user.id, -listing.priceMcoins, 'listing_purchase', `${listing.title} (${listing.format})`);
  if (seller) addLedger(req.db, seller.id, sellerEarningsMcoins, 'listing_sale', `${listing.title}; 10% platform fee: ${platformFeeMcoins} Mcoins`);
  if (platform) addLedger(req.db, platform.id, platformFeeMcoins, 'marketplace_fee', listing.title);
  writeDb(req.db);
  res.status(201).json({ purchase, user: safeUser(req.user) });
});

app.get('/api/listings/:listingId/download', requireAuth, (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  const allowed = listing.sellerId === req.user.id || req.db.purchases.some(
    (purchase) => purchase.listingId === listing.id && purchase.buyerId === req.user.id,
  );
  if (!allowed) return res.status(403).json({ error: 'Purchase this listing before downloading it.' });
  if (!listing.assetPath) {
    return res.status(409).json({ error: 'This demonstration listing does not include a downloadable asset.' });
  }
  res.download(path.join(UPLOAD_DIR, listing.assetPath), listing.filename || listing.assetPath);
});

app.get('/api/bands', (req, res) => {
  const db = readDb();
  const visible = db.bands
    .filter((band) => band.accessMode !== 'invite')
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((band) => safeBand(band, db));
  res.json({ bands: visible });
});

app.get('/api/bands/me', requireAuth, (req, res) => {
  const bands = req.db.bands
    .filter((band) => band.hostId === req.user.id || bandMembership(req.db, band.id, req.user.id))
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((band) => safeBand(band, req.db, req.user.id));
  res.json({ bands });
});

app.post('/api/bands', requireAuth, (req, res) => {
  const name = String(req.body.name || '').trim().slice(0, 80);
  const description = String(req.body.description || '').trim().slice(0, 500);
  const accessMode = String(req.body.accessMode || 'open').toLowerCase();
  const entryFeeMcoins = Math.max(0, Math.floor(Number(req.body.entryFeeMcoins || 0)));
  const password = String(req.body.password || '');
  if (name.length < 2) return res.status(400).json({ error: 'Band name must contain at least 2 characters.' });
  if (!['open', 'password', 'invite', 'paid'].includes(accessMode)) {
    return res.status(400).json({ error: 'Choose open, password, invite-only, or paid entry.' });
  }
  if (accessMode === 'password' && password.length < 4) {
    return res.status(400).json({ error: 'Band password must contain at least 4 characters.' });
  }
  if (accessMode === 'paid' && entryFeeMcoins < 1) {
    return res.status(400).json({ error: 'Paid bands need an entry fee of at least 1 Mcoin.' });
  }
  const secret = accessMode === 'password' ? hashPassword(password) : null;
  const band = {
    id: id('band'),
    hostId: req.user.id,
    name,
    description,
    accessMode,
    entryFeeMcoins: accessMode === 'paid' ? entryFeeMcoins : 0,
    passwordHash: secret?.hash || '',
    passwordSalt: secret?.salt || '',
    inviteCode: crypto.randomBytes(5).toString('hex').toUpperCase(),
    instruments: [],
    generalScore: null,
    createdAt: new Date().toISOString(),
  };
  req.db.bands.push(band);
  req.db.bandMemberships.push({
    id: id('band_member'),
    bandId: band.id,
    userId: req.user.id,
    role: 'host',
    joinedAt: band.createdAt,
  });
  writeDb(req.db);
  res.status(201).json({ band: safeBand(band, req.db, req.user.id), user: safeUser(req.user) });
});

app.post('/api/bands/join-by-code', requireAuth, (req, res) => {
  const code = String(req.body.code || '').trim().toUpperCase();
  const band = req.db.bands.find((candidate) => candidate.inviteCode === code);
  if (!band) return res.status(404).json({ error: 'That friend invite code is not valid.' });
  if (!bandMembership(req.db, band.id, req.user.id)) {
    req.db.bandMemberships.push({
      id: id('band_member'),
      bandId: band.id,
      userId: req.user.id,
      role: 'member',
      joinedAt: new Date().toISOString(),
    });
    writeDb(req.db);
  }
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.post('/api/bands/:bandId/join', requireAuth, (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (bandMembership(req.db, band.id, req.user.id)) {
    return res.json({ band: safeBand(band, req.db, req.user.id), user: safeUser(req.user) });
  }
  if (band.accessMode === 'invite') {
    return res.status(403).json({ error: 'This band can only be joined with its friend invite code.' });
  }
  if (band.accessMode === 'password') {
    const password = String(req.body.password || '');
    const attempted = hashPassword(password, band.passwordSalt).hash;
    const validHash = attempted.length === band.passwordHash.length
      && crypto.timingSafeEqual(Buffer.from(attempted, 'hex'), Buffer.from(band.passwordHash, 'hex'));
    if (!validHash) return res.status(403).json({ error: 'The band password is incorrect.' });
  }
  if (band.accessMode === 'paid') {
    const fee = Math.max(1, Number(band.entryFeeMcoins || 0));
    if (req.user.mcoins < fee) return res.status(402).json({ error: 'Not enough Mcoins to join this band.' });
    const host = req.db.users.find((user) => user.id === band.hostId);
    req.user.mcoins -= fee;
    if (host && host.id !== req.user.id) {
      host.mcoins += fee;
      host.withdrawableMcoins = Number(host.withdrawableMcoins || 0) + fee;
      addLedger(req.db, host.id, fee, 'band_entry_received', `${req.user.name} joined ${band.name}`);
    }
    addLedger(req.db, req.user.id, -fee, 'band_entry_paid', band.name);
  }
  req.db.bandMemberships.push({
    id: id('band_member'),
    bandId: band.id,
    userId: req.user.id,
    role: 'member',
    joinedAt: new Date().toISOString(),
  });
  writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id), user: safeUser(req.user) });
});

app.put('/api/bands/:bandId/general-score', requireAuth, (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  const membership = bandMembership(req.db, band.id, req.user.id);
  if (!membership) return res.status(403).json({ error: 'Join the band before changing its general sheet.' });
  const score = req.body.score;
  if (score === null) {
    band.generalScore = null;
  } else {
    if (!score || typeof score !== 'object' || !Array.isArray(score.notes)) {
      return res.status(400).json({ error: 'The general music sheet is not valid.' });
    }
    if (score.notes.length > 30000) return res.status(400).json({ error: 'General sheets support up to 30,000 notes.' });
    band.generalScore = {
      title: String(score.title || 'General band sheet').slice(0, 140),
      composer: String(score.composer || '').slice(0, 140),
      bpm: Math.max(20, Math.min(320, Number(score.bpm || 120))),
      notes: score.notes.map((note, index) => ({
        id: String(note.id || `band-general-${index}`).slice(0, 100),
        note: String(note.note || '').slice(0, 12),
        time: Math.max(0, Number(note.time || 0)),
        duration: Math.max(0.02, Number(note.duration || 0.4)),
        velocity: Math.max(0.01, Math.min(1.2, Number(note.velocity || 0.8))),
        stringIndex: Number.isInteger(note.stringIndex) ? note.stringIndex : undefined,
        fret: Number.isFinite(Number(note.fret)) ? Number(note.fret) : undefined,
      })),
    };
  }
  writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.post('/api/bands/:bandId/instruments', requireAuth, (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  const membership = bandMembership(req.db, band.id, req.user.id);
  if (!membership) return res.status(403).json({ error: 'Join the band before adding an instrument.' });
  const instrument = String(req.body.instrument || '');
  if (!validBandInstrument(instrument)) return res.status(400).json({ error: 'Choose a supported instrument.' });
  if (!Array.isArray(band.instruments)) band.instruments = [];
  if (band.instruments.length >= 16) return res.status(400).json({ error: 'A band can contain up to 16 instrument parts.' });
  const part = {
    id: id('band_part'),
    instrument,
    name: String(req.body.name || INSTRUMENTS[instrument].label).trim().slice(0, 80),
    addedBy: req.user.id,
    score: null,
    muted: false,
    volume: 0.82,
    visualEnabled: false,
    createdAt: new Date().toISOString(),
  };
  band.instruments.push(part);
  writeDb(req.db);
  res.status(201).json({ band: safeBand(band, req.db, req.user.id), part });
});

app.put('/api/bands/:bandId/instruments/:partId', requireAuth, (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  const membership = bandMembership(req.db, band.id, req.user.id);
  if (!membership) return res.status(403).json({ error: 'Join the band before editing its parts.' });
  const part = (band.instruments || []).find((candidate) => candidate.id === req.params.partId);
  if (!part) return res.status(404).json({ error: 'Instrument part not found.' });
  const score = req.body.score;
  if (score != null) {
    if (!score || typeof score !== 'object' || !Array.isArray(score.notes)) {
      return res.status(400).json({ error: 'The uploaded band score is not valid.' });
    }
    if (score.notes.length > 30000) return res.status(400).json({ error: 'Band parts support up to 30,000 notes.' });
    part.score = {
      title: String(score.title || 'Uploaded part').slice(0, 140),
      composer: String(score.composer || '').slice(0, 140),
      bpm: Math.max(20, Math.min(320, Number(score.bpm || 120))),
      notes: score.notes.map((note, index) => ({
        id: String(note.id || `band-note-${index}`).slice(0, 100),
        note: String(note.note || '').slice(0, 12),
        time: Math.max(0, Number(note.time || 0)),
        duration: Math.max(0.02, Number(note.duration || 0.4)),
        velocity: Math.max(0.01, Math.min(1.2, Number(note.velocity || 0.8))),
        stringIndex: Number.isInteger(note.stringIndex) ? note.stringIndex : undefined,
        fret: Number.isFinite(Number(note.fret)) ? Number(note.fret) : undefined,
      })),
    };
  }
  if (req.body.name !== undefined) part.name = String(req.body.name || '').trim().slice(0, 80) || part.name;
  if (req.body.muted !== undefined) part.muted = Boolean(req.body.muted);
  if (req.body.volume !== undefined) part.volume = Math.max(0, Math.min(1.2, Number(req.body.volume || 0)));
  if (req.body.visualEnabled !== undefined) part.visualEnabled = Boolean(req.body.visualEnabled);
  writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id), part });
});

app.delete('/api/bands/:bandId/instruments/:partId', requireAuth, (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  const part = (band.instruments || []).find((candidate) => candidate.id === req.params.partId);
  if (!part) return res.status(404).json({ error: 'Instrument part not found.' });
  if (band.hostId !== req.user.id && part.addedBy !== req.user.id) {
    return res.status(403).json({ error: 'Only the host or the person who added this part can remove it.' });
  }
  band.instruments = band.instruments.filter((candidate) => candidate.id !== part.id);
  writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.get('/api/messages/threads', requireAuth, (req, res) => {
  const related = req.db.messages.filter((message) => message.fromUserId === req.user.id || message.toUserId === req.user.id);
  const latestByOther = new Map();
  related.forEach((message) => {
    const otherId = message.fromUserId === req.user.id ? message.toUserId : message.fromUserId;
    const current = latestByOther.get(otherId);
    if (!current || current.createdAt < message.createdAt) latestByOther.set(otherId, message);
  });
  const threads = [...latestByOther.entries()]
    .map(([otherId, lastMessage]) => {
      const other = req.db.users.find((user) => user.id === otherId);
      return { otherUser: other ? { user_id: other.id, name: other.name } : { user_id: otherId, name: 'User' }, lastMessage };
    })
    .sort((a, b) => b.lastMessage.createdAt.localeCompare(a.lastMessage.createdAt));
  res.json({ threads });
});

app.get('/api/messages/:otherUserId', requireAuth, (req, res) => {
  const otherId = req.params.otherUserId;
  const messages = req.db.messages
    .filter((message) => (
      (message.fromUserId === req.user.id && message.toUserId === otherId)
      || (message.fromUserId === otherId && message.toUserId === req.user.id)
    ))
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  const other = req.db.users.find((user) => user.id === otherId);
  res.json({ messages, otherUser: other ? { user_id: other.id, name: other.name } : null });
});

app.post('/api/messages', requireAuth, (req, res) => {
  const toUserId = String(req.body.toUserId || '');
  const text = String(req.body.text || '').trim().slice(0, 2000);
  if (!text) return res.status(400).json({ error: 'Message cannot be empty.' });
  if (toUserId === req.user.id) return res.status(400).json({ error: 'You cannot message yourself.' });
  if (!req.db.users.some((user) => user.id === toUserId)) return res.status(404).json({ error: 'Recipient not found.' });
  const message = { id: id('message'), fromUserId: req.user.id, toUserId, text, createdAt: new Date().toISOString() };
  req.db.messages.push(message);
  writeDb(req.db);
  res.status(201).json({ message });
});

app.get('/api/wallet', requireAuth, (req, res) => {
  const ledger = req.db.ledger.filter((entry) => entry.userId === req.user.id).slice(-100).reverse();
  const withdrawals = req.db.withdrawals.filter((item) => item.userId === req.user.id).slice(-20).reverse();
  res.json({ user: safeUser(req.user), ledger, withdrawals, withdrawalFeeRate: WITHDRAWAL_FEE_RATE });
});

app.get('/api/admin/customer-purchases', requireAuth, requireAdmin, (req, res) => {
  const rows = adminPurchaseRows(req.db);
  const totals = rows.reduce((result, row) => {
    result[row.currency] = Number((Number(result[row.currency] || 0) + row.amount).toFixed(2));
    return result;
  }, {});
  res.json({
    columns: ['name', 'email', 'purchase', 'amount', 'currency', 'status', 'purchasedAt'],
    rows,
    footer: {
      label: 'Total',
      amounts: totals,
    },
  });
});

app.get('/api/admin/users', requireAuth, requireAdmin, (req, res) => {
  const customers = req.db.users.filter((user) => user.id !== 'platform');
  const rows = customers.map((user) => {
    const completedOrders = req.db.paymentOrders.filter((order) => order.userId === user.id && order.status === 'COMPLETED');
    const usdSpent = completedOrders.reduce((total, order) => {
      const product = PRODUCTS[order.productId];
      const amount = Number(order.amount ?? product?.price);
      return total + (Number.isFinite(amount) && (order.currency || product?.currency || 'USD') === 'USD' ? amount : 0);
    }, 0);
    const marketplacePurchases = req.db.purchases.filter((purchase) => purchase.buyerId === user.id);
    const marketplaceSpentMcoins = marketplacePurchases.reduce(
      (total, purchase) => total + Number(purchase.amountMcoins ?? purchase.amount ?? purchase.grossMcoins ?? 0),
      0,
    );
    const mcoins = Number(user.mcoins || 0);
    return {
      userId: user.id,
      name: user.name,
      email: user.email,
      phone: user.phone || '',
      mcoins,
      mcoinUsdEquivalent: Number((mcoins / MCOINS_PER_USD).toFixed(2)),
      usdSpent: Number(usdSpent.toFixed(2)),
      marketplaceSpentMcoins,
      marketplaceSpentUsdEquivalent: Number((marketplaceSpentMcoins / MCOINS_PER_USD).toFixed(2)),
      purchaseCount: completedOrders.length + marketplacePurchases.length,
      proStatus: user.proStatus || (user.pro ? 'ACTIVE' : 'INACTIVE'),
      createdAt: user.createdAt,
    };
  }).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));

  const platformFeesMcoins = req.db.purchases.reduce(
    (total, purchase) => total + Number(purchase.platformFeeMcoins || 0),
    0,
  );
  const totalUsdRevenue = rows.reduce((total, row) => total + row.usdSpent, 0);
  res.json({
    rows,
    footer: {
      userCount: rows.length,
      totalUsdRevenue: Number(totalUsdRevenue.toFixed(2)),
      totalMcoinsHeld: rows.reduce((total, row) => total + row.mcoins, 0),
      totalMcoinsHeldUsdEquivalent: Number((rows.reduce((total, row) => total + row.mcoins, 0) / MCOINS_PER_USD).toFixed(2)),
      marketplaceFeesMcoins: platformFeesMcoins,
      marketplaceFeesUsdEquivalent: Number((platformFeesMcoins / MCOINS_PER_USD).toFixed(2)),
    },
  });
});

app.post('/api/admin/users/:userId/reset-password', requireAuth, requireAdmin, (req, res) => {
  const user = req.db.users.find((candidate) => candidate.id === req.params.userId && candidate.id !== 'platform');
  if (!user) return res.status(404).json({ error: 'User account not found.' });
  if (user.id === req.user.id) {
    return res.status(400).json({ error: 'Use your own account password-change flow instead of an administrator reset.' });
  }
  const password = String(req.body.password || '');
  if (password.length < 8) {
    return res.status(400).json({ error: 'The password must contain at least 8 characters.' });
  }
  const { salt, hash } = hashPassword(password);
  user.passwordHash = hash;
  user.passwordSalt = salt;
  user.mustChangePassword = false;
  user.passwordResetAt = new Date().toISOString();
  user.passwordResetBy = req.user.id;
  req.db.sessions = req.db.sessions.filter((session) => session.userId !== user.id);
  writeDb(req.db);
  res.json({
    userId: user.id,
    message: `Password updated for ${user.name}. All of their existing sessions were signed out.`,
  });
});

app.post('/api/wallet/buy-pro', requireAuth, (req, res) => {
  res.status(410).json({
    error: 'Pro is a recurring PayPal subscription. Mcoins are reserved for one-time marketplace purchases.',
  });
});

app.post('/api/wallet/withdraw', requireAuth, (req, res) => {
  const amountMcoins = Math.floor(Number(req.body.amountMcoins));
  const payoutEmail = String(req.body.payoutEmail || '').trim().toLowerCase();
  if (!Number.isFinite(amountMcoins) || amountMcoins < 100) return res.status(400).json({ error: 'Minimum withdrawal is 100 Mcoins.' });
  if (req.user.mcoins < amountMcoins) return res.status(402).json({ error: 'Insufficient Mcoin balance.' });
  if (Number(req.user.withdrawableMcoins || 0) < amountMcoins) return res.status(402).json({ error: 'Only Mcoins earned from song sales can be withdrawn.' });
  if (!/^\S+@\S+\.\S+$/.test(payoutEmail)) return res.status(400).json({ error: 'Enter a valid payout email.' });
  const feeMcoins = 0;
  const netMcoins = amountMcoins;
  req.user.mcoins -= amountMcoins;
  req.user.withdrawableMcoins = Number(req.user.withdrawableMcoins || 0) - amountMcoins;
  const withdrawal = {
    id: id('withdrawal'),
    userId: req.user.id,
    amountMcoins,
    feeMcoins,
    netMcoins,
    feeRate: WITHDRAWAL_FEE_RATE,
    payoutEmail,
    status: 'pending_manual_review',
    createdAt: new Date().toISOString(),
  };
  req.db.withdrawals.push(withdrawal);
  addLedger(req.db, req.user.id, -amountMcoins, 'withdrawal_requested', `No additional Polymath Musician fee; net: ${netMcoins} Mcoins`);
  writeDb(req.db);
  res.status(201).json({ withdrawal, user: safeUser(req.user) });
});

async function getPayPalAccessToken() {
  const clientId = String(process.env.PAYPAL_CLIENT_ID || '').trim();
  const clientSecret = String(process.env.PAYPAL_SECRET_KEY || '').trim();
  if (!clientId || !clientSecret) {
    throw new Error('PAYPAL_CLIENT_ID or PAYPAL_SECRET_KEY is missing');
  }

  const credentials = Buffer.from(`${clientId}:${clientSecret}`).toString('base64');
  const response = await fetch(`${PAYPAL_API_BASE}/v1/oauth2/token`, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded',
      Accept: 'application/json',
    },
    body: 'grant_type=client_credentials',
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error_description || data.error || 'Could not authenticate with PayPal');
  }
  return data.access_token;
}

async function paypalRequest(pathname, options = {}) {
  const accessToken = await getPayPalAccessToken();
  const response = await fetch(`${PAYPAL_API_BASE}${pathname}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers || {}),
    },
  });
  const data = await response.json().catch(() => ({}));
  return { response, data };
}

function findPaypalLink(data, relations) {
  const accepted = new Set(relations);
  return data?.links?.find((link) => accepted.has(link.rel))?.href || '';
}

function subscriptionStatusGrantsPro(status) {
  return String(status || '').toUpperCase() === 'ACTIVE';
}

function applySubscriptionStatus(db, subscriptionId, status, fallbackUserId = '') {
  const normalizedStatus = String(status || 'UNKNOWN').toUpperCase();
  let record = db.subscriptions.find((item) => item.subscriptionId === subscriptionId);
  const userId = record?.userId || fallbackUserId;
  const user = db.users.find((candidate) => candidate.id === userId);

  if (!record && user && subscriptionId) {
    record = {
      subscriptionId,
      productId: 'polymath-pro',
      planId: String(process.env.PAYPAL_PRO_PLAN_ID || ''),
      userId: user.id,
      status: normalizedStatus,
      createdAt: new Date().toISOString(),
    };
    db.subscriptions.push(record);
  }

  if (record) {
    record.status = normalizedStatus;
    record.updatedAt = new Date().toISOString();
  }

  if (user) {
    user.paypalSubscriptionId = subscriptionId;
    user.proStatus = normalizedStatus;
    user.pro = subscriptionStatusGrantsPro(normalizedStatus);
  }

  return user || null;
}

async function validatePayPalSubscriptionPlan(product, planId) {
  const { response, data } = await paypalRequest(`/v1/billing/plans/${encodeURIComponent(planId)}`, {
    method: 'GET',
  });
  if (!response.ok) {
    const error = new Error('Could not verify the configured PayPal subscription plan.');
    error.status = response.status;
    error.details = data;
    throw error;
  }

  const regularCycle = (data.billing_cycles || []).find((cycle) => cycle.tenure_type === 'REGULAR');
  const fixedPrice = regularCycle?.pricing_scheme?.fixed_price;
  const frequency = regularCycle?.frequency;
  const expectedPrice = Number(product.price).toFixed(2);
  const actualPrice = Number(fixedPrice?.value).toFixed(2);

  if (
    data.status !== 'ACTIVE'
    || fixedPrice?.currency_code !== product.currency
    || actualPrice !== expectedPrice
    || frequency?.interval_unit !== product.interval
    || Number(frequency?.interval_count || 0) !== 1
  ) {
    const error = new Error(
      `PAYPAL_PRO_PLAN_ID must be an active ${product.currency} ${expectedPrice} monthly plan.`,
    );
    error.status = 409;
    throw error;
  }

  return data;
}

app.post('/api/paypal/create-order', requireAuth, async (req, res) => {
  try {
    const product = PRODUCTS[String(req.body.productId || '')];
    if (!product) return res.status(404).json({ error: 'Product not found.' });
    if (product.kind !== 'mcoins') {
      return res.status(400).json({ error: 'Recurring Pro access must use the subscription checkout.' });
    }

    const requestId = id('paypal_order');
    const { response, data } = await paypalRequest('/v2/checkout/orders', {
      method: 'POST',
      headers: { 'PayPal-Request-Id': requestId },
      body: JSON.stringify({
        intent: 'CAPTURE',
        purchase_units: [{
          reference_id: product.id,
          custom_id: req.user.id,
          description: product.name,
          amount: { currency_code: product.currency, value: product.price },
        }],
        payment_source: {
          paypal: {
            experience_context: {
              brand_name: 'Polymath Musician',
              return_url: `${CLIENT_ORIGIN}/?paymentStatus=approved&productId=${encodeURIComponent(product.id)}`,
              cancel_url: `${CLIENT_ORIGIN}/?paymentStatus=cancelled&productId=${encodeURIComponent(product.id)}`,
              user_action: 'PAY_NOW',
            },
          },
        },
      }),
    });

    if (!response.ok) {
      return res.status(response.status).json({ error: 'Could not create PayPal order.', details: data });
    }

    const approveUrl = findPaypalLink(data, ['payer-action', 'approve']);
    req.db.paymentOrders.push({
      orderId: data.id,
      productId: product.id,
      userId: req.user.id,
      amount: Number(product.price),
      currency: product.currency,
      requestId,
      status: data.status,
      createdAt: new Date().toISOString(),
    });
    writeDb(req.db);
    res.status(201).json({ orderId: data.id, approveUrl, product });
  } catch (error) {
    console.error('PayPal create order failed:', error.message);
    res.status(error.status || 500).json({ error: error.message, ...(error.details ? { details: error.details } : {}) });
  }
});

app.post('/api/paypal/capture-order', requireAuth, async (req, res) => {
  try {
    const orderId = String(req.body.orderId || '').trim();
    const record = req.db.paymentOrders.find((item) => item.orderId === orderId && item.userId === req.user.id);
    if (!record) return res.status(404).json({ error: 'Payment order not found.' });
    if (record.status === 'COMPLETED') {
      return res.json({ user: safeUser(req.user), product: PRODUCTS[record.productId], alreadyCaptured: true });
    }

    const product = PRODUCTS[record.productId];
    if (!product || product.kind !== 'mcoins') {
      return res.status(409).json({ error: 'This order is not an Mcoin purchase.' });
    }

    const captureResult = await paypalRequest(`/v2/checkout/orders/${encodeURIComponent(orderId)}/capture`, {
      method: 'POST',
      headers: { 'PayPal-Request-Id': `capture-${orderId}` },
      body: '{}',
    });

    let paymentData = captureResult.data;
    if (!captureResult.response.ok) {
      const retrieved = await paypalRequest(`/v2/checkout/orders/${encodeURIComponent(orderId)}`, {
        method: 'GET',
      });
      if (!retrieved.response.ok || retrieved.data.status !== 'COMPLETED') {
        return res.status(captureResult.response.status).json({
          error: 'Could not capture PayPal payment.',
          details: captureResult.data,
        });
      }
      paymentData = retrieved.data;
    }

    if (paymentData.status !== 'COMPLETED') {
      return res.status(409).json({ error: `Payment status is ${paymentData.status}.` });
    }

    const capturedAmount = paymentData.purchase_units?.[0]?.payments?.captures?.[0]?.amount;
    if (
      capturedAmount?.currency_code !== product.currency
      || Number(capturedAmount?.value).toFixed(2) !== Number(product.price).toFixed(2)
    ) {
      return res.status(409).json({ error: 'Captured PayPal amount does not match the selected Mcoin product.' });
    }

    record.status = 'COMPLETED';
    record.amount = Number(capturedAmount.value);
    record.currency = capturedAmount.currency_code;
    record.completedAt = new Date().toISOString();
    req.user.mcoins += product.mcoins;
    addLedger(req.db, req.user.id, product.mcoins, 'mcoin_purchase', product.name);
    writeDb(req.db);
    res.json({ user: safeUser(req.user), product, paypalStatus: paymentData.status });
  } catch (error) {
    console.error('PayPal capture order failed:', error.message);
    res.status(error.status || 500).json({ error: error.message, ...(error.details ? { details: error.details } : {}) });
  }
});

app.post('/api/paypal/create-subscription', requireAuth, async (req, res) => {
  try {
    const product = PRODUCTS[String(req.body.productId || 'polymath-pro')];
    if (!product || product.kind !== 'subscription') {
      return res.status(404).json({ error: 'Subscription product not found.' });
    }

    const planId = String(process.env.PAYPAL_PRO_PLAN_ID || '').trim();
    if (!planId) {
      return res.status(503).json({ error: 'PAYPAL_PRO_PLAN_ID is not configured on the server.' });
    }

    if (req.user.pro && req.user.proStatus === 'ACTIVE') {
      return res.status(409).json({ error: 'Pro is already active for this account.' });
    }

    const reusablePending = req.db.subscriptions.find((item) => (
      item.userId === req.user.id
      && item.productId === product.id
      && ['APPROVAL_PENDING', 'APPROVED', 'CREATED'].includes(String(item.status || '').toUpperCase())
      && item.approveUrl
      && Date.now() - new Date(item.createdAt).getTime() < 24 * 60 * 60 * 1000
    ));
    if (reusablePending) {
      return res.json({
        subscriptionId: reusablePending.subscriptionId,
        approveUrl: reusablePending.approveUrl,
        product,
        status: reusablePending.status,
        reused: true,
      });
    }

    await validatePayPalSubscriptionPlan(product, planId);

    const requestId = id('paypal_subscription');
    const { response, data } = await paypalRequest('/v1/billing/subscriptions', {
      method: 'POST',
      headers: { 'PayPal-Request-Id': requestId },
      body: JSON.stringify({
        plan_id: planId,
        custom_id: req.user.id,
        application_context: {
          brand_name: 'Polymath Musician',
          locale: 'en-US',
          shipping_preference: 'NO_SHIPPING',
          user_action: 'SUBSCRIBE_NOW',
          return_url: `${CLIENT_ORIGIN}/?paymentStatus=subscription-approved&productId=${encodeURIComponent(product.id)}`,
          cancel_url: `${CLIENT_ORIGIN}/?paymentStatus=cancelled&productId=${encodeURIComponent(product.id)}`,
        },
      }),
    });

    if (!response.ok) {
      return res.status(response.status).json({ error: 'Could not create PayPal subscription.', details: data });
    }

    const approveUrl = findPaypalLink(data, ['approve']);
    req.db.subscriptions.push({
      subscriptionId: data.id,
      productId: product.id,
      planId,
      userId: req.user.id,
      requestId,
      approveUrl,
      status: data.status || 'APPROVAL_PENDING',
      createdAt: new Date().toISOString(),
    });
    req.user.paypalSubscriptionId = data.id;
    req.user.proStatus = data.status || 'APPROVAL_PENDING';
    req.user.pro = false;
    writeDb(req.db);
    res.status(201).json({ subscriptionId: data.id, approveUrl, product, status: data.status });
  } catch (error) {
    console.error('PayPal create subscription failed:', error.message);
    res.status(error.status || 500).json({ error: error.message, ...(error.details ? { details: error.details } : {}) });
  }
});

app.post('/api/paypal/confirm-subscription', requireAuth, async (req, res) => {
  try {
    const subscriptionId = String(req.body.subscriptionId || '').trim();
    if (!subscriptionId) return res.status(400).json({ error: 'Subscription ID is required.' });

    const record = req.db.subscriptions.find(
      (item) => item.subscriptionId === subscriptionId && item.userId === req.user.id,
    );
    if (!record) return res.status(404).json({ error: 'Subscription record not found.' });

    const { response, data } = await paypalRequest(`/v1/billing/subscriptions/${encodeURIComponent(subscriptionId)}`, {
      method: 'GET',
    });
    if (!response.ok) {
      return res.status(response.status).json({ error: 'Could not verify PayPal subscription.', details: data });
    }
    if (data.custom_id && data.custom_id !== req.user.id) {
      return res.status(403).json({ error: 'Subscription owner does not match this account.' });
    }
    if (data.plan_id && data.plan_id !== record.planId) {
      return res.status(409).json({ error: 'PayPal subscription plan does not match the configured Pro plan.' });
    }

    const user = applySubscriptionStatus(req.db, subscriptionId, data.status, req.user.id);
    if (user?.pro && !record.activatedAt) {
      record.activatedAt = new Date().toISOString();
      addLedger(req.db, user.id, 0, 'pro_subscription_active', PRODUCTS['polymath-pro'].name);
    }
    writeDb(req.db);

    res.json({
      user: safeUser(req.user),
      product: PRODUCTS['polymath-pro'],
      subscriptionStatus: data.status,
      active: subscriptionStatusGrantsPro(data.status),
    });
  } catch (error) {
    console.error('PayPal confirm subscription failed:', error.message);
    res.status(error.status || 500).json({ error: error.message, ...(error.details ? { details: error.details } : {}) });
  }
});

app.post('/api/paypal/webhook', async (req, res) => {
  const webhookId = String(process.env.PAYPAL_WEBHOOK_ID || '').trim();
  if (!webhookId) return res.status(503).json({ error: 'PAYPAL_WEBHOOK_ID is not configured.' });

  try {
    const verificationPayload = {
      auth_algo: req.get('paypal-auth-algo'),
      cert_url: req.get('paypal-cert-url'),
      transmission_id: req.get('paypal-transmission-id'),
      transmission_sig: req.get('paypal-transmission-sig'),
      transmission_time: req.get('paypal-transmission-time'),
      webhook_id: webhookId,
      webhook_event: req.body,
    };
    const { response, data } = await paypalRequest('/v1/notifications/verify-webhook-signature', {
      method: 'POST',
      body: JSON.stringify(verificationPayload),
    });
    if (!response.ok || data.verification_status !== 'SUCCESS') {
      return res.status(400).json({ error: 'PayPal webhook signature verification failed.' });
    }

    const db = readDb();
    const eventId = String(req.body.id || '');
    if (eventId && db.webhookEvents.some((event) => event.eventId === eventId)) {
      return res.json({ received: true, duplicate: true });
    }

    const eventType = String(req.body.event_type || '');
    const resource = req.body.resource || {};
    const subscriptionId = String(resource.id || resource.billing_agreement_id || '');
    const statusByType = {
      'BILLING.SUBSCRIPTION.ACTIVATED': 'ACTIVE',
      'BILLING.SUBSCRIPTION.CANCELLED': 'CANCELLED',
      'BILLING.SUBSCRIPTION.SUSPENDED': 'SUSPENDED',
      'BILLING.SUBSCRIPTION.EXPIRED': 'EXPIRED',
      'BILLING.SUBSCRIPTION.PAYMENT.FAILED': 'PAYMENT_FAILED',
    };
    const webhookStatus = eventType === 'BILLING.SUBSCRIPTION.UPDATED'
      ? resource.status
      : statusByType[eventType];

    if (subscriptionId && webhookStatus) {
      applySubscriptionStatus(db, subscriptionId, webhookStatus, resource.custom_id || '');
    }

    db.webhookEvents.push({
      eventId: eventId || id('webhook'),
      eventType,
      resourceId: subscriptionId || String(resource.id || ''),
      receivedAt: new Date().toISOString(),
    });
    if (db.webhookEvents.length > 1000) db.webhookEvents = db.webhookEvents.slice(-1000);
    writeDb(db);
    res.json({ received: true });
  } catch (error) {
    console.error('PayPal webhook failed:', error.message);
    res.status(500).json({ error: 'Webhook processing failed.' });
  }
});

app.get('/api/youtube/search', async (req, res) => {
  const apiKey = String(process.env.YOUTUBE_API_KEY || '').trim();
  if (!apiKey) {
    return res.status(503).json({ error: 'YOUTUBE_API_KEY is not configured on the server.' });
  }

  const query = String(req.query.q || '').trim().slice(0, 180);
  if (query.length < 2) return res.status(400).json({ error: 'Enter a YouTube search query.' });

  try {
    const params = new URLSearchParams({
      part: 'snippet',
      type: 'video',
      videoEmbeddable: 'true',
      safeSearch: 'moderate',
      maxResults: '8',
      q: query,
      key: apiKey,
    });
    const response = await fetch(`https://www.googleapis.com/youtube/v3/search?${params.toString()}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return res.status(response.status).json({ error: 'YouTube search failed.', details: data });
    }

    const videos = (data.items || [])
      .filter((item) => item.id?.videoId)
      .map((item) => ({
        videoId: item.id.videoId,
        title: decodeYouTubeText(item.snippet?.title || 'YouTube video'),
        channelTitle: decodeYouTubeText(item.snippet?.channelTitle || ''),
        thumbnail: item.snippet?.thumbnails?.medium?.url || item.snippet?.thumbnails?.default?.url || '',
        publishedAt: item.snippet?.publishedAt || '',
      }));
    res.json({ query, videos });
  } catch (error) {
    res.status(502).json({ error: 'Could not reach YouTube search.', details: error.message });
  }
});

function decodePdfBase64(contentBase64) {
  if (!contentBase64) throw new Error('Attach a PDF music sheet.');
  let bytes;
  try {
    bytes = Buffer.from(contentBase64, 'base64');
  } catch {
    throw new Error('Invalid PDF music sheet. The file could not be decoded.');
  }
  if (!bytes.length || bytes.length > MAX_PDF_BYTES) {
    throw new Error('Invalid PDF music sheet. The PDF must be smaller than 10 MB.');
  }
  if (bytes.subarray(0, 5).toString('ascii') !== '%PDF-') {
    throw new Error('Invalid PDF music sheet. The selected file is not a valid PDF.');
  }
  return bytes;
}

function rejectClearlyNonMusicPdf(bytes, filename) {
  const sample = bytes.subarray(0, Math.min(bytes.length, 300000)).toString('latin1').toLowerCase();
  const haystack = `${String(filename || '').toLowerCase()}\n${sample}`;
  const musicSignals = [
    'music', 'score', 'sheet', 'piano', 'guitar', 'violin', 'fiddle', 'banjo',
    'mandolin', 'dobro', 'bass', 'ukulele', 'drum', 'synth', 'electric guitar', 'tempo', 'treble', 'clef', 'chord', 'staff',
    'tablature', 'composer', 'opus', 'allegro', 'andante',
  ];
  const nonMusicSignals = [
    'invoice', 'bank statement', 'curriculum vitae', 'resume', 'tax return',
    'lease agreement', 'insurance policy', 'purchase order', 'medical report',
  ];
  const hasMusicSignal = musicSignals.some((signal) => haystack.includes(signal));
  const hasClearNonMusicSignal = nonMusicSignals.some((signal) => haystack.includes(signal));
  if (hasClearNonMusicSignal && !hasMusicSignal) {
    throw new Error('Invalid PDF music sheet. Please upload a readable instrumental music sheet.');
  }
}

function extendTranslationEstimate(job) {
  if (job.status !== 'processing') return false;
  let estimatedReady = new Date(job.estimatedReadyAt).getTime();
  if (!Number.isFinite(estimatedReady)) {
    estimatedReady = new Date(job.startedAt).getTime() + TRANSLATION_INITIAL_ESTIMATE_MS;
  }
  let changed = false;
  while (Date.now() >= estimatedReady) {
    estimatedReady += TRANSLATION_EXTENSION_MS;
    job.estimateExtensionCount = Number(job.estimateExtensionCount || 0) + 1;
    changed = true;
  }
  if (changed) job.estimatedReadyAt = new Date(estimatedReady).toISOString();
  return changed;
}

function publicTranslationJob(job) {
  const estimatedReady = new Date(job.estimatedReadyAt).getTime();
  return {
    id: job.id,
    filename: job.filename,
    instrument: job.instrument,
    paymentMethod: job.paymentMethod,
    costMcoins: job.costMcoins || 0,
    status: job.status,
    stage: job.stage,
    progress: Number(job.progress || 0),
    startedAt: job.startedAt,
    estimatedReadyAt: job.estimatedReadyAt,
    estimatedRemainingSeconds: job.status === 'processing'
      ? Math.max(0, Math.ceil((estimatedReady - Date.now()) / 1000))
      : 0,
    estimateExtensionCount: Number(job.estimateExtensionCount || 0),
    completedAt: job.completedAt || null,
    failedAt: job.failedAt || null,
    error: job.error || '',
    refunded: Boolean(job.refundedAt),
  };
}

function refundTranslationJob(db, job, reason) {
  if (job.refundedAt) return;
  const user = db.users.find((candidate) => candidate.id === job.userId);
  if (user) {
    if (job.paymentMethod === 'mcoins') {
      user.mcoins += TRANSLATION_MCOIN_COST;
      addLedger(db, user.id, TRANSLATION_MCOIN_COST, 'translation_refund', job.filename);
    } else if (job.allowanceBucket) {
      restoreTranslationAllowance(user, job.allowanceBucket);
      addLedger(db, user.id, 0, 'translation_allowance_restored', job.filename);
    }
  }
  job.status = 'failed';
  job.stage = 'Translation stopped';
  job.progress = Math.max(5, Number(job.progress || 5));
  job.error = reason || 'The music sheet could not be translated.';
  job.failedAt = new Date().toISOString();
  job.refundedAt = job.failedAt;
}

const READY_TO_PLAY_SHEET_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    isInstrumentalMusicSheet: {
      type: 'boolean',
      description: 'True only when the PDF contains readable instrumental music notation.',
    },
    rejectionReason: {
      type: 'string',
      description: 'Empty when valid; otherwise a concise reason the PDF cannot be translated.',
    },
    title: {
      type: 'string',
      description: 'Printed song or work title, or an empty string when unavailable.',
    },
    composer: {
      type: 'string',
      description: 'Printed composer or arranger, or an empty string when unavailable.',
    },
    instrument: {
      type: 'string',
      description: 'The selected instrument part that was transcribed.',
    },
    bpm: {
      type: 'number',
      minimum: 20,
      maximum: 300,
      description: 'Main tempo in beats per minute. Use a reasonable notation-derived value only when no BPM is printed.',
    },
    timeSignature: {
      type: 'object',
      additionalProperties: false,
      properties: {
        numerator: { type: 'integer', minimum: 1, maximum: 32 },
        denominator: { type: 'integer', enum: [1, 2, 4, 8, 16, 32] },
      },
      required: ['numerator', 'denominator'],
    },
    keySignature: {
      type: 'string',
      description: 'Key signature such as C major, A minor, F# minor, or an empty string.',
    },
    notes: {
      type: 'array',
      description: 'Absolute-pitch playable notes. Times and durations are measured in seconds from the start.',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          note: {
            type: 'string',
            description: 'Scientific pitch notation such as C4, F#3, or Bb5.',
          },
          time: { type: 'number', minimum: 0 },
          duration: { type: 'number', exclusiveMinimum: 0 },
          velocity: { type: 'number', minimum: 0, maximum: 1 },
          hand: {
            type: 'string',
            enum: ['left', 'right', 'both', ''],
          },
          voice: {
            type: 'string',
            description: 'Voice, staff, or part label, or an empty string.',
          },
          articulation: {
            type: 'string',
            description: 'Articulation such as staccato, tenuto, accent, legato, or an empty string.',
          },
        },
        required: ['note', 'time', 'duration', 'velocity', 'hand', 'voice', 'articulation'],
      },
    },
    events: {
      type: 'array',
      description: 'General playable events for chords, strums, plucks, bows, rests, or other instrument actions.',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          type: {
            type: 'string',
            enum: ['note', 'chord', 'rest', 'strum', 'pluck', 'bow', 'other'],
          },
          time: { type: 'number', minimum: 0 },
          duration: { type: 'number', exclusiveMinimum: 0 },
          notes: {
            type: 'array',
            items: { type: 'string' },
          },
          chord: { type: 'string' },
          direction: {
            type: 'string',
            enum: ['up', 'down', 'alternating', 'none', ''],
          },
          velocity: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: ['type', 'time', 'duration', 'notes', 'chord', 'direction', 'velocity'],
      },
    },
    tabs: {
      type: 'array',
      description: 'String-and-fret events when tablature is present.',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          stringNumber: { type: 'integer', minimum: 1, maximum: 12 },
          fret: { type: 'integer', minimum: 0, maximum: 36 },
          note: { type: 'string' },
          time: { type: 'number', minimum: 0 },
          duration: { type: 'number', exclusiveMinimum: 0 },
          velocity: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: ['stringNumber', 'fret', 'note', 'time', 'duration', 'velocity'],
      },
    },
    pedals: {
      type: 'array',
      description: 'Sustain-pedal state changes where printed or strongly implied by the score.',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          time: { type: 'number', minimum: 0 },
          down: { type: 'boolean' },
        },
        required: ['time', 'down'],
      },
    },
    warnings: {
      type: 'array',
      items: { type: 'string' },
      description: 'Specific uncertainties or unsupported notation. Empty when none.',
    },
    confidence: {
      type: 'number',
      minimum: 0,
      maximum: 1,
      description: 'Overall confidence in the transcription.',
    },
  },
  required: [
    'isInstrumentalMusicSheet',
    'rejectionReason',
    'title',
    'composer',
    'instrument',
    'bpm',
    'timeSignature',
    'keySignature',
    'notes',
    'events',
    'tabs',
    'pedals',
    'warnings',
    'confidence',
  ],
};

function clampNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(maximum, Math.max(minimum, number));
}

function normalizePitchName(value) {
  const match = String(value || '').trim().match(/^([A-Ga-g])([#b]?)(-?\d+)$/);
  if (!match) return '';
  return `${match[1].toUpperCase()}${match[2]}${match[3]}`;
}

function normalizeReadyToPlaySong(rawResult, selectedInstrument) {
  if (!rawResult || typeof rawResult !== 'object') {
    throw new Error('OpenAI did not return a readable music-sheet result.');
  }

  if (rawResult.isInstrumentalMusicSheet !== true) {
    throw new Error(
      rawResult.rejectionReason
      || 'Invalid PDF music sheet. Please upload a readable instrumental music sheet.',
    );
  }

  const notes = (Array.isArray(rawResult.notes) ? rawResult.notes : [])
    .map((note) => ({
      note: normalizePitchName(note.note),
      time: clampNumber(note.time, 0, 24 * 60 * 60, 0),
      duration: clampNumber(note.duration, 0.01, 60 * 60, 0.25),
      velocity: clampNumber(note.velocity, 0.01, 1, 0.75),
      hand: ['left', 'right', 'both', ''].includes(note.hand) ? note.hand : '',
      voice: String(note.voice || '').slice(0, 80),
      articulation: String(note.articulation || '').slice(0, 80),
    }))
    .filter((note) => note.note)
    .sort((a, b) => a.time - b.time || a.note.localeCompare(b.note));

  const events = (Array.isArray(rawResult.events) ? rawResult.events : [])
    .map((event) => ({
      type: ['note', 'chord', 'rest', 'strum', 'pluck', 'bow', 'other'].includes(event.type)
        ? event.type
        : 'other',
      time: clampNumber(event.time, 0, 24 * 60 * 60, 0),
      duration: clampNumber(event.duration, 0.01, 60 * 60, 0.25),
      notes: (Array.isArray(event.notes) ? event.notes : [])
        .map(normalizePitchName)
        .filter(Boolean),
      chord: String(event.chord || '').slice(0, 80),
      direction: ['up', 'down', 'alternating', 'none', ''].includes(event.direction)
        ? event.direction
        : '',
      velocity: clampNumber(event.velocity, 0.01, 1, 0.75),
    }))
    .filter((event) => (
      event.type === 'rest'
      || event.notes.length > 0
      || event.chord
      || ['strum', 'pluck', 'bow', 'other'].includes(event.type)
    ))
    .sort((a, b) => a.time - b.time);

  const tabs = (Array.isArray(rawResult.tabs) ? rawResult.tabs : [])
    .map((tab) => ({
      stringNumber: Math.round(clampNumber(tab.stringNumber, 1, 12, 1)),
      fret: Math.round(clampNumber(tab.fret, 0, 36, 0)),
      note: normalizePitchName(tab.note),
      time: clampNumber(tab.time, 0, 24 * 60 * 60, 0),
      duration: clampNumber(tab.duration, 0.01, 60 * 60, 0.25),
      velocity: clampNumber(tab.velocity, 0.01, 1, 0.75),
    }))
    .sort((a, b) => a.time - b.time);

  const pedals = (Array.isArray(rawResult.pedals) ? rawResult.pedals : [])
    .map((pedal) => ({
      time: clampNumber(pedal.time, 0, 24 * 60 * 60, 0),
      down: Boolean(pedal.down),
    }))
    .sort((a, b) => a.time - b.time);

  const playableEventCount = notes.length
    + events.filter((event) => event.type !== 'rest').length
    + tabs.length;

  if (playableEventCount <= 0) {
    throw new Error('Invalid PDF music sheet. No playable instrumental notes were detected.');
  }

  const numerator = Math.round(clampNumber(
    rawResult.timeSignature?.numerator,
    1,
    32,
    4,
  ));
  const allowedDenominators = [1, 2, 4, 8, 16, 32];
  const requestedDenominator = Math.round(Number(rawResult.timeSignature?.denominator || 4));
  const denominator = allowedDenominators.includes(requestedDenominator)
    ? requestedDenominator
    : 4;

  return {
    title: String(rawResult.title || '').trim().slice(0, 180),
    composer: String(rawResult.composer || '').trim().slice(0, 180),
    instrument: selectedInstrument,
    bpm: clampNumber(rawResult.bpm, 20, 300, 120),
    timeSignature: { numerator, denominator },
    keySignature: String(rawResult.keySignature || '').trim().slice(0, 80),
    notes,
    events,
    tabs,
    pedals,
    warnings: (Array.isArray(rawResult.warnings) ? rawResult.warnings : [])
      .map((warning) => String(warning || '').trim().slice(0, 300))
      .filter(Boolean)
      .slice(0, 50),
    confidence: clampNumber(rawResult.confidence, 0, 1, 0.5),
  };
}

function buildMusicTranslationPrompt(instrument) {
  const instrumentLabel = INSTRUMENTS[instrument]?.label || instrument;

  return [
    'You are an expert optical-music-recognition editor and professional music engraver.',
    `Analyze the attached PDF visually and translate the ${instrumentLabel} part into Polymath Musician ready-to-play data.`,
    '',
    'Critical transcription rules:',
    '1. Transcribe the printed score accurately. Do not invent notes, rhythms, chords, tablature, pedal markings, or repeats.',
    '2. If the PDF is not a readable instrumental music sheet, set isInstrumentalMusicSheet to false, explain why in rejectionReason, and return empty playable arrays.',
    `3. Focus on the selected instrument: ${instrumentLabel}. If that part is absent or unreadable, reject the sheet.`,
    '4. Convert musical timing to seconds beginning at time 0. Use the printed tempo. When no metronome BPM is printed, infer a conservative BPM from the tempo marking and mention that in warnings.',
    '5. Use scientific pitch notation: C4 is middle C; accidentals look like F#4 or Bb3.',
    '6. Preserve simultaneous notes by giving them the same start time. Preserve rests, ties, note lengths, voices, hands/staves, articulations, and chords when visible.',
    '7. Expand clearly marked repeats into playback order. Do not guess ambiguous jumps, codas, or endings; report ambiguity in warnings.',
    '8. For guitar, banjo, mandolin, or dobro tablature, include stringNumber and fret in tabs whenever visible, while also including sounding pitch when reliably determined.',
    '9. For piano or synth keyboard, include exact pitches and simultaneous notes. For piano, include left/right hand and printed sustain-pedal changes. Do not invent pedal markings.',
    '10. For drum-set notation, convert strikes to these Polymath Musician trigger notes: kick C2, snare D2, closed hi-hat F#2, low/floor tom G2, mid tom A2, high tom C3, crash C#3, and ride D#3. Preserve simultaneous strikes and do not interpret these triggers as pitched melody.',
    '11. For chord-only lead sheets, use chord events at the correct musical times. Do not fabricate individual voicings unless the notation prints them.',
    '12. Process all readable pages and all measures belonging to the selected part.',
    '13. Return only data conforming to the required JSON schema.',
  ].join('\n');
}

function extractOpenAIOutputText(data) {
  if (typeof data?.output_text === 'string' && data.output_text.trim()) {
    return data.output_text.trim();
  }

  const refusals = [];
  const textParts = [];

  (Array.isArray(data?.output) ? data.output : []).forEach((item) => {
    (Array.isArray(item?.content) ? item.content : []).forEach((content) => {
      if (content?.type === 'output_text' && typeof content.text === 'string') {
        textParts.push(content.text);
      }
      if (content?.type === 'refusal' && typeof content.refusal === 'string') {
        refusals.push(content.refusal);
      }
    });
  });

  if (refusals.length > 0) {
    throw new Error(`OpenAI could not process this sheet: ${refusals.join(' ')}`);
  }

  const combined = textParts.join('\n').trim();
  if (!combined) {
    throw new Error('OpenAI returned no ready-to-play sheet data.');
  }
  return combined;
}

async function translatePdfWithOpenAI(bytes, filename, instrument) {
  const apiKey = String(process.env.OPENAI_API_KEY || '').trim();
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not configured on the server.');
  }

  const response = await fetch(OPENAI_RESPONSES_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
    body: JSON.stringify({
      model: OPENAI_MODEL,
      store: false,
      max_output_tokens: OPENAI_MAX_OUTPUT_TOKENS,
      input: [
        {
          role: 'user',
          content: [
            {
              type: 'input_file',
              filename,
              file_data: `data:application/pdf;base64,${bytes.toString('base64')}`,
              detail: OPENAI_PDF_DETAIL,
            },
            {
              type: 'input_text',
              text: buildMusicTranslationPrompt(instrument),
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'polymath_ready_to_play_sheet',
          strict: true,
          schema: READY_TO_PLAY_SHEET_SCHEMA,
        },
      },
    }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const providerMessage = String(
      data?.error?.message
      || data?.error
      || `OpenAI request failed with HTTP ${response.status}.`,
    );

    if (response.status === 401) {
      throw new Error('OpenAI rejected OPENAI_API_KEY. Check the key in server/.env.');
    }
    if (response.status === 429) {
      throw new Error(`OpenAI rate limit or billing limit reached. ${providerMessage}`);
    }
    throw new Error(providerMessage);
  }

  if (data.status === 'incomplete') {
    const reason = data.incomplete_details?.reason || 'unknown reason';
    throw new Error(
      `OpenAI could not finish the full sheet (${reason}). Increase OPENAI_MAX_OUTPUT_TOKENS or use a shorter PDF.`,
    );
  }

  const outputText = extractOpenAIOutputText(data);

  let parsed;
  try {
    parsed = JSON.parse(outputText);
  } catch {
    throw new Error('OpenAI returned a response that was not valid ready-to-play JSON.');
  }

  return {
    song: normalizeReadyToPlaySong(parsed, instrument),
    openaiResponseId: String(data.id || ''),
    model: String(data.model || OPENAI_MODEL),
  };
}

async function processTranslationJob(jobId) {
  let db = readDb();
  let job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
  if (!job || job.status !== 'processing') return;

  try {
    job.stage = 'Reading music notation with OpenAI';
    job.progress = 18;
    writeDb(db);

    const sourcePath = path.join(UPLOAD_DIR, job.sourcePath);
    const bytes = fs.readFileSync(sourcePath);

    db = readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.stage = 'Translating notes and timing';
    job.progress = 42;
    writeDb(db);

    const openaiResult = await translatePdfWithOpenAI(
      bytes,
      job.filename,
      job.instrument,
    );
    const result = openaiResult.song;

    db = readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.stage = 'Checking ready-to-play sheet';
    job.progress = 82;
    job.openaiResponseId = openaiResult.openaiResponseId;
    job.openaiModel = openaiResult.model;
    writeDb(db);

    const outputName = `${job.id}-${sanitizeFilename(job.filename.replace(/\.pdf$/i, '') || 'ready-to-play-sheet')}.json`;
    fs.writeFileSync(path.join(UPLOAD_DIR, outputName), JSON.stringify({
      ...result,
      sourcePdf: job.filename,
      readyToPlayFormat: 'polymath-musician-json-v1',
      translationProvider: 'OpenAI',
      translatedAt: new Date().toISOString(),
    }, null, 2));

    db = readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.outputPath = outputName;
    job.outputFilename = `${sanitizeFilename(job.filename.replace(/\.pdf$/i, '') || 'ready-to-play-sheet')}.json`;
    job.status = 'completed';
    job.stage = 'Ready to download';
    job.progress = 100;
    job.completedAt = new Date().toISOString();
    writeDb(db);
  } catch (error) {
    db = readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    refundTranslationJob(db, job, error.message);
    writeDb(db);
  }
}

app.get('/api/score-translations/usage', requireAuth, (req, res) => {
  writeDb(req.db);
  res.json({
    user: safeUser(req.user),
    translationMcoinCost: TRANSLATION_MCOIN_COST,
    mcoinsPerUsd: MCOINS_PER_USD,
  });
});

app.get('/api/score-translations', requireAuth, (req, res) => {
  let changed = false;
  const jobs = req.db.scoreTranslationJobs
    .filter((job) => job.userId === req.user.id)
    .sort((a, b) => b.startedAt.localeCompare(a.startedAt))
    .slice(0, 20)
    .map((job) => {
      if (extendTranslationEstimate(job)) changed = true;
      return publicTranslationJob(job);
    });
  if (changed) writeDb(req.db);
  res.json({ jobs, user: safeUser(req.user) });
});

app.post('/api/score-translations', requireAuth, (req, res) => {
  const openaiApiKey = String(process.env.OPENAI_API_KEY || '').trim();
  if (!openaiApiKey) {
    return res.status(503).json({
      error: 'PDF translation is temporarily unavailable because OpenAI is not configured. Nothing was charged.',
      setup: 'Set OPENAI_API_KEY in server/.env and restart the backend.',
    });
  }

  const filename = sanitizeFilename(String(req.body.filename || 'music-sheet.pdf'));
  const instrument = String(req.body.instrument || '').trim().toLowerCase();
  const paymentMethod = String(req.body.paymentMethod || '').trim().toLowerCase();

  if (!filename.toLowerCase().endsWith('.pdf')) {
    return res.status(400).json({ error: 'Invalid PDF music sheet. Please upload a PDF music sheet.' });
  }
  if (!INSTRUMENTS[instrument]) {
    return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  }
  if (!['allowance', 'mcoins'].includes(paymentMethod)) {
    return res.status(400).json({ error: 'Choose a monthly translation or 30-Mcoin payment.' });
  }

  let bytes;
  try {
    bytes = decodePdfBase64(String(req.body.contentBase64 || ''));
    rejectClearlyNonMusicPdf(bytes, filename);
  } catch (error) {
    return res.status(400).json({ error: error.message });
  }

  const fileHash = crypto.createHash('sha256').update(bytes).digest('hex');
  const recentDuplicate = req.db.scoreTranslationJobs.find((job) => (
    job.userId === req.user.id
    && job.fileHash === fileHash
    && ['processing', 'completed'].includes(job.status)
    && Date.now() - new Date(job.startedAt).getTime() < 10 * 60 * 1000
  ));
  if (recentDuplicate) {
    extendTranslationEstimate(recentDuplicate);
    writeDb(req.db);
    return res.json({ job: publicTranslationJob(recentDuplicate), user: safeUser(req.user), duplicate: true });
  }

  let allowanceBucket = null;
  if (paymentMethod === 'allowance') {
    allowanceBucket = deductTranslationAllowance(req.user);
    if (!allowanceBucket) {
      return res.status(402).json({
        error: req.user.pro
          ? '0 of 20 Pro translations remain this month. Pay 30 Mcoins to continue.'
          : '0 free translations remain this month. Pay 30 Mcoins or buy Pro.',
      });
    }
  } else {
    if (req.user.mcoins < TRANSLATION_MCOIN_COST) {
      return res.status(402).json({ error: 'You need 30 Mcoins for this translation.' });
    }
    req.user.mcoins -= TRANSLATION_MCOIN_COST;
    addLedger(req.db, req.user.id, -TRANSLATION_MCOIN_COST, 'pdf_translation', filename);
  }

  const jobId = id('translation');
  const sourceName = `${jobId}-${filename}`;
  fs.writeFileSync(path.join(UPLOAD_DIR, sourceName), bytes);
  const now = Date.now();
  const job = {
    id: jobId,
    userId: req.user.id,
    filename,
    instrument,
    paymentMethod,
    allowanceBucket,
    costMcoins: paymentMethod === 'mcoins' ? TRANSLATION_MCOIN_COST : 0,
    fileHash,
    sourcePath: sourceName,
    status: 'processing',
    stage: 'Queued for music translation',
    progress: 5,
    startedAt: new Date(now).toISOString(),
    estimatedReadyAt: new Date(now + TRANSLATION_INITIAL_ESTIMATE_MS).toISOString(),
    estimateExtensionCount: 0,
  };
  req.db.scoreTranslationJobs.push(job);
  if (paymentMethod === 'allowance') {
    addLedger(req.db, req.user.id, 0, 'translation_allowance_used', `${filename} (${allowanceBucket})`);
  }
  writeDb(req.db);

  setImmediate(() => {
    processTranslationJob(job.id).catch((error) => console.error('Translation job failed:', error));
  });

  res.status(202).json({
    job: publicTranslationJob(job),
    user: safeUser(req.user),
    translationMcoinCost: TRANSLATION_MCOIN_COST,
  });
});

app.get('/api/score-translations/:jobId', requireAuth, (req, res) => {
  const job = req.db.scoreTranslationJobs.find((candidate) => candidate.id === req.params.jobId && candidate.userId === req.user.id);
  if (!job) return res.status(404).json({ error: 'Translation job not found.' });
  const changed = extendTranslationEstimate(job);
  if (changed) writeDb(req.db);
  res.json({ job: publicTranslationJob(job), user: safeUser(req.user) });
});

app.get('/api/score-translations/:jobId/download', requireAuth, (req, res) => {
  const job = req.db.scoreTranslationJobs.find((candidate) => candidate.id === req.params.jobId && candidate.userId === req.user.id);
  if (!job) return res.status(404).json({ error: 'Translation job not found.' });
  if (job.status !== 'completed' || !job.outputPath) {
    return res.status(409).json({ error: 'The ready-to-play sheet is not available yet.' });
  }
  res.download(path.join(UPLOAD_DIR, job.outputPath), job.outputFilename || 'ready-to-play-sheet.json');
});

app.post('/api/score-import', (req, res) => {
  res.status(410).json({
    error: 'Direct PDF conversion has been replaced by the user-facing translation queue. Sign in and use /api/score-translations.',
  });
});

function resumePendingTranslationJobs() {
  if (!String(process.env.OPENAI_API_KEY || '').trim()) return;
  const db = readDb();
  db.scoreTranslationJobs
    .filter((job) => job.status === 'processing')
    .forEach((job) => setImmediate(() => {
      processTranslationJob(job.id).catch((error) => console.error('Translation resume failed:', error));
    }));
}

ensureStorage();
if (require.main === module) {
  resumePendingTranslationJobs();
  app.listen(PORT, () => console.log(`Polymath Musician backend running on http://localhost:${PORT}`));
}

module.exports = { app, applyLyricsHint };
