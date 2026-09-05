const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const multer = require('multer');
const bundledFfmpegPath = require('ffmpeg-static');
const { postProcessMuscriptorResult } = require('./muscriptorPostprocess');
const { createMuscriptorEventCollector } = require('./muscriptorEvents');
const { RegistrationOtpError, createRegistrationOtpService } = require('./registrationOtp');
const { StateConflictError, createStateStore } = require('./stateStore');
const { createArtifactStore } = require('./artifactStore');
const { createDirectUploadService } = require('./directUpload');
const { createJobQueue } = require('./jobQueue');
const { createModelLab } = require('./modelLab');
const { createRunpodServerlessClient } = require('./runpodServerless');
const { localOmrAvailability, runLocalOmr } = require('./localOmr');
const { createPolymathAssistant } = require('./polymathAssistant');
const {
  refundSupportQuestion,
  reserveSupportQuestion,
  supportQuestionAllowance,
} = require('./supportUsage');
const {
  activeVirtualLesson,
  appendSessionMessage,
  createVirtualLesson,
  DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
  endVirtualLesson,
  expireVirtualLessons,
  lessonCatalog,
  lessonQuote,
  normalizeClientRequestId,
  normalizeConversationMode,
  parseTeacherDemonstration,
  publicVirtualLesson,
  sessionIsActive,
  updateSessionMemory,
} = require('./virtualLessons');
const {
  GLOBAL_ROOM_ID,
  canReadRoom,
  canWriteRoom,
  cleanCommunityText,
  ensureGlobalRoom,
  membershipFor,
  publicMessage,
  publicRoom,
  trimRoomMessages,
} = require('./communityChat');
require('dotenv').config({
  path: path.join(__dirname, '.env'),
});

const app = express();
const PORT = Number(process.env.PORT || 3000);
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || 'http://localhost:5173';
const CLIENT_ORIGINS = new Set(
  [CLIENT_ORIGIN, ...String(process.env.CLIENT_ORIGINS || '').split(',')]
    .map((origin) => origin.trim().replace(/\/+$/, ''))
    .filter(Boolean),
);
const IS_PRODUCTION = String(process.env.NODE_ENV || '').trim().toLowerCase() === 'production';
if (!IS_PRODUCTION) {
  CLIENT_ORIGINS.add('http://localhost:5173');
  CLIENT_ORIGINS.add('http://127.0.0.1:5173');
  CLIENT_ORIGINS.add('http://localhost:5174');
  CLIENT_ORIGINS.add('http://127.0.0.1:5174');
}
const REGISTRATION_OTP = createRegistrationOtpService(process.env);
const POLYMATH_ASSISTANT = createPolymathAssistant(process.env);
const PROCESS_INSTANCE_ID = crypto.randomUUID();
const JOB_CLAIM_MS = 7 * 60 * 60 * 1000;
const BUILT_IN_VIRTUAL_TEACHERS = Object.freeze({
  aria: Object.freeze({
    id: 'aria', name: 'Aria', title: 'Piano performance teacher',
    style: 'Calm, warm, precise, and focused on posture, phrasing, and connected movement.',
    voice: 'Warm and precise', voiceType: 'feminine', minimumAge: 0,
    requiresAdultConfirmation: false, adultCompanionEnabled: false,
  }),
  nova: Object.freeze({
    id: 'nova', name: 'Padme', title: 'Expressive performance coach',
    style: 'Warm, confident, affectionate, and focused on expressive melody.',
    voice: 'Warm, expressive, and playfully flirtatious', voiceType: 'feminine', minimumAge: 18,
    requiresAdultConfirmation: true, adultCompanionEnabled: true,
  }),
  anakin: Object.freeze({
    id: 'anakin', name: 'Anakin', title: 'Technique coach',
    style: 'Direct, energetic, and focused on timing, power, and confident movement.',
    voice: 'Focused and assured', voiceType: 'masculine', minimumAge: 0,
    requiresAdultConfirmation: false, adultCompanionEnabled: false,
  }),
  taylor: Object.freeze({
    id: 'taylor', name: 'Taylor', title: 'Songwriting coach',
    style: 'Friendly and thoughtful, with strong melody, phrasing, and storytelling guidance.',
    voice: 'Thoughtful and expressive', voiceType: 'feminine', minimumAge: 0,
    requiresAdultConfirmation: false, adultCompanionEnabled: false,
  }),
  mace: Object.freeze({
    id: 'mace', name: 'Mace Windu', title: 'Piano master',
    style: 'Disciplined, exact, concise, and demanding without empty praise.',
    voice: 'Deep, calm, and exact', voiceType: 'masculine', minimumAge: 0,
    requiresAdultConfirmation: false, adultCompanionEnabled: false,
  }),
});

const PAYPAL_ENV = String(process.env.PAYPAL_ENV || 'live').trim().toLowerCase();
const PAYPAL_API_BASE = PAYPAL_ENV === 'sandbox'
  ? 'https://api-m.sandbox.paypal.com'
  : 'https://api-m.paypal.com';

const DATA_DIR = process.env.POLYMATH_DATA_DIR
  ? path.resolve(process.env.POLYMATH_DATA_DIR)
  : path.join(__dirname, "data");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const DB_PATH = path.join(DATA_DIR, "database.json");

const ARTIFACT_STORE = createArtifactStore({
  localRoot: UPLOAD_DIR,
  bucket: process.env.ARTIFACT_S3_BUCKET,
  region: process.env.ARTIFACT_S3_REGION,
  endpoint: process.env.ARTIFACT_S3_ENDPOINT,
  accessKeyId: process.env.ARTIFACT_S3_ACCESS_KEY_ID,
  secretAccessKey: process.env.ARTIFACT_S3_SECRET_ACCESS_KEY,
});

const DIRECT_UPLOADS = createDirectUploadService({
  artifactStore: ARTIFACT_STORE,
  signingSecret: process.env.DIRECT_UPLOAD_SIGNING_SECRET,
  ttlSeconds: process.env.DIRECT_UPLOAD_TTL_SECONDS,
});

const STATE_STORE = createStateStore({
  databaseUrl: process.env.DATABASE_URL,
  databaseHost: process.env.PGHOST,
  databasePort: process.env.PGPORT,
  databaseUser: process.env.PGUSER,
  databasePassword: process.env.PGPASSWORD,
  databaseName: process.env.PGDATABASE,
  filePath: DB_PATH,
  stateKey: process.env.DATABASE_STATE_KEY || 'primary',
});

const JOB_QUEUE = createJobQueue({
  queueUrl: process.env.JOB_QUEUE_URL,
  region: process.env.JOB_QUEUE_REGION || process.env.AWS_REGION,
});

const WITHDRAWAL_FEE_RATE = 0.25;
const MARKETPLACE_FEE_RATE = 0.25;
const TEACHER_MARKETPLACE_FEE_RATE = 0.25;
const MCOINS_PER_USD = 1;
const READY_SHEET_UPLOAD_MCOIN_COST = 0.5;
const FREE_READY_SHEET_MONTHLY_LIMIT = 2;
const FREE_TRANSLATION_LIMIT = 0;
const CHILL_TRANSLATION_LIMIT = 10;
const MUSICIAN_TRANSLATION_LIMIT = 20;
const FREE_TRANSLATION_MCOIN_COST = 2;
const SUBSCRIBER_TRANSLATION_MCOIN_COST = 0.5;
const TRANSLATION_INITIAL_ESTIMATE_MS = 5 * 60 * 1000;
const TRANSLATION_EXTENSION_MS = 5 * 60 * 1000;
const MAX_PDF_BYTES = 10 * 1024 * 1024;
const DIRECT_UPLOAD_MAX_BYTES = Math.max(
  MAX_PDF_BYTES,
  Math.min(
    5 * 1024 * 1024 * 1024,
    Number(process.env.DIRECT_UPLOAD_MAX_BYTES) || 5 * 1024 * 1024 * 1024,
  ),
);
const MAX_MEDIA_SECONDS = 10 * 60;
const WELCOME_MCOINS = Math.max(0, Math.floor(Number(process.env.WELCOME_MCOINS || 0)));
const MARKETPLACE_MAX_BYTES = 8 * 1024 * 1024;
const VIRTUAL_TEACHER_IMAGE_MAX_BYTES = 8 * 1024 * 1024;
const VIRTUAL_TEACHER_MODEL_MAX_BYTES = 25 * 1024 * 1024;
const BUILT_IN_VIRTUAL_TEACHER_IDS = new Set(Object.keys(BUILT_IN_VIRTUAL_TEACHERS));
const MUSCRIPTOR_ENABLED = String(process.env.MUSCRIPTOR_ENABLED || 'false').trim().toLowerCase() === 'true';
const MUSCRIPTOR_ADMIN_ONLY = String(
  process.env.MUSCRIPTOR_ADMIN_ONLY || (IS_PRODUCTION ? 'true' : 'false'),
).trim().toLowerCase() === 'true';
const MUSCRIPTOR_MODEL = ['small', 'medium', 'large'].includes(
  String(process.env.MUSCRIPTOR_MODEL || 'large').trim().toLowerCase(),
)
  ? String(process.env.MUSCRIPTOR_MODEL || 'large').trim().toLowerCase()
  : 'large';
const MUSCRIPTOR_INFERENCE_VERSION = String(
  process.env.MUSCRIPTOR_INFERENCE_VERSION || 'phase1-v002',
).trim().toLowerCase();
const MUSCRIPTOR_PYTHON = String(process.env.MUSCRIPTOR_PYTHON || '').trim() || (
  process.platform === 'win32'
    ? path.join(os.homedir(), 'muscriptor-eval-env', 'Scripts', 'python.exe')
    : 'python3'
);
const MUSCRIPTOR_WORKER = String(process.env.MUSCRIPTOR_WORKER || '').trim()
  || path.join(__dirname, 'muscriptor_worker.py');
const PIANO_ARRANGER_PYTHON = String(process.env.PIANO_ARRANGER_PYTHON || '').trim()
  || MUSCRIPTOR_PYTHON;
const PIANO_ARRANGER_WORKER = String(process.env.PIANO_ARRANGER_WORKER || '').trim()
  || path.join(__dirname, 'piano_arranger.py');
const MUSCRIPTOR_REMOTE_URL = String(process.env.MUSCRIPTOR_REMOTE_URL || '').trim().replace(/\/+$/, '');
const MUSCRIPTOR_REMOTE_TOKEN = String(process.env.MUSCRIPTOR_REMOTE_TOKEN || '').trim();
const MUSCRIPTOR_TIMEOUT_MS = Math.min(
  12 * 60 * 60 * 1000,
  Math.max(
    10 * 60 * 1000,
    Math.floor(Number(process.env.MUSCRIPTOR_TIMEOUT_MINUTES || 360)) * 60 * 1000,
  ),
);
const FFMPEG_PATH = String(process.env.FFMPEG_PATH || '').trim() || bundledFfmpegPath;
const MEDIA_EXTENSIONS = new Set([
  '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
  '.mp4', '.mov', '.webm', '.mkv', '.avi', '.mpeg', '.mpg',
]);

// Permanent primary GPU path; the SSH-forwarded HTTP worker is only a fallback.
const RUNPOD_SERVERLESS = createRunpodServerlessClient({
  endpointId: process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || process.env.RUNPOD_ENDPOINT_ID,
  apiKey: process.env.RUNPOD_API_KEY,
  volumeId: process.env.RUNPOD_NETWORK_VOLUME_ID,
  region: process.env.RUNPOD_S3_REGION,
  s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
  s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
  s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
  replicas: process.env.RUNPOD_S3_REPLICAS,
  inferenceVersion: MUSCRIPTOR_INFERENCE_VERSION,
  timeoutMs: MUSCRIPTOR_TIMEOUT_MS,
  pollIntervalMs: 2_000,
});

const MODEL_LAB = createModelLab(process.env, {
  dataRoot: path.join(DATA_DIR, 'model-lab'),
  artifactStore: ARTIFACT_STORE,
  inferenceVersion: MUSCRIPTOR_INFERENCE_VERSION,
});

function artifactKey(group, filename) {
  const name = path.basename(String(filename || 'artifact'));
  return ARTIFACT_STORE.remote ? `${group}/${name}` : name;
}

function uploadContentType(filename, suppliedType = '') {
  const extension = path.extname(String(filename || '')).toLowerCase();
  if (extension === '.pdf') return 'application/pdf';
  if (extension === '.json') return 'application/json';
  if (extension === '.mid' || extension === '.midi') return 'audio/midi';
  const normalized = String(suppliedType || '').trim().toLowerCase();
  if (/^(audio|video)\/[a-z0-9.+-]+$/.test(normalized)) return normalized;
  return 'application/octet-stream';
}

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
  'polymath-chill-monthly': {
    id: 'polymath-chill-monthly',
    name: 'Chill',
    price: '7.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'MONTH',
    tier: 'chill',
    translations: CHILL_TRANSLATION_LIMIT,
    mcoins: 0,
  },
  'polymath-chill-yearly': {
    id: 'polymath-chill-yearly',
    name: 'Chill',
    price: '49.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'YEAR',
    tier: 'chill',
    translations: CHILL_TRANSLATION_LIMIT,
    mcoins: 0,
  },
  'polymath-musician-monthly': {
    id: 'polymath-musician-monthly',
    name: 'Musician',
    price: '14.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'MONTH',
    tier: 'musician',
    translations: MUSICIAN_TRANSLATION_LIMIT,
    mcoins: 0,
  },
  'polymath-musician-yearly': {
    id: 'polymath-musician-yearly',
    name: 'Musician',
    price: '93.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'YEAR',
    tier: 'musician',
    translations: MUSICIAN_TRANSLATION_LIMIT,
    mcoins: 0,
  },
  'polymath-institution-class-monthly': {
    id: 'polymath-institution-class-monthly', name: 'Class', price: '300.00', currency: 'USD', kind: 'subscription', recurring: true,
    interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'class', seats: 30, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  'polymath-institution-class-yearly': {
    id: 'polymath-institution-class-yearly', name: 'Class', price: '2880.00', monthlyPrice: '300.00', annualListPrice: '3600.00', annualDiscountPercent: 20,
    currency: 'USD', kind: 'subscription', recurring: true, interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'class', seats: 30, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  'polymath-institution-cohort-monthly': {
    id: 'polymath-institution-cohort-monthly', name: 'Cohort', price: '2250.00', currency: 'USD', kind: 'subscription', recurring: true,
    interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'cohort', seats: 300, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  'polymath-institution-cohort-yearly': {
    id: 'polymath-institution-cohort-yearly', name: 'Cohort', price: '21600.00', monthlyPrice: '2250.00', annualListPrice: '27000.00', annualDiscountPercent: 20,
    currency: 'USD', kind: 'subscription', recurring: true, interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'cohort', seats: 300, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  'polymath-institution-school-monthly': {
    id: 'polymath-institution-school-monthly', name: 'School', price: '7500.00', currency: 'USD', kind: 'subscription', recurring: true,
    interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'school', seats: 1000, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  'polymath-institution-school-yearly': {
    id: 'polymath-institution-school-yearly', name: 'School', price: '72000.00', monthlyPrice: '7500.00', annualListPrice: '90000.00', annualDiscountPercent: 20,
    currency: 'USD', kind: 'subscription', recurring: true, interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'school', seats: 1000, translations: MUSICIAN_TRANSLATION_LIMIT, mcoins: 0,
  },
  // Kept only so existing Pro subscription records remain understandable.
  'polymath-pro': {
    id: 'polymath-pro',
    name: 'Musician (legacy Pro)',
    price: '19.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'MONTH',
    tier: 'musician',
    translations: MUSICIAN_TRANSLATION_LIMIT,
    legacy: true,
    mcoins: 0,
  },
  'mcoins-50': {
    id: 'mcoins-50',
    name: '50 Mcoins',
    price: '50.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 50,
  },
  'mcoins-100': {
    id: 'mcoins-100',
    name: '100 Mcoins',
    price: '100.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 100,
  },
  'mcoins-300': {
    id: 'mcoins-300',
    name: '300 Mcoins',
    price: '300.00',
    currency: 'USD',
    kind: 'mcoins',
    mcoins: 300,
  },
};

const MUSCRIPTOR_INSTRUMENTS = {
  piano: ['acoustic_piano', 'electric_piano'],
  guitar: ['acoustic_guitar'],
  'electric-guitar': ['clean_electric_guitar', 'distorted_electric_guitar'],
  fiddle: ['violin'],
  violin: ['violin'],
  cello: ['cello'],
  'upright-bass': ['contrabass', 'acoustic_bass'],
  banjo: ['acoustic_guitar'],
  mandolin: ['acoustic_guitar'],
  dobro: ['acoustic_guitar'],
  ukulele: ['acoustic_guitar'],
  drums: ['drums'],
  synth: ['synth_lead', 'synth_pad', 'electric_piano'],
  flute: ['flutes'],
  saxophone: ['soprano_and_alto_sax', 'tenor_sax', 'baritone_sax'],
  trumpet: ['trumpet'],
  clarinet: ['clarinet'],
};

const DEFAULT_SITE_POLICIES = Object.freeze({
  registrationEnabled: true,
  minimumSignupAge: 0,
  minimumPasswordLength: 8,
  minimumMarketplacePriceMcoins: 0,
  maximumMarketplacePriceMcoins: 100000,
  marketplaceFeePercent: MARKETPLACE_FEE_RATE * 100,
  teacherDirectoryEnabled: true,
  teacherApplicationsEnabled: true,
  teacherReviewsEnabled: true,
  minimumTeacherHourlyRateMcoins: 0,
  maximumTeacherHourlyRateMcoins: 100000,
  teacherMarketplaceFeePercent: TEACHER_MARKETPLACE_FEE_RATE * 100,
  teacherMarketplaceNotice: '',
  listenerRewardsEnabled: true,
  maximumListenerRewardMcoins: 100,
  maximumRewardOutflowPerListingMcoins: 1000,
  minimumWithdrawalMcoins: 20,
  maximumWithdrawalMcoins: 1000000,
  dailyWithdrawalLimitMcoins: 0,
  maximumPendingWithdrawalOutflowMcoins: 0,
  withdrawalFeePercent: WITHDRAWAL_FEE_RATE * 100,
  welcomeMcoins: WELCOME_MCOINS,
  virtualLessonPricePer30MinutesMcoins: DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
  policyNotice: '',
  termsUrl: '',
  privacyUrl: '',
  supportEmail: '',
  supportPhone: '',
});

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
      personalSongs: [],
      listingReviews: [],
      composerFollows: [],
      messages: [],
      communityRooms: [{
        id: GLOBAL_ROOM_ID,
        name: 'Polymath Free Flow',
        topic: 'Meet musicians, share ideas, and talk about what matters to you.',
        visibility: 'global',
        ownerId: 'platform',
        createdAt: now,
      }],
      communityMemberships: [],
      communityMessages: [],
      communityReports: [],
      teacherProfiles: [],
      teacherReviews: [],
      virtualTeacherCharacters: [],
      virtualLessonSessions: [],
      withdrawals: [],
      paymentOrders: [],
      subscriptions: [],
      webhookEvents: [],
      scoreTranslationJobs: [],
      mediaTranscriptionJobs: [],
      bands: [],
      bandMemberships: [],
      bandMessages: [],
      ledger: [],
      promotions: [],
      promotionRedemptions: [],
      institutions: [],
      institutionMemberships: [],
      passwordResetEvents: [],
      authEvents: [],
      registrationVerifications: [],
      settings: { ...DEFAULT_SITE_POLICIES },
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
    'personalSongs',
    'listingReviews',
    'composerFollows',
    'messages',
    'communityRooms',
    'communityMemberships',
    'communityMessages',
    'communityReports',
    'teacherProfiles',
    'teacherReviews',
    'virtualTeacherCharacters',
    'virtualLessonSessions',
    'withdrawals',
    'paymentOrders',
    'subscriptions',
    'webhookEvents',
    'scoreTranslationJobs',
    'mediaTranscriptionJobs',
    'bands',
    'bandMemberships',
    'bandMessages',
    'ledger',
    'promotions',
    'promotionRedemptions',
    'institutions',
    'institutionMemberships',
    'passwordResetEvents',
    'authEvents',
    'registrationVerifications',
  ];
  arrays.forEach((key) => {
    if (!Array.isArray(normalized[key])) normalized[key] = [];
  });
  normalized.sessions.forEach((session) => {
    if (!session.tokenHash && session.token) {
      session.tokenHash = hashSessionToken(session.token);
      delete session.token;
    }
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
  normalized.listings.forEach((listing) => {
    if (!listing.listingMode) {
      listing.listingMode = Number(listing.priceMcoins || 0) > 0 ? 'sale' : 'free';
    }
    listing.priceMcoins = Number(listing.priceMcoins || 0);
    listing.listenerRewardMcoins = Number(listing.listenerRewardMcoins || 0);
    listing.rewardPaidMcoins = Number(listing.rewardPaidMcoins || 0);
    if (!Number.isFinite(Number(listing.marketplaceFeeRate))) {
      listing.marketplaceFeeRate = MARKETPLACE_FEE_RATE;
    }
  });
  normalized.promotions.forEach((promotion) => {
    if (!['marketplace_percent', 'marketplace_fixed', 'friend_id_percent', 'subscription_percent'].includes(promotion.kind)) {
      promotion.active = false;
      promotion.retired = true;
    }
  });
  ensureFriendIds(normalized.users);
  normalized.settings = {
    ...DEFAULT_SITE_POLICIES,
    ...(normalized.settings && typeof normalized.settings === 'object' ? normalized.settings : {}),
  };
  if (!normalized.settings.minimumWithdrawal20MigrationApplied) {
    normalized.settings.minimumWithdrawalMcoins = 20;
    normalized.settings.minimumWithdrawal20MigrationApplied = true;
  }
  ensureGlobalRoom(normalized);
  return normalized;
}

async function readDb() {
  ensureStorage();
  const seed = JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
  return normalizeDb(await STATE_STORE.read(seed));
}

async function writeDb(db) {
  await STATE_STORE.write(db);
}

function id(prefix) {
  return `${prefix}_${crypto.randomUUID()}`;
}

function friendIdFromSeed(seed, attempt = 0) {
  const suffix = crypto.createHash('sha256')
    .update(`${seed}:${attempt}`)
    .digest('hex')
    .slice(0, 5);
  return `user_${suffix}`;
}

function createFriendId(users, seed = crypto.randomUUID()) {
  const used = new Set(users.map((user) => String(user.friendId || '').toLowerCase()).filter(Boolean));
  for (let attempt = 0; attempt < 1000; attempt += 1) {
    const candidate = friendIdFromSeed(seed, attempt);
    if (!used.has(candidate)) return candidate;
  }
  throw new Error('Could not allocate a unique Friend ID.');
}

function ensureFriendIds(users) {
  const used = new Set();
  users.forEach((user) => {
    if (user.id === 'platform') return;
    const current = String(user.friendId || '').trim().toLowerCase();
    if (/^user_[a-f0-9]{5}$/.test(current) && !used.has(current)) {
      user.friendId = current;
      used.add(current);
      return;
    }
    let attempt = 0;
    let candidate;
    do {
      candidate = friendIdFromSeed(user.id, attempt);
      attempt += 1;
    } while (used.has(candidate));
    user.friendId = candidate;
    used.add(candidate);
  });
}

function normalizeFriendId(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return /^user_[a-f0-9]{5}$/.test(normalized) ? normalized : '';
}

function normalizeInstitutionCode(value) {
  return String(value || '').trim().toUpperCase().replace(/[^A-Z0-9-]/g, '').slice(0, 32);
}

function createInstitutionAccessCode(db, planName) {
  const prefix = String(planName || 'MUSIC').toUpperCase().replace(/[^A-Z]/g, '').slice(0, 8) || 'MUSIC';
  const used = new Set(db.institutions.map((item) => item.accessCode));
  for (let attempt = 0; attempt < 1000; attempt += 1) {
    const suffix = crypto.randomBytes(4).toString('hex').toUpperCase().slice(0, 6);
    const candidate = `${prefix}-${suffix}`;
    if (!used.has(candidate)) return candidate;
  }
  throw new Error('Could not allocate a unique institution access code.');
}

function activeInstitutionByCode(db, value) {
  const code = normalizeInstitutionCode(value);
  if (!code) return null;
  return db.institutions.find((item) => item.accessCode === code && item.status === 'ACTIVE') || null;
}

function institutionSeatCount(db, institutionId) {
  return db.institutionMemberships.filter((item) => (
    item.institutionId === institutionId && item.role === 'member' && item.status === 'ACTIVE'
  )).length;
}

function applyInstitutionToUser(user, institution, role) {
  user.institutionId = institution.id;
  user.institutionName = institution.name;
  user.institutionPlan = institution.plan;
  user.institutionRole = role;
  user.institutionStatus = institution.status;
  user.institutionSeatLimit = institution.seatLimit;
  user.institutionAccessCode = role === 'owner' ? institution.accessCode : '';
}

function addInstitutionMembership(db, user, institution, role = 'member') {
  let membership = db.institutionMemberships.find((item) => (
    item.institutionId === institution.id && item.userId === user.id
  ));
  if (!membership) {
    membership = {
      id: id('institution_member'),
      institutionId: institution.id,
      userId: user.id,
      role,
      status: 'ACTIVE',
      joinedAt: new Date().toISOString(),
    };
    db.institutionMemberships.push(membership);
  } else {
    membership.role = role;
    membership.status = 'ACTIVE';
  }
  applyInstitutionToUser(user, institution, role);
  return membership;
}

function syncInstitutionSubscription(db, record, user, product, active) {
  if (!product?.institutionTier) return null;
  let institution = db.institutions.find((item) => item.subscriptionId === record.subscriptionId);
  if (!institution && active) {
    institution = {
      id: id('institution'),
      ownerUserId: user.id,
      subscriptionId: record.subscriptionId,
      name: `${user.name}'s ${product.name}`,
      plan: product.institutionTier,
      seatLimit: product.seats,
      accessCode: createInstitutionAccessCode(db, product.name),
      status: 'ACTIVE',
      createdAt: new Date().toISOString(),
    };
    db.institutions.push(institution);
  }
  if (!institution) return null;
  institution.status = active ? 'ACTIVE' : 'INACTIVE';
  institution.plan = product.institutionTier;
  institution.seatLimit = product.seats;
  institution.updatedAt = new Date().toISOString();
  db.institutionMemberships
    .filter((item) => item.institutionId === institution.id)
    .forEach((membership) => {
      membership.status = institution.status;
      const member = db.users.find((candidate) => candidate.id === membership.userId);
      if (member) applyInstitutionToUser(member, institution, membership.role);
    });
  if (active) addInstitutionMembership(db, user, institution, 'owner');
  return institution;
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return { salt, hash };
}

async function bootstrapAdminAccounts() {
  const password = String(process.env.ADMIN_PASSWORD || '');
  if (!password) return { created: 0 };
  if (ADMIN_EMAILS.size === 0) throw new Error('ADMIN_PASSWORD requires at least one address in ADMIN_EMAILS.');
  if (password.length < 12) throw new Error('ADMIN_PASSWORD must contain at least 12 characters.');

  const db = await readDb();
  const now = new Date().toISOString();
  let created = 0;
  ADMIN_EMAILS.forEach((email) => {
    if (db.users.some((user) => String(user.email || '').trim().toLowerCase() === email)) return;
    const userId = id('user');
    const { salt, hash } = hashPassword(password);
    db.users.push({
      id: userId,
      friendId: createFriendId(db.users, userId),
      name: 'Polymath Administrator',
      email,
      phone: '',
      passwordHash: hash,
      passwordSalt: salt,
      mustChangePassword: true,
      mcoins: 0,
      withdrawableMcoins: 0,
      pro: false,
      proStatus: 'INACTIVE',
      subscriptionTier: '',
      subscriptionInterval: null,
      subscriptionStartedAt: null,
      paypalSubscriptionId: null,
      translationUsage: { period: currentTranslationPeriod(), includedUsed: 0 },
      readySheetUploadUsage: { period: currentTranslationPeriod(), freeUsed: 0 },
      passwordBootstrappedAt: now,
      createdAt: now,
    });
    db.authEvents.push({ id: id('auth'), userId, type: 'admin_bootstrap', createdAt: now });
    created += 1;
  });
  if (created > 0) {
    db.authEvents = db.authEvents.slice(-5000);
    await writeDb(db);
  }
  return { created };
}

function hashSessionToken(token) {
  return crypto.createHash('sha256').update(String(token || '')).digest('hex');
}

function createSession(db, userId) {
  const token = crypto.randomBytes(32).toString('hex');
  db.sessions.push({
    tokenHash: hashSessionToken(token),
    userId,
    expiresAt: new Date(Date.now() + 30 * 86400000).toISOString(),
    createdAt: new Date().toISOString(),
  });
  return token;
}

function clampInteger(value, minimum, maximum, fallback) {
  const number = Math.floor(Number(value));
  if (!Number.isFinite(number)) return fallback;
  return Math.min(maximum, Math.max(minimum, number));
}

function clampDecimal(value, minimum, maximum, fallback, decimals = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  const clamped = Math.min(maximum, Math.max(minimum, number));
  return Number(clamped.toFixed(decimals));
}

function mcoinAmount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Number(number.toFixed(2)) : Number.NaN;
}

function sitePolicies(db) {
  const raw = db?.settings && typeof db.settings === 'object' ? db.settings : {};
  return {
    registrationEnabled: raw.registrationEnabled !== false,
    minimumSignupAge: clampInteger(raw.minimumSignupAge, 0, 120, DEFAULT_SITE_POLICIES.minimumSignupAge),
    minimumPasswordLength: clampInteger(raw.minimumPasswordLength, 1, 256, DEFAULT_SITE_POLICIES.minimumPasswordLength),
    minimumMarketplacePriceMcoins: clampDecimal(raw.minimumMarketplacePriceMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.minimumMarketplacePriceMcoins),
    maximumMarketplacePriceMcoins: clampDecimal(raw.maximumMarketplacePriceMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.maximumMarketplacePriceMcoins),
    marketplaceFeePercent: clampDecimal(raw.marketplaceFeePercent, 0, 100, DEFAULT_SITE_POLICIES.marketplaceFeePercent),
    teacherDirectoryEnabled: raw.teacherDirectoryEnabled !== false,
    teacherApplicationsEnabled: raw.teacherApplicationsEnabled !== false,
    teacherReviewsEnabled: raw.teacherReviewsEnabled !== false,
    minimumTeacherHourlyRateMcoins: clampDecimal(
      raw.minimumTeacherHourlyRateMcoins,
      0,
      1000000000,
      DEFAULT_SITE_POLICIES.minimumTeacherHourlyRateMcoins,
    ),
    maximumTeacherHourlyRateMcoins: clampDecimal(
      raw.maximumTeacherHourlyRateMcoins,
      0,
      1000000000,
      DEFAULT_SITE_POLICIES.maximumTeacherHourlyRateMcoins,
    ),
    teacherMarketplaceFeePercent: clampDecimal(
      raw.teacherMarketplaceFeePercent,
      0,
      100,
      DEFAULT_SITE_POLICIES.teacherMarketplaceFeePercent,
    ),
    teacherMarketplaceNotice: String(raw.teacherMarketplaceNotice || '').trim().slice(0, 600),
    listenerRewardsEnabled: raw.listenerRewardsEnabled !== false,
    maximumListenerRewardMcoins: clampDecimal(raw.maximumListenerRewardMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.maximumListenerRewardMcoins),
    maximumRewardOutflowPerListingMcoins: clampDecimal(raw.maximumRewardOutflowPerListingMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.maximumRewardOutflowPerListingMcoins),
    minimumWithdrawalMcoins: clampDecimal(raw.minimumWithdrawalMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.minimumWithdrawalMcoins),
    maximumWithdrawalMcoins: clampDecimal(raw.maximumWithdrawalMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.maximumWithdrawalMcoins),
    dailyWithdrawalLimitMcoins: clampDecimal(raw.dailyWithdrawalLimitMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.dailyWithdrawalLimitMcoins),
    maximumPendingWithdrawalOutflowMcoins: clampDecimal(raw.maximumPendingWithdrawalOutflowMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.maximumPendingWithdrawalOutflowMcoins),
    withdrawalFeePercent: clampDecimal(raw.withdrawalFeePercent, 0, 100, DEFAULT_SITE_POLICIES.withdrawalFeePercent),
    welcomeMcoins: clampDecimal(raw.welcomeMcoins, 0, 1000000000, DEFAULT_SITE_POLICIES.welcomeMcoins),
    virtualLessonPricePer30MinutesMcoins: clampDecimal(
      raw.virtualLessonPricePer30MinutesMcoins,
      0,
      1000000000,
      clampDecimal(
        raw.virtualLessonPricesMcoins?.[30],
        0,
        1000000000,
        DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
      ),
    ),
    policyNotice: String(raw.policyNotice || '').trim().slice(0, 1000),
    termsUrl: String(raw.termsUrl || '').trim().slice(0, 500),
    privacyUrl: String(raw.privacyUrl || '').trim().slice(0, 500),
    supportEmail: String(raw.supportEmail || '').trim().toLowerCase().slice(0, 254),
    supportPhone: String(raw.supportPhone || '').trim().replace(/[^+\d() .-]/g, '').slice(0, 40),
    updatedAt: raw.updatedAt || null,
    updatedBy: raw.updatedBy || null,
  };
}

function ageOnDate(birthDate, now = new Date()) {
  const born = new Date(`${birthDate}T00:00:00Z`);
  if (Number.isNaN(born.getTime()) || born > now) return -1;
  let age = now.getUTCFullYear() - born.getUTCFullYear();
  const beforeBirthday = now.getUTCMonth() < born.getUTCMonth()
    || (now.getUTCMonth() === born.getUTCMonth() && now.getUTCDate() < born.getUTCDate());
  if (beforeBirthday) age -= 1;
  return age;
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

function activeSubscriptionTier(user) {
  if (String(user?.institutionStatus || '').toUpperCase() === 'ACTIVE') return 'musician';
  const administratorGrant = activeAdminSubscriptionGrant(user);
  if (administratorGrant) return administratorGrant.tier;
  const explicitTier = String(user?.subscriptionTier || '').toLowerCase();
  if (['chill', 'musician'].includes(explicitTier)
    && String(user?.proStatus || '').toUpperCase() === 'ACTIVE') {
    return explicitTier;
  }
  // Existing Pro members migrate to Musician without losing access.
  if (user?.pro) return 'musician';
  return 'free';
}

function activeAdminSubscriptionGrant(user, now = new Date()) {
  const grant = user?.adminSubscriptionGrant;
  const tier = String(grant?.tier || '').toLowerCase();
  const expiresAt = new Date(grant?.expiresAt || '');
  if (!['chill', 'musician'].includes(tier) || Number.isNaN(expiresAt.getTime()) || expiresAt <= now) {
    return null;
  }
  return grant;
}

function effectiveSubscriptionInterval(user) {
  return activeAdminSubscriptionGrant(user)?.interval || user?.subscriptionInterval || null;
}

function effectiveSubscriptionStartedAt(user) {
  return activeAdminSubscriptionGrant(user)?.startedAt || user?.subscriptionStartedAt || null;
}

function hasMusicianAccess(user) {
  return isAdministrator(user) || activeSubscriptionTier(user) === 'musician';
}

function ensureReadySheetUsage(user, now = new Date()) {
  const period = currentTranslationPeriod(now);
  if (!user.readySheetUploadUsage || user.readySheetUploadUsage.period !== period) {
    user.readySheetUploadUsage = { period, freeUsed: 0 };
  }
  user.readySheetUploadUsage.freeUsed = Math.max(0, Number(user.readySheetUploadUsage.freeUsed || 0));
  return user.readySheetUploadUsage;
}

function readySheetAllowance(user, now = new Date()) {
  if (isAdministrator(user) || activeSubscriptionTier(user) !== 'free') {
    return {
      unlimited: true,
      limit: null,
      used: 0,
      remaining: null,
      resetAt: null,
      overageCostMcoins: 0,
    };
  }
  const usage = ensureReadySheetUsage(user, now);
  return {
    unlimited: false,
    limit: FREE_READY_SHEET_MONTHLY_LIMIT,
    used: Math.min(FREE_READY_SHEET_MONTHLY_LIMIT, usage.freeUsed),
    remaining: Math.max(0, FREE_READY_SHEET_MONTHLY_LIMIT - usage.freeUsed),
    resetAt: nextTranslationResetAt(now),
    overageCostMcoins: READY_SHEET_UPLOAD_MCOIN_COST,
  };
}

function readySheetUploadCost(user, now = new Date()) {
  const allowance = readySheetAllowance(user, now);
  return allowance.unlimited || allowance.remaining > 0 ? 0 : READY_SHEET_UPLOAD_MCOIN_COST;
}

function chargeReadySheetUpload(db, user, filename, now = new Date()) {
  const allowance = readySheetAllowance(user, now);
  if (allowance.unlimited) {
    addLedger(db, user.id, 0, isAdministrator(user) ? 'admin_ready_sheet_upload' : 'subscriber_ready_sheet_upload', filename);
    return { costMcoins: 0, paymentMethod: 'unlimited' };
  }
  if (allowance.remaining > 0) {
    ensureReadySheetUsage(user, now).freeUsed += 1;
    addLedger(db, user.id, 0, 'free_ready_sheet_upload', filename);
    return { costMcoins: 0, paymentMethod: 'free_attempt' };
  }
  const costMcoins = READY_SHEET_UPLOAD_MCOIN_COST;
  if (costMcoins > 0 && Number(user.mcoins || 0) < costMcoins) return null;
  user.mcoins = Number((Number(user.mcoins || 0) - costMcoins).toFixed(2));
  addLedger(db, user.id, -costMcoins, 'ready_sheet_upload', filename);
  return { costMcoins, paymentMethod: 'mcoins' };
}

function translationMcoinCost(user) {
  return activeSubscriptionTier(user) === 'free'
    ? FREE_TRANSLATION_MCOIN_COST
    : SUBSCRIBER_TRANSLATION_MCOIN_COST;
}

function utcMonthAnniversary(anchor, monthsToAdd) {
  const targetYear = anchor.getUTCFullYear() + Math.floor((anchor.getUTCMonth() + monthsToAdd) / 12);
  const targetMonth = ((anchor.getUTCMonth() + monthsToAdd) % 12 + 12) % 12;
  const daysInMonth = new Date(Date.UTC(targetYear, targetMonth + 1, 0)).getUTCDate();
  return new Date(Date.UTC(
    targetYear,
    targetMonth,
    Math.min(anchor.getUTCDate(), daysInMonth),
    anchor.getUTCHours(),
    anchor.getUTCMinutes(),
    anchor.getUTCSeconds(),
    anchor.getUTCMilliseconds(),
  ));
}

function translationUsageWindow(user, now = new Date()) {
  const tier = activeSubscriptionTier(user);
  const anchor = new Date(effectiveSubscriptionStartedAt(user) || '');
  if (tier === 'free' || Number.isNaN(anchor.getTime()) || anchor > now) {
    return {
      key: currentTranslationPeriod(now),
      resetAt: nextTranslationResetAt(now),
    };
  }
  let monthIndex = Math.max(0,
    (now.getUTCFullYear() - anchor.getUTCFullYear()) * 12
    + now.getUTCMonth() - anchor.getUTCMonth());
  if (utcMonthAnniversary(anchor, monthIndex) > now) monthIndex = Math.max(0, monthIndex - 1);
  const periodStart = utcMonthAnniversary(anchor, monthIndex);
  return {
    key: `subscription:${tier}:${periodStart.toISOString()}`,
    resetAt: utcMonthAnniversary(anchor, monthIndex + 1).toISOString(),
  };
}

function ensureTranslationUsage(user, now = new Date()) {
  const period = translationUsageWindow(user, now).key;
  if (!user.translationUsage || user.translationUsage.period !== period) {
    user.translationUsage = {
      period,
      includedUsed: 0,
    };
  }
  if (!Number.isFinite(Number(user.translationUsage.includedUsed))) {
    const legacyUsed = activeSubscriptionTier(user) === 'musician'
      ? user.translationUsage.proUsed
      : user.translationUsage.freeUsed;
    user.translationUsage.includedUsed = legacyUsed;
  }
  user.translationUsage.includedUsed = Math.max(0, Number(user.translationUsage.includedUsed || 0));
  return user.translationUsage;
}

function isAdministrator(user) {
  return ADMIN_EMAILS.has(String(user?.email || '').toLowerCase());
}

function hasUnlimitedMcoins(user) {
  return isAdministrator(user);
}

function translationAllowance(user, now = new Date()) {
  if (isAdministrator(user)) {
    return {
      plan: 'admin',
      unlimited: true,
      limit: null,
      used: 0,
      remaining: null,
      resetAt: null,
    };
  }
  const usage = ensureTranslationUsage(user, now);
  const plan = activeSubscriptionTier(user);
  const limits = {
    free: FREE_TRANSLATION_LIMIT,
    chill: CHILL_TRANSLATION_LIMIT,
    musician: MUSICIAN_TRANSLATION_LIMIT,
  };
  const limit = limits[plan];
  const used = usage.includedUsed;
  return {
    plan,
    unlimited: false,
    limit,
    used: Math.min(limit, used),
    remaining: Math.max(0, limit - used),
    resetAt: translationUsageWindow(user, now).resetAt,
    overageCostMcoins: translationMcoinCost(user),
  };
}

function deductTranslationAllowance(user) {
  const allowance = translationAllowance(user);
  if (allowance.unlimited) return 'admin';
  if (allowance.remaining <= 0) return null;
  user.translationUsage.includedUsed += 1;
  return 'included';
}

function restoreTranslationAllowance(user, bucket) {
  ensureTranslationUsage(user);
  if (['included', 'pro', 'free'].includes(bucket)) {
    user.translationUsage.includedUsed = Math.max(0, user.translationUsage.includedUsed - 1);
  }
}

function safeUser(user) {
  const subscriptionTier = activeSubscriptionTier(user);
  const administrator = isAdministrator(user);
  const administratorGrant = activeAdminSubscriptionGrant(user);
  return {
    user_id: user.id,
    friend_id: user.friendId || '',
    name: user.name,
    avatarUrl: user.avatarUrl || '',
    email: user.email,
    phone: user.phone || '',
    mcoins: user.mcoins,
    unlimitedMcoins: hasUnlimitedMcoins(user),
    withdrawableMcoins: Number(user.withdrawableMcoins || 0),
    cashoutEligibleMcoins: Number(user.mcoins || 0),
    pro: subscriptionTier !== 'free',
    subscriptionTier,
    subscriptionInterval: effectiveSubscriptionInterval(user),
    subscriptionStartedAt: effectiveSubscriptionStartedAt(user),
    proStatus: administratorGrant ? 'ACTIVE' : (user.proStatus || (user.pro ? 'ACTIVE' : 'INACTIVE')),
    adminSubscriptionGrant: user.adminSubscriptionGrant ? {
      tier: String(user.adminSubscriptionGrant.tier || '').toLowerCase(),
      interval: String(user.adminSubscriptionGrant.interval || '').toUpperCase(),
      startedAt: user.adminSubscriptionGrant.startedAt || null,
      expiresAt: user.adminSubscriptionGrant.expiresAt || null,
      active: Boolean(administratorGrant),
    } : null,
    paypalSubscriptionId: user.paypalSubscriptionId || null,
    luckyCodeApplied: Boolean(user.luckyCodeClaim),
    institution: user.institutionId ? {
      id: user.institutionId,
      name: user.institutionName || '',
      plan: user.institutionPlan || '',
      role: user.institutionRole || 'member',
      status: user.institutionStatus || 'INACTIVE',
      seats: Number(user.institutionSeatLimit || 0),
      accessCode: user.institutionRole === 'owner' ? (user.institutionAccessCode || '') : '',
    } : null,
    translationAllowance: translationAllowance(user),
    readySheetAllowance: readySheetAllowance(user),
    readySheetUploadCostMcoins: readySheetUploadCost(user),
    adultCompanionConfirmed: Boolean(user.adultCompanionConfirmedAt),
    admin: administrator,
    access: {
      regular: true,
      learn: administrator || subscriptionTier === 'musician',
      band: administrator || subscriptionTier === 'musician',
      community: administrator || subscriptionTier !== 'free',
    },
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
  const tokenHash = hashSessionToken(token);
  const session = db.sessions.find((item) => item.tokenHash === tokenHash || item.token === token);
  if (!session || new Date(session.expiresAt).getTime() < Date.now()) return null;
  return db.users.find((user) => user.id === session.userId) || null;
}

async function requireAuth(req, res, next) {
  const db = await readDb();
  const user = authUser(req, db);
  if (!user) return res.status(401).json({ error: 'Please sign in first.' });
  req.db = db;
  req.user = user;
  next();
}

function requireMusician(req, res, next) {
  return requireAuth(req, res, () => {
    if (!hasMusicianAccess(req.user)) {
      res.status(403).json({ error: 'Band access requires the Musician plan.' });
      return;
    }
    next();
  });
}

function requireAdmin(req, res, next) {
  if (ADMIN_EMAILS.size === 0) {
    return res.status(503).json({ error: 'Admin access is not configured. Set ADMIN_EMAILS on the backend.' });
  }
  if (!isAdministrator(req.user)) {
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
      purchase: listing?.title || purchase.listingId || 'Composers purchase',
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

function normalizePromotionCode(value) {
  return String(value || '').trim().toUpperCase().replace(/[^A-Z0-9_-]/g, '').slice(0, 32);
}

function promotionRedemptionCounts(db, promotionId, userId = '') {
  const matching = db.promotionRedemptions.filter((entry) => entry.promotionId === promotionId);
  return {
    total: matching.length,
    user: userId ? matching.filter((entry) => entry.userId === userId).length : 0,
  };
}

function publicPromotion(promotion, db) {
  const counts = promotionRedemptionCounts(db, promotion.id);
  return {
    id: promotion.id,
    code: promotion.code,
    name: promotion.name,
    kind: promotion.kind,
    value: promotion.value,
    minimumSpendMcoins: promotion.minimumSpendMcoins,
    minimumAccountAgeDays: promotion.minimumAccountAgeDays,
    maxRedemptions: promotion.maxRedemptions,
    perUserLimit: promotion.perUserLimit,
    startsAt: promotion.startsAt || null,
    expiresAt: promotion.expiresAt || null,
    active: promotion.active !== false,
    retired: promotion.retired === true,
    redemptionCount: counts.total,
    createdAt: promotion.createdAt,
    updatedAt: promotion.updatedAt || null,
  };
}

function promotionForUse(db, code, user, expectedKinds, spendMcoins = 0) {
  const normalizedCode = normalizePromotionCode(code);
  if (!normalizedCode) return { promotion: null, error: null };
  const promotion = db.promotions.find((item) => item.code === normalizedCode);
  if (!promotion) return { promotion: null, error: 'That voucher or coupon code is not valid.' };
  if (promotion.active === false) return { promotion: null, error: 'That promotion is inactive.' };
  if (!expectedKinds.includes(promotion.kind)) {
    return { promotion: null, error: promotion.retired
      ? 'That legacy promotion has been retired.'
      : 'This promotion cannot be used for this purchase.' };
  }
  const now = Date.now();
  if (promotion.startsAt && new Date(promotion.startsAt).getTime() > now) {
    return { promotion: null, error: 'That promotion has not started yet.' };
  }
  if (promotion.expiresAt && new Date(promotion.expiresAt).getTime() < now) {
    return { promotion: null, error: 'That promotion has expired.' };
  }
  const counts = promotionRedemptionCounts(db, promotion.id, user.id);
  if (promotion.maxRedemptions > 0 && counts.total >= promotion.maxRedemptions) {
    return { promotion: null, error: 'That promotion has reached its redemption limit.' };
  }
  if (promotion.perUserLimit > 0 && counts.user >= promotion.perUserLimit) {
    return { promotion: null, error: 'You have already used that promotion.' };
  }
  if (promotion.minimumAccountAgeDays > 0) {
    const accountAge = (now - new Date(user.createdAt).getTime()) / 86400000;
    if (!Number.isFinite(accountAge) || accountAge < promotion.minimumAccountAgeDays) {
      return { promotion: null, error: `This promotion requires an account at least ${promotion.minimumAccountAgeDays} days old.` };
    }
  }
  if (promotion.minimumSpendMcoins > 0 && spendMcoins < promotion.minimumSpendMcoins) {
    return { promotion: null, error: `This coupon requires a minimum spend of ${promotion.minimumSpendMcoins} Mcoins.` };
  }
  return { promotion, error: null };
}

function friendPromotionForUse(db, friendId, user, spendMcoins = 0) {
  const normalizedFriendId = normalizeFriendId(friendId);
  if (!normalizedFriendId) {
    return { promotion: null, friendUser: null, error: 'Enter a Friend ID in the format user_aa123.' };
  }
  const friendUser = db.users.find((candidate) => (
    candidate.id !== 'platform' && candidate.friendId === normalizedFriendId
  ));
  if (!friendUser) {
    return { promotion: null, friendUser: null, error: 'That Friend ID does not belong to a registered user.' };
  }
  if (friendUser.id === user.id) {
    return { promotion: null, friendUser: null, error: 'Use a friend’s ID, not your own Friend ID.' };
  }
  const promotion = db.promotions
    .filter((item) => item.kind === 'friend_id_percent' && item.active !== false)
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0];
  if (!promotion) {
    return { promotion: null, friendUser: null, error: 'The Friend ID voucher is not active right now.' };
  }
  const result = promotionForUse(
    db,
    promotion.code,
    user,
    ['friend_id_percent'],
    spendMcoins,
  );
  return { ...result, friendUser };
}

function recordPromotionRedemption(db, promotion, user, details = {}) {
  db.promotionRedemptions.push({
    id: id('promo_use'),
    promotionId: promotion.id,
    code: promotion.code,
    userId: user.id,
    ...details,
    createdAt: new Date().toISOString(),
  });
}

function promotionSignupAvailability(db, promotion) {
  if (!promotion || promotion.active === false) return 'That Lucky code is not active.';
  const now = Date.now();
  if (promotion.startsAt && new Date(promotion.startsAt).getTime() > now) return 'That Lucky code is not active yet.';
  if (promotion.expiresAt && new Date(promotion.expiresAt).getTime() < now) return 'That Lucky code has expired.';
  const counts = promotionRedemptionCounts(db, promotion.id);
  if (promotion.maxRedemptions > 0 && counts.total >= promotion.maxRedemptions) {
    return 'That Lucky code has reached its usage limit.';
  }
  if (promotion.minimumAccountAgeDays > 0) {
    return 'That Lucky code is not available during registration.';
  }
  return '';
}

function resolveSignupLuckyCode(db, value) {
  const raw = String(value || '').trim();
  if (!raw) return { claim: null, error: '' };

  const institution = activeInstitutionByCode(db, raw);
  if (institution) {
    if (institutionSeatCount(db, institution.id) >= institution.seatLimit) {
      return { claim: null, error: 'That institution has no seats remaining.' };
    }
    return { claim: { type: 'institution', institution }, error: '' };
  }

  const friendId = normalizeFriendId(raw);
  if (friendId) {
    const friendUser = db.users.find((candidate) => candidate.id !== 'platform' && candidate.friendId === friendId);
    if (!friendUser) return { claim: null, error: 'That Lucky code is not valid.' };
    const promotion = db.promotions
      .filter((item) => item.kind === 'friend_id_percent' && item.active !== false)
      .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0];
    const error = promotionSignupAvailability(db, promotion);
    if (error) return { claim: null, error };
    return { claim: { type: 'subscription_percent', promotion, friendId }, error: '' };
  }

  const promotionCode = normalizePromotionCode(raw);
  const promotion = db.promotions.find((item) => item.code === promotionCode);
  if (!promotion || promotion.kind !== 'subscription_percent') {
    return { claim: null, error: 'That Lucky code is not valid.' };
  }
  const error = promotionSignupAvailability(db, promotion);
  if (error) return { claim: null, error };
  return {
    claim: {
      type: 'subscription_percent',
      promotion,
    },
    error: '',
  };
}

function applySignupLuckyCode(db, user, claim) {
  if (!claim) return;
  if (claim.type === 'institution') {
    addInstitutionMembership(db, user, claim.institution, 'member');
    user.luckyCodeClaim = {
      type: 'institution',
      institutionId: claim.institution.id,
      claimedAt: new Date().toISOString(),
    };
    return;
  }

  const promotion = claim.promotion;
  user.luckyCodeClaim = {
    type: claim.type,
    promotionId: promotion.id,
    code: promotion.code,
    value: promotion.value,
    friendId: claim.friendId || null,
    claimedAt: new Date().toISOString(),
  };
  recordPromotionRedemption(db, promotion, user, {
    source: 'registration_lucky_code',
    friendId: claim.friendId || null,
    creditedMcoins: 0,
    subscriptionDiscountPercent: promotion.value,
  });
}

function subscriptionPriceForUser(product, user) {
  const claim = user?.luckyCodeClaim;
  if (!claim || claim.type !== 'subscription_percent' || !Number.isFinite(Number(claim.value))) {
    return { price: Number(product.price).toFixed(2), discountPercent: 0, luckyCode: '' };
  }
  const discountPercent = Math.min(100, Math.max(0, Number(claim.value)));
  const baseCents = Math.round(Number(product.price) * 100);
  const discountedCents = Math.max(1, Math.round(baseCents * (100 - discountPercent) / 100));
  return { price: (discountedCents / 100).toFixed(2), discountPercent, luckyCode: claim.code || claim.friendId || '' };
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

function requireSubscriber(req, res, next) {
  return requireAuth(req, res, () => {
    if (!isAdministrator(req.user) && activeSubscriptionTier(req.user) === 'free') {
      res.status(403).json({
        error: 'Community chat is included with Chill and Musician subscriptions.',
        code: 'SUBSCRIPTION_REQUIRED',
      });
      return;
    }
    next();
  });
  normalized.virtualTeacherCharacters.forEach((character) => {
    character.builtIn = BUILT_IN_VIRTUAL_TEACHER_IDS.has(String(character.id || ''));
    character.active = character.active !== false;
    character.adultCompanionEnabled = Boolean(
      character.adultCompanionEnabled ?? character.requiresAdultConfirmation,
    );
    character.minimumAge = normalizeVirtualTeacherMinimumAge(
      character.minimumAge,
      character.requiresAdultConfirmation,
    );
    if (character.adultCompanionEnabled) character.minimumAge = Math.max(18, character.minimumAge);
    character.requiresAdultConfirmation = character.minimumAge >= 18;
    character.pricePer30MinutesMcoins = normalizeVirtualTeacherPrice(
      character.pricePer30MinutesMcoins,
    );
  });
}

function readySheetFormat(filename) {
  const extension = path.extname(String(filename || '')).toLowerCase();
  if (extension === '.json') return 'JSON';
  if (extension === '.mid' || extension === '.midi') return 'MIDI';
  return '';
}

function readySheetMetadata(bytes, format, fallback = {}) {
  if (format !== 'JSON') return {
    title: String(fallback.title || '').trim(),
    artist: String(fallback.artist || '').trim(),
  };
  const parsed = JSON.parse(bytes.toString('utf8'));
  if (!Array.isArray(parsed) && !Array.isArray(parsed?.notes)
    && !Array.isArray(parsed?.events) && !Array.isArray(parsed?.tabs)) {
    throw new Error('Ready-to-play JSON must contain notes, events, or tabs.');
  }
  return {
    title: String(fallback.title || parsed?.title || '').trim(),
    artist: String(fallback.artist || parsed?.artist || parsed?.composer || '').trim(),
  };
}

function publicSupportContact(policies) {
  return {
    email: String(policies?.supportEmail || ''),
    phone: String(policies?.supportPhone || ''),
  };
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

function marketplaceRanking(averageRating = 0, audienceCount = 0) {
  const rating = Math.min(5, Math.max(0, Number(averageRating) || 0));
  const audience = Math.max(0, Number(audienceCount) || 0);
  const ratingPoints = Number(((rating / 5) * 10).toFixed(1));
  const audiencePoints = Number(Math.min(40, audience).toFixed(1));
  return {
    ratingPoints,
    audiencePoints,
    totalPoints: Number((ratingPoints + audiencePoints).toFixed(1)),
    maximumPoints: 50,
  };
}

function listingMode(listing) {
  const mode = String(listing?.listingMode || '').trim().toLowerCase();
  if (['sale', 'free', 'listener-reward'].includes(mode)) return mode;
  return Number(listing?.priceMcoins || 0) > 0 ? 'sale' : 'free';
}

function listenerRewardStatus(listing, db) {
  const policies = sitePolicies(db);
  const rewardMcoins = Math.max(0, mcoinAmount(listing.listenerRewardMcoins) || 0);
  const paidMcoins = Math.max(0, mcoinAmount(listing.rewardPaidMcoins) || 0);
  const capMcoins = policies.maximumRewardOutflowPerListingMcoins;
  const remainingCapMcoins = capMcoins > 0
    ? Math.max(0, Number((capMcoins - paidMcoins).toFixed(2)))
    : null;
  const seller = db.users.find((user) => user.id === listing.sellerId);
  const sellerCanFund = Boolean(seller && (hasUnlimitedMcoins(seller) || Number(seller.mcoins || 0) >= rewardMcoins));
  const withinCap = remainingCapMcoins === null || remainingCapMcoins >= rewardMcoins;
  return {
    rewardMcoins,
    paidMcoins,
    remainingCapMcoins,
    available: listingMode(listing) === 'listener-reward'
      && policies.listenerRewardsEnabled
      && rewardMcoins > 0
      && withinCap
      && sellerCanFund,
  };
}

function publicListing(listing, db, viewerId = null) {
  const seller = db.users.find((user) => user.id === listing.sellerId);
  const purchased = Boolean(viewerId && db.purchases.some(
    (purchase) => purchase.listingId === listing.id && purchase.buyerId === viewerId,
  ));
  const reviewSummary = listingReviewSummary(db, listing.id);
  const composer = publicComposer(seller, db, viewerId);
  const mode = listingMode(listing);
  const reward = listenerRewardStatus(listing, db);
  return {
    ...listing,
    listingMode: mode,
    priceMcoins: mode === 'sale' ? Number(listing.priceMcoins || 0) : 0,
    listenerRewardMcoins: mode === 'listener-reward' ? reward.rewardMcoins : 0,
    rewardPaidMcoins: reward.paidMcoins,
    rewardRemainingMcoins: reward.remainingCapMcoins,
    rewardAvailable: reward.available,
    assetPath: undefined,
    seller: seller
      ? composer
      : {
          user_id: listing.sellerId,
          friend_id: '',
          name: 'Composer',
          avatarUrl: '',
          followerCount: 0,
          averageRating: 0,
          ratingCount: 0,
          buyerCount: 0,
          ranking: marketplaceRanking(),
        },
    reviewSummary,
    purchased,
    owned: viewerId === listing.sellerId,
  };
}

function publicPersonalSong(song) {
  return {
    id: song.id,
    title: song.title,
    artist: song.artist || '',
    instrument: song.instrument,
    format: song.format,
    filename: song.filename,
    size: Number(song.size || 0),
    createdAt: song.createdAt,
  };
}

function attachGeneratedPersonalSong(db, job, {
  title,
  artist = '',
  instrument,
  filename,
  assetPath,
  bytes,
  sourceJobType,
}) {
  const existingForJob = db.personalSongs.find((song) => (
    song.userId === job.userId && song.sourceJobId === job.id
  ));
  if (existingForJob) {
    job.personalSongId = existingForJob.id;
    return existingForJob;
  }

  const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
  const duplicate = db.personalSongs.find((song) => (
    song.userId === job.userId && song.sha256 === sha256
  ));
  if (duplicate) {
    job.personalSongId = duplicate.id;
    return duplicate;
  }

  const personalSong = {
    id: id('song'),
    userId: job.userId,
    title: String(title || path.basename(filename, path.extname(filename)) || 'Untitled song').slice(0, 160),
    artist: String(artist || '').slice(0, 120),
    instrument: String(instrument || 'piano').slice(0, 60),
    format: 'JSON',
    filename: sanitizeFilename(filename || 'ready-to-play-song.json'),
    assetPath,
    size: bytes.length,
    sha256,
    sourceJobId: job.id,
    sourceJobType,
    createdAt: new Date().toISOString(),
  };
  db.personalSongs.push(personalSong);
  job.personalSongId = personalSong.id;
  return personalSong;
}

function backfillGeneratedPersonalSongs(db, userId) {
  let changed = false;
  const sources = [
    ...(db.mediaTranscriptionJobs || []).map((job) => ({
      job,
      sourceJobType: 'media-transcription',
      title: job.title,
    })),
    ...(db.scoreTranslationJobs || []).map((job) => ({
      job,
      sourceJobType: 'score-translation',
      title: path.basename(job.filename || '', path.extname(job.filename || '')),
    })),
  ];
  for (const source of sources) {
    const { job } = source;
    if (job.userId !== userId || job.status !== 'completed' || !job.outputPath || job.personalSongHiddenAt) continue;
    const existing = db.personalSongs.find((song) => (
      song.userId === userId && song.sourceJobId === job.id
    ));
    if (existing) {
      if (job.personalSongId !== existing.id) {
        job.personalSongId = existing.id;
        changed = true;
      }
      continue;
    }
    const personalSong = {
      id: job.personalSongId || `song_${job.id}`,
      userId,
      title: String(source.title || 'Untitled song').slice(0, 160),
      artist: '',
      instrument: String(job.instrument || 'piano').slice(0, 60),
      format: 'JSON',
      filename: sanitizeFilename(job.outputFilename || 'ready-to-play-song.json'),
      assetPath: job.outputPath,
      size: 0,
      sha256: '',
      sourceJobId: job.id,
      sourceJobType: source.sourceJobType,
      createdAt: job.completedAt || job.startedAt || new Date().toISOString(),
    };
    db.personalSongs.push(personalSong);
    job.personalSongId = personalSong.id;
    changed = true;
  }
  return changed;
}

function listingReviewSummary(db, listingId) {
  const reviews = db.listingReviews.filter((review) => review.listingId === listingId);
  const total = reviews.reduce((sum, review) => sum + Number(review.rating || 0), 0);
  return {
    averageRating: reviews.length ? Number((total / reviews.length).toFixed(2)) : 0,
    reviewCount: reviews.length,
  };
}

function composerRatingSummary(db, composerId) {
  const listingIds = new Set(db.listings.filter((listing) => listing.sellerId === composerId).map((listing) => listing.id));
  const reviews = db.listingReviews.filter((review) => listingIds.has(review.listingId));
  const total = reviews.reduce((sum, review) => sum + Number(review.rating || 0), 0);
  return {
    averageRating: reviews.length ? Number((total / reviews.length).toFixed(2)) : 0,
    ratingCount: reviews.length,
  };
}

function composerBuyerCount(db, composerId) {
  const listingIds = new Set(
    db.listings.filter((listing) => listing.sellerId === composerId).map((listing) => listing.id),
  );
  return new Set(
    db.purchases
      .filter((purchase) => listingIds.has(purchase.listingId))
      .filter((purchase) => Number(purchase.grossMcoins ?? purchase.amountMcoins ?? purchase.amount ?? 0) > 0)
      .map((purchase) => purchase.buyerId),
  ).size;
}

function publicComposer(user, db, viewerId = null) {
  if (!user) return null;
  const rating = composerRatingSummary(db, user.id);
  const buyerCount = composerBuyerCount(db, user.id);
  return {
    user_id: user.id,
    friend_id: user.friendId || '',
    name: user.name || 'Composer',
    avatarUrl: user.avatarUrl || '',
    followerCount: db.composerFollows.filter((follow) => follow.composerId === user.id).length,
    averageRating: rating.averageRating,
    ratingCount: rating.ratingCount,
    buyerCount,
    ranking: marketplaceRanking(rating.averageRating, buyerCount),
    publishedCount: db.listings.filter((listing) => listing.sellerId === user.id).length,
    isFollowing: Boolean(viewerId && db.composerFollows.some(
      (follow) => follow.composerId === user.id && follow.followerId === viewerId,
    )),
    isSelf: viewerId === user.id,
  };
}

function publicListingReview(review, db, viewerId = null) {
  const author = db.users.find((user) => user.id === review.userId);
  return {
    id: review.id,
    listingId: review.listingId,
    rating: Number(review.rating),
    comment: review.comment,
    createdAt: review.createdAt,
    updatedAt: review.updatedAt || null,
    mine: viewerId === review.userId,
    verifiedPurchase: db.purchases.some(
      (purchase) => purchase.listingId === review.listingId && purchase.buyerId === review.userId,
    ),
    author: {
      user_id: review.userId,
      name: author?.name || 'Former customer',
      avatarUrl: author?.avatarUrl || '',
    },
  };
}

function bandMembership(db, bandId, userId) {
  return db.bandMemberships.find(
    (membership) => membership.bandId === bandId && membership.userId === userId,
  ) || null;
}

function bandBan(band, userId) {
  return (Array.isArray(band.bans) ? band.bans : []).find((ban) => ban.userId === userId) || null;
}

function safeBandMessage(message, db) {
  const author = db.users.find((user) => user.id === message.userId);
  return {
    id: message.id,
    bandId: message.bandId,
    text: message.text,
    createdAt: message.createdAt,
    author: {
      userId: message.userId,
      name: author?.name || 'Former band member',
      avatarUrl: author?.avatarUrl || '',
    },
  };
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
        avatarUrl: user?.avatarUrl || '',
        role: item.role,
        joinedAt: item.joinedAt,
      };
    });
  return {
    id: band.id,
    name: band.name,
    description: band.description || '',
    host: { userId: band.hostId, name: host?.name || 'Band host', avatarUrl: host?.avatarUrl || '' },
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
    bannedMembers: isHost ? (Array.isArray(band.bans) ? band.bans : []).map((ban) => {
      const user = db.users.find((candidate) => candidate.id === ban.userId);
      return {
        userId: ban.userId,
        name: user?.name || 'Unknown account',
        avatarUrl: user?.avatarUrl || '',
        bannedAt: ban.bannedAt,
      };
    }) : undefined,
    createdAt: band.createdAt,
  };
}

function validBandInstrument(value) {
  return Object.prototype.hasOwnProperty.call(INSTRUMENTS, value);
}

const mediaUpload = multer({
  storage: multer.diskStorage({
    destination(req, file, callback) {
      ensureStorage();
      callback(null, UPLOAD_DIR);
    },
    filename(req, file, callback) {
      const extension = path.extname(file.originalname || '').toLowerCase();
      callback(null, `media-${Date.now()}-${crypto.randomBytes(8).toString('hex')}${extension}`);
    },
  }),
  // The multipart piano transcription form sends instrument, title,
  // playbackMode, rightsConfirmed, and paymentMethod alongside one media file.
  limits: { files: 1, fields: 5 },
  fileFilter(req, file, callback) {
    const extension = path.extname(file.originalname || '').toLowerCase();
    callback(null, MEDIA_EXTENSIONS.has(extension));
  },
});

const virtualTeacherUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    files: 2,
    fields: 8,
    fileSize: VIRTUAL_TEACHER_MODEL_MAX_BYTES,
  },
  fileFilter(req, file, callback) {
    const mimetype = String(file.mimetype || '').toLowerCase();
    const extension = path.extname(String(file.originalname || '')).toLowerCase();
    const isImage = file.fieldname === 'image'
      && ['image/png', 'image/jpeg', 'image/webp'].includes(mimetype);
    const isModel = file.fieldname === 'model'
      && extension === '.glb'
      && ['model/gltf-binary', 'application/octet-stream'].includes(mimetype);
    if (isImage || isModel) {
      callback(null, true);
      return;
    }
    const error = new Error(file.fieldname === 'model'
      ? 'The optional 3D model must be a binary glTF 2.0 (.glb) file.'
      : 'Upload a PNG, JPEG, or WebP character image.');
    error.status = 400;
    callback(error);
  },
});

function virtualTeacherImageContentType(bytes) {
  const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes || []);
  if (buffer.length >= 12
      && buffer.subarray(0, 4).equals(Buffer.from('RIFF'))
      && buffer.subarray(8, 12).equals(Buffer.from('WEBP'))) return 'image/webp';
  if (buffer.length >= 8
      && buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return 'image/png';
  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return 'image/jpeg';
  return '';
}

function inspectVirtualTeacherGlb(bytes) {
  const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes || []);
  if (buffer.length < 20 || buffer.subarray(0, 4).toString('ascii') !== 'glTF') {
    throw Object.assign(new Error('The uploaded model is not a valid binary glTF file.'), { status: 400 });
  }
  const version = buffer.readUInt32LE(4);
  const declaredLength = buffer.readUInt32LE(8);
  if (version !== 2 || declaredLength !== buffer.length) {
    throw Object.assign(new Error('The character model must use binary glTF 2.0.'), { status: 400 });
  }
  const jsonLength = buffer.readUInt32LE(12);
  const jsonType = buffer.readUInt32LE(16);
  if (jsonType !== 0x4e4f534a || jsonLength < 2 || 20 + jsonLength > buffer.length) {
    throw Object.assign(new Error('The GLB is missing its required JSON scene data.'), { status: 400 });
  }
  let document;
  try {
    document = JSON.parse(buffer.subarray(20, 20 + jsonLength).toString('utf8').trim());
  } catch {
    throw Object.assign(new Error('The GLB contains invalid scene data.'), { status: 400 });
  }
  if (String(document?.asset?.version || '') !== '2.0') {
    throw Object.assign(new Error('The character model must declare glTF 2.0.'), { status: 400 });
  }
  const nodes = Array.isArray(document.nodes) ? document.nodes : [];
  const skins = Array.isArray(document.skins) ? document.skins : [];
  const meshes = Array.isArray(document.meshes) ? document.meshes : [];
  if (nodes.length > 2500 || skins.length > 50 || meshes.length > 200 || (document.accessors || []).length > 5000) {
    throw Object.assign(new Error('The GLB rig is too complex for safe browser playback.'), { status: 400 });
  }
  const externalResources = [
    ...(Array.isArray(document.buffers) ? document.buffers : []),
    ...(Array.isArray(document.images) ? document.images : []),
  ].some((resource) => typeof resource?.uri === 'string' && resource.uri.trim());
  if (externalResources) {
    throw Object.assign(new Error('The GLB must contain all geometry and textures internally; external links are not allowed.'), { status: 400 });
  }
  const jointIndexes = new Set(skins.flatMap((skin) => Array.isArray(skin.joints) ? skin.joints : []));
  if (!skins.length || !meshes.length || jointIndexes.size < 6) {
    throw Object.assign(new Error('The GLB needs a skinned human rig with at least 6 joints and a mesh.'), { status: 400 });
  }
  const recognisedPatterns = [
    /hip|pelvis/i, /spine|chest/i, /head/i, /(left|[_ .-]l).*arm|arm.*(left|[_ .-]l)/i,
    /(right|[_ .-]r).*arm|arm.*(right|[_ .-]r)/i, /(left|right).*leg|leg.*(left|right)/i,
  ];
  const names = [...jointIndexes].map((index) => String(nodes[index]?.name || '')).filter(Boolean);
  const recognised = recognisedPatterns.filter((pattern) => names.some((name) => pattern.test(name))).length;
  if (recognised < 4) {
    throw Object.assign(new Error('Use a human GLB rig with named hips, spine, head, arms, and legs.'), { status: 400 });
  }
  return { version, jointCount: jointIndexes.size, nodeCount: nodes.length, meshCount: meshes.length };
}

function normalizeVirtualTeacherMinimumAge(value, legacyAdultConfirmation = false) {
  if (value === null || value === undefined || String(value).trim() === '') {
    return legacyAdultConfirmation ? 18 : 0;
  }
  return clampInteger(value, 0, 99, legacyAdultConfirmation ? 18 : 0);
}

function normalizeVirtualTeacherPrice(value) {
  if (value === null || value === undefined || String(value).trim() === '') return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return Number(Math.min(1000000000, Math.max(0, number)).toFixed(2));
}

function virtualTeacherInput(body = {}, fallback = {}) {
  const has = (key) => Object.prototype.hasOwnProperty.call(body, key);
  const text = (key, maximum) => String(has(key) ? body[key] : fallback[key] || '').trim().slice(0, maximum);
  const name = text('name', 50);
  const title = text('title', 80);
  const description = text('description', 240);
  const voice = text('voice', 50);
  const requestedVoiceType = String(has('voiceType') ? body.voiceType : fallback.voiceType || '').trim().toLowerCase();
  const voiceType = ['feminine', 'masculine'].includes(requestedVoiceType) ? requestedVoiceType : 'neutral';
  const armTone = String(has('armTone') ? body.armTone : fallback.armTone || '').trim().toLowerCase() === 'dark'
    ? 'dark'
    : 'light';
  const legacyAdult = has('requiresAdultConfirmation')
    ? String(body.requiresAdultConfirmation).trim().toLowerCase() === 'true'
    : Boolean(fallback.requiresAdultConfirmation);
  const adultCompanionEnabled = has('adultCompanionEnabled')
    ? String(body.adultCompanionEnabled).trim().toLowerCase() === 'true'
    : Boolean(fallback.adultCompanionEnabled ?? legacyAdult);
  const ageValue = has('minimumAge') ? body.minimumAge : fallback.minimumAge;
  if (ageValue !== null && ageValue !== undefined && String(ageValue).trim() !== '') {
    const parsedAge = Number(ageValue);
    if (!Number.isInteger(parsedAge) || parsedAge < 0 || parsedAge > 99) {
      throw Object.assign(new Error('Character minimum age must be a whole number from 0 to 99.'), { status: 400 });
    }
  }
  const minimumAge = Math.max(
    adultCompanionEnabled ? 18 : 0,
    normalizeVirtualTeacherMinimumAge(ageValue, legacyAdult),
  );
  const priceValue = has('pricePer30MinutesMcoins')
    ? body.pricePer30MinutesMcoins
    : fallback.pricePer30MinutesMcoins;
  if (priceValue !== null && priceValue !== undefined && String(priceValue).trim() !== '') {
    const parsedPrice = Number(priceValue);
    if (!Number.isFinite(parsedPrice) || parsedPrice < 0 || parsedPrice > 1000000000) {
      throw Object.assign(new Error('Character price must be between 0 and 1,000,000,000 Mcoins per 30 minutes.'), { status: 400 });
    }
  }
  if (name.length < 2) throw Object.assign(new Error('Character name must contain at least 2 characters.'), { status: 400 });
  if (title.length < 2) throw Object.assign(new Error('Add a short role or title.'), { status: 400 });
  if (description.length < 5) throw Object.assign(new Error('Add a short character description.'), { status: 400 });
  if (voice.length < 2) throw Object.assign(new Error('Describe the character voice or teaching style.'), { status: 400 });
  return {
    name,
    title,
    description,
    voice,
    voiceType,
    armTone,
    minimumAge,
    requiresAdultConfirmation: minimumAge >= 18,
    adultCompanionEnabled,
    pricePer30MinutesMcoins: normalizeVirtualTeacherPrice(priceValue),
    active: has('active') ? String(body.active).trim().toLowerCase() !== 'false' : fallback.active !== false,
  };
}

function mergedVirtualTeacher(character, fallback = null) {
  const source = character || {};
  const base = fallback || {};
  const adultCompanionEnabled = Boolean(
    source.adultCompanionEnabled
      ?? base.adultCompanionEnabled
      ?? source.requiresAdultConfirmation
      ?? base.requiresAdultConfirmation,
  );
  const minimumAge = Math.max(
    adultCompanionEnabled ? 18 : 0,
    normalizeVirtualTeacherMinimumAge(
      source.minimumAge ?? base.minimumAge,
      source.requiresAdultConfirmation ?? base.requiresAdultConfirmation,
    ),
  );
  return {
    ...base,
    ...source,
    id: String(source.id || base.id || '').trim(),
    name: String(source.name || base.name || 'Virtual teacher').trim().slice(0, 80),
    title: String(source.title || base.title || 'Polymath music teacher').trim().slice(0, 120),
    description: String(source.description || source.style || base.description || base.style || 'Clear, patient, and precise').trim().slice(0, 280),
    style: String(source.description || source.style || base.description || base.style || 'Clear, patient, and precise').trim().slice(0, 280),
    voice: String(source.voice || base.voice || 'Natural and expressive').trim().slice(0, 100),
    voiceType: ['feminine', 'masculine'].includes(String(source.voiceType || base.voiceType || '').toLowerCase())
      ? String(source.voiceType || base.voiceType).toLowerCase()
      : 'neutral',
    armTone: (source.armTone || base.armTone) === 'dark' ? 'dark' : 'light',
    minimumAge,
    requiresAdultConfirmation: minimumAge >= 18,
    adultCompanionEnabled,
    pricePer30MinutesMcoins: normalizeVirtualTeacherPrice(
      source.pricePer30MinutesMcoins ?? base.pricePer30MinutesMcoins,
    ),
    active: source.active !== false,
    builtIn: Boolean(source.builtIn || base.builtIn),
  };
}

function virtualTeacherCatalog(db, { includeInactive = false } = {}) {
  const records = Array.isArray(db?.virtualTeacherCharacters) ? db.virtualTeacherCharacters : [];
  const overrideById = new Map(
    records
      .filter((character) => BUILT_IN_VIRTUAL_TEACHER_IDS.has(String(character.id || '')))
      .map((character) => [character.id, character]),
  );
  const builtIns = Object.values(BUILT_IN_VIRTUAL_TEACHERS).map((teacher) => mergedVirtualTeacher(
    overrideById.get(teacher.id),
    { ...teacher, description: teacher.style, builtIn: true, active: true },
  ));
  const custom = records
    .filter((character) => !BUILT_IN_VIRTUAL_TEACHER_IDS.has(String(character.id || '')))
    .map((character) => mergedVirtualTeacher(character));
  return [...builtIns, ...custom]
    .filter((character) => includeInactive || character.active)
    .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { sensitivity: 'base' }));
}

function publicVirtualTeacherCharacter(character, globalPricePer30MinutesMcoins) {
  const pricePer30MinutesMcoins = normalizeVirtualTeacherPrice(character.pricePer30MinutesMcoins);
  const effectivePricePer30MinutesMcoins = pricePer30MinutesMcoins === null
    ? clampDecimal(
      globalPricePer30MinutesMcoins,
      0,
      1000000000,
      DEFAULT_LESSON_PRICE_PER_30_MINUTES_MCOINS,
    )
    : pricePer30MinutesMcoins;
  return {
    id: character.id,
    name: character.name,
    title: character.title,
    description: character.description || character.style,
    voice: character.voice,
    voiceType: ['feminine', 'masculine'].includes(character.voiceType) ? character.voiceType : 'neutral',
    armTone: character.armTone === 'dark' ? 'dark' : 'light',
    minimumAge: normalizeVirtualTeacherMinimumAge(character.minimumAge, character.requiresAdultConfirmation),
    requiresAdultConfirmation: normalizeVirtualTeacherMinimumAge(character.minimumAge, character.requiresAdultConfirmation) >= 18,
    adultCompanionEnabled: Boolean(character.adultCompanionEnabled),
    pricePer30MinutesMcoins,
    effectivePricePer30MinutesMcoins,
    active: character.active !== false,
    builtIn: Boolean(character.builtIn),
    imagePath: character.imageKey
      ? `/api/virtual-teachers/${encodeURIComponent(character.id)}/image?v=${encodeURIComponent(character.updatedAt || character.createdAt || '')}`
      : '',
    modelPath: character.modelKey
      ? `/api/virtual-teachers/${encodeURIComponent(character.id)}/model?v=${encodeURIComponent(character.updatedAt || character.createdAt || '')}`
      : '',
    rig: character.rig || null,
    custom: !character.builtIn,
    createdAt: character.createdAt || null,
    updatedAt: character.updatedAt || character.createdAt || null,
  };
}

function resolvedVirtualLessonTeacher(db, candidate) {
  const requestedId = String(candidate?.id || '').trim().replace(/[^a-z0-9_-]/gi, '').slice(0, 64);
  const catalog = virtualTeacherCatalog(db);
  const teacher = requestedId
    ? catalog.find((character) => character.id === requestedId) || null
    : catalog.find((character) => character.id === 'aria') || catalog[0] || null;
  if (!teacher) return null;
  return {
    id: teacher.id,
    name: teacher.name,
    title: teacher.title,
    style: teacher.style || teacher.description,
    voice: teacher.voice,
    voiceType: teacher.voiceType,
    minimumAge: teacher.minimumAge,
    requiresAdultConfirmation: teacher.requiresAdultConfirmation,
    adultCompanionEnabled: teacher.adultCompanionEnabled,
    pricePer30MinutesMcoins: teacher.pricePer30MinutesMcoins,
  };
}

function selectMuscriptorExecution({ serverlessConfigured, remoteUrl }) {
  if (serverlessConfigured) return 'runpod-serverless';
  if (String(remoteUrl || '').trim()) return 'remote-gpu';
  return 'local';
}

function muscriptorAvailability() {
  const serverlessConfigured = RUNPOD_SERVERLESS.configured;
  const remoteConfigured = Boolean(MUSCRIPTOR_REMOTE_URL);
  const execution = selectMuscriptorExecution({
    serverlessConfigured,
    remoteUrl: MUSCRIPTOR_REMOTE_URL,
  });
  const pythonExists = path.isAbsolute(MUSCRIPTOR_PYTHON)
    ? fs.existsSync(MUSCRIPTOR_PYTHON)
    : true;
  const ffmpegExists = Boolean(FFMPEG_PATH && fs.existsSync(FFMPEG_PATH));
  const workerExists = fs.existsSync(MUSCRIPTOR_WORKER);
  let reason = '';
  if (!MUSCRIPTOR_ENABLED) {
    reason = 'Polymath transcription is disabled. Enable it only for use permitted by the model licence.';
  } else if (!serverlessConfigured && !remoteConfigured && !pythonExists) {
    reason = 'The Polymath transcription environment was not found.';
  } else if (!serverlessConfigured && !remoteConfigured && !workerExists) {
    reason = 'The Polymath transcription worker was not found.';
  } else if (!ffmpegExists) {
    reason = 'FFmpeg is unavailable for audio and video preparation.';
  }
  return {
    enabled: MUSCRIPTOR_ENABLED
      && ffmpegExists
      && (serverlessConfigured || remoteConfigured || (pythonExists && workerExists)),
    configured: MUSCRIPTOR_ENABLED,
    adminOnly: MUSCRIPTOR_ADMIN_ONLY,
    model: MUSCRIPTOR_MODEL,
    checkpoint: RUNPOD_SERVERLESS.inferenceVersion,
    execution,
    storageTargets: serverlessConfigured ? RUNPOD_SERVERLESS.storageTargetCount : 0,
    maxBytes: null,
    maxDurationSeconds: MAX_MEDIA_SECONDS,
    timeoutMinutes: Math.round(MUSCRIPTOR_TIMEOUT_MS / 60000),
    acceptedExtensions: [...MEDIA_EXTENSIONS],
    license: 'CC-BY-NC-4.0',
    commercialUseAllowed: false,
    reason,
  };
}

function publicMediaTranscriptionJob(job) {
  return {
    id: job.id,
    filename: job.filename,
    title: job.title,
    instrument: job.instrument,
    model: job.model,
    status: job.status,
    stage: job.stage,
    progress: Number(job.progress || 0),
    noteCount: Number(job.noteCount || 0),
    instrumentGroups: Array.isArray(job.instrumentGroups) ? job.instrumentGroups : [],
    playbackMode: job.playbackMode || 'instrumental',
    paymentMethod: job.paymentMethod || 'legacy',
    costMcoins: Number(job.costMcoins || 0),
    vocalMelodyNoteCount: Number(job.vocalMelodyNoteCount || 0),
    startedAt: job.startedAt,
    completedAt: job.completedAt || null,
    failedAt: job.failedAt || null,
    error: job.error || '',
    outputFilename: job.status === 'completed' ? job.outputFilename : undefined,
    personalSongId: job.status === 'completed' ? job.personalSongId : undefined,
  };
}

async function updateMediaTranscriptionJob(jobId, changes) {
  const db = await readDb();
  const job = db.mediaTranscriptionJobs.find((candidate) => candidate.id === jobId);
  if (!job) return null;
  Object.assign(job, changes);
  await writeDb(db);
  return job;
}

async function claimBackgroundJob(collection, jobId) {
  const db = await readDb();
  const job = db[collection]?.find((candidate) => candidate.id === jobId);
  if (!job || job.status !== 'processing') return null;
  const claimExpires = Date.parse(job.claimExpiresAt || '');
  const activeClaim = Number.isFinite(claimExpires) && claimExpires > Date.now();
  if (activeClaim && job.claimedBy !== PROCESS_INSTANCE_ID && JOB_QUEUE.enabled) return null;
  if (activeClaim && job.claimedBy === PROCESS_INSTANCE_ID) return null;
  job.claimedBy = PROCESS_INSTANCE_ID;
  job.claimExpiresAt = new Date(Date.now() + JOB_CLAIM_MS).toISOString();
  try {
    await writeDb(db);
    return job;
  } catch (error) {
    if (error instanceof StateConflictError) return null;
    throw error;
  }
}

let mediaProgressWriteQueue = Promise.resolve();

function queueMediaTranscriptionUpdate(jobId, changes) {
  mediaProgressWriteQueue = mediaProgressWriteQueue
    .then(() => updateMediaTranscriptionJob(jobId, changes))
    .catch((error) => console.error('Polymath progress update failed:', error));
}

function safeRemoveUpload(filePath) {
  if (!filePath) return;
  const resolved = path.resolve(filePath);
  const uploadRoot = `${path.resolve(UPLOAD_DIR)}${path.sep}`;
  if (!resolved.startsWith(uploadRoot)) return;
  try {
    fs.rmSync(resolved, { force: true });
  } catch {
    // Cleanup failure should not replace the useful transcription result/error.
  }
}

async function safeRemoveArtifact(key) {
  if (!key) return;
  try {
    await ARTIFACT_STORE.remove(key);
  } catch (error) {
    console.warn(`Artifact cleanup failed for ${key}: ${error.message}`);
  }
}

function runChild(command, args, { timeoutMs, timeoutMessage, onLine } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    let stderr = '';
    let stdoutBuffer = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      child.kill();
      settled = true;
      reject(new Error(timeoutMessage || 'The media-processing time limit was reached.'));
    }, timeoutMs || 60 * 60 * 1000);

    child.stdout.on('data', (chunk) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || '';
      lines.filter(Boolean).forEach((line) => onLine?.(line));
    });
    child.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-12000);
    });
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (stdoutBuffer.trim()) onLine?.(stdoutBuffer.trim());
      if (code === 0) resolve({ stderr });
      else reject(new Error(stderr.trim() || `Transcription process exited with code ${code}.`));
    });
  });
}

function muscriptorConstraints(instrument, playbackMode = 'instrumental') {
  if (instrument === 'band') return [];
  // Score-arranged instruments deliberately transcribe the complete song
  // before revoicing it. Full mode keeps the singer melody; instrumental mode
  // removes voice during post-processing. Constraining the model to guitar or
  // piano here would discard musical parts before the arranger can hear them.
  if (
    ['piano', 'guitar', 'electric-guitar'].includes(instrument)
    && ['full', 'instrumental'].includes(playbackMode)
  ) return [];
  return MUSCRIPTOR_INSTRUMENTS[instrument] || [];
}

async function runRemoteMuscriptor(job, preparedPath, outputPath, constraints) {
  const form = new FormData();
  form.append('file', new Blob([fs.readFileSync(preparedPath)], { type: 'audio/wav' }), `${job.id}.wav`);
  constraints.forEach((instrument) => form.append('instruments', instrument));
  form.append('detect_tempo', 'best-effort');

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MUSCRIPTOR_TIMEOUT_MS);
  const headers = { 'X-Client-Id': job.id };
  if (MUSCRIPTOR_REMOTE_TOKEN) headers.Authorization = `Bearer ${MUSCRIPTOR_REMOTE_TOKEN}`;

  let response;
  try {
    while (true) {
      response = await fetch(`${MUSCRIPTOR_REMOTE_URL}/transcribe`, {
        method: 'POST',
        headers,
        body: form,
        signal: controller.signal,
      });
      if (response.status !== 503) break;
      await response.text();
      await updateMediaTranscriptionJob(job.id, {
        stage: 'Waiting for the RunPod GPU to finish the previous transcription',
        progress: 20,
      });
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  } catch (error) {
    clearTimeout(timer);
    if (error.name === 'AbortError') {
      throw new Error(`Music transcription took longer than the ${Math.round(MUSCRIPTOR_TIMEOUT_MS / 60000)}-minute processing limit.`);
    }
    throw new Error(`Could not reach the RunPod Polymath worker: ${error.message}`);
  }

  if (!response.ok || !response.body) {
    clearTimeout(timer);
    const details = (await response.text().catch(() => '')).trim().slice(0, 1000);
    throw new Error(`RunPod Polymath returned HTTP ${response.status}${details ? `: ${details}` : ''}`);
  }

  const collector = createMuscriptorEventCollector({
    model: MUSCRIPTOR_MODEL,
    source: 'runpod',
    onProgress(progress) {
      const ratio = Math.max(0, Math.min(1, progress.completed / progress.total));
      queueMediaTranscriptionUpdate(job.id, {
        stage: `Transcribing ${progress.completed} of ${progress.total} audio sections on RunPod GPU`,
        progress: 22 + Math.round(ratio * 70),
      });
    },
  });
  let buffer = '';
  const decoder = new TextDecoder();

  function consumeEvents(final = false) {
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = final ? '' : blocks.pop() || '';
    blocks.forEach((block) => {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('\n');
      if (!data) return;
      try {
        collector.accept(JSON.parse(data));
      } catch {
        // Ignore malformed diagnostics while preserving valid streamed events.
      }
    });
  }

  try {
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      consumeEvents();
    }
    buffer += decoder.decode();
    buffer += '\n\n';
    consumeEvents(true);
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(`Music transcription took longer than the ${Math.round(MUSCRIPTOR_TIMEOUT_MS / 60000)}-minute processing limit.`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }

  const { notes, progress, beatGrid, diagnostics } = collector.finish();
  if (!notes.length) throw new Error('Polymath could not detect playable notes in this recording.');

  const payload = {
    title: job.title || 'Uploaded recording',
    composer: 'Polymath transcription',
    instrument: job.instrument,
    bpm: beatGrid?.bpm || 120,
    beatsPerBar: beatGrid?.beatsPerBar || 4,
    beatGrid,
    notes,
    instrumentGroups: [...new Set(notes.map((note) => note.instrument))].sort(),
    sourceType: 'muscriptor-audio-transcription',
    readyToPlayFormat: 'polymath-musician-json-v1',
    transcriptionProvider: `Polymath ${MUSCRIPTOR_MODEL[0].toUpperCase()}${MUSCRIPTOR_MODEL.slice(1)} on RunPod GPU`,
    modelLicense: 'CC-BY-NC-4.0',
    progress,
    transcriptionDiagnostics: diagnostics,
  };
  fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2), 'utf8');
  return payload;
}

async function runServerlessMuscriptor(job, preparedPath, outputPath, constraints) {
  const raw = await RUNPOD_SERVERLESS.transcribe({
    job,
    preparedPath,
    constraints,
    onProgress(remote) {
      const state = String(remote.state || '').trim().toUpperCase();
      if (state === 'IN_QUEUE') {
        queueMediaTranscriptionUpdate(job.id, {
          stage: 'Waiting for a RunPod Serverless GPU worker',
          progress: 22,
        });
      } else if (state === 'IN_PROGRESS') {
        queueMediaTranscriptionUpdate(job.id, {
          stage: String(remote.progress || 'Polymath is detecting notes and instruments'),
          progress: 55,
        });
      }
    },
  });
  const payload = {
    ...raw,
    title: raw.title || job.title || 'Uploaded recording',
    composer: raw.composer || 'Polymath transcription',
    instrument: raw.instrument || job.instrument || 'band',
    bpm: Number(raw.bpm) || 120,
    notes: Array.isArray(raw.notes) ? raw.notes : [],
    instrumentGroups: Array.isArray(raw.instrumentGroups)
      ? raw.instrumentGroups
      : [...new Set((raw.notes || []).map((note) => note.instrument).filter(Boolean))].sort(),
    sourceType: raw.sourceType || 'muscriptor-audio-transcription',
    readyToPlayFormat: raw.readyToPlayFormat || 'polymath-musician-json-v1',
    transcriptionProvider: raw.transcriptionProvider || `Polymath ${MUSCRIPTOR_MODEL} on RunPod Serverless`,
    modelLicense: raw.modelLicense || 'CC-BY-NC-4.0',
  };
  if (!payload.notes.length) {
    throw new Error('RunPod Serverless completed without playable Polymath notes.');
  }
  fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2), 'utf8');
  return payload;
}

async function processMediaTranscriptionJob(jobId) {
  let job = await claimBackgroundJob('mediaTranscriptionJobs', jobId);
  if (!job) return;
  let db;

  const sourceWorkPath = path.join(UPLOAD_DIR, `${job.id}-source${path.extname(job.filename || '')}`);
  let sourcePath = '';
  const preparedPath = path.join(UPLOAD_DIR, `${job.id}-prepared.wav`);
  const outputPath = path.join(UPLOAD_DIR, `${job.id}-ready-to-play.json`);
  const arrangedPath = path.join(UPLOAD_DIR, `${job.id}-piano-arranged.json`);

  try {
    sourcePath = await ARTIFACT_STORE.materialize(job.sourcePath, sourceWorkPath);
    await updateMediaTranscriptionJob(jobId, {
      stage: 'Preparing audio from your upload',
      progress: 12,
    });
    await runChild(FFMPEG_PATH, [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-i', sourcePath,
      '-vn', '-ac', '1', '-ar', '16000',
      '-t', String(MAX_MEDIA_SECONDS),
      preparedPath,
    ], { timeoutMs: 10 * 60 * 1000 });

    const execution = selectMuscriptorExecution({
      serverlessConfigured: RUNPOD_SERVERLESS.configured,
      remoteUrl: MUSCRIPTOR_REMOTE_URL,
    });
    await updateMediaTranscriptionJob(jobId, {
      stage: execution === 'runpod-serverless'
        ? `Submitting Polymath ${MUSCRIPTOR_MODEL[0].toUpperCase()}${MUSCRIPTOR_MODEL.slice(1)} to RunPod Serverless`
        : execution === 'remote-gpu'
          ? `Connecting to Polymath ${MUSCRIPTOR_MODEL[0].toUpperCase()}${MUSCRIPTOR_MODEL.slice(1)} on RunPod GPU`
          : `Loading Polymath ${MUSCRIPTOR_MODEL[0].toUpperCase()}${MUSCRIPTOR_MODEL.slice(1)}`,
      progress: 20,
    });
    const constraints = muscriptorConstraints(job.instrument, job.playbackMode);
    if (execution === 'runpod-serverless') {
      await runServerlessMuscriptor(job, preparedPath, outputPath, constraints);
    } else if (execution === 'remote-gpu') {
      await runRemoteMuscriptor(job, preparedPath, outputPath, constraints);
    } else {
      const args = [
        MUSCRIPTOR_WORKER,
        '--input', preparedPath,
        '--output', outputPath,
        '--title', job.title,
        '--instrument', job.instrument,
        '--model', MUSCRIPTOR_MODEL,
      ];
      if (constraints.length) args.push('--instruments', constraints.join(','));

      await runChild(MUSCRIPTOR_PYTHON, args, {
        timeoutMs: MUSCRIPTOR_TIMEOUT_MS,
        timeoutMessage: `Music transcription took longer than the ${Math.round(MUSCRIPTOR_TIMEOUT_MS / 60000)}-minute processing limit.`,
        onLine(line) {
          try {
            const message = JSON.parse(line);
            if (message.type === 'stage') {
              queueMediaTranscriptionUpdate(jobId, { stage: message.stage });
            } else if (message.type === 'progress' && Number(message.total) > 0) {
              const ratio = Math.max(0, Math.min(1, Number(message.completed) / Number(message.total)));
              queueMediaTranscriptionUpdate(jobId, {
                stage: `Transcribing ${message.completed} of ${message.total} audio sections`,
                progress: 22 + Math.round(ratio * 70),
              });
            }
          } catch {
            // MuScriptor diagnostic output is intentionally ignored here.
          }
        },
      });
    }

    await updateMediaTranscriptionJob(jobId, {
      stage: 'Cleaning rapid repeats and shaping the piano arrangement',
      progress: 94,
    });
    const rawResult = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    let result = postProcessMuscriptorResult(rawResult, {
      instrument: job.instrument,
      playbackMode: job.playbackMode,
      preparedPath,
    });
    result.playbackMode = job.playbackMode || 'instrumental';
    result.vocalMelodyIncluded = job.playbackMode === 'full';
    result.selectedInstrument = job.instrument;
    result.arrangementProfile = job.instrument === 'piano'
      ? 'piano-reduction-with-midi-phrasing-v3'
      : ['guitar', 'electric-guitar'].includes(job.instrument)
        ? 'selected-guitar-midi-phrasing-v1'
        : 'detected-instrument-performance-v1';
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');
    if (job.instrument === 'piano') {
      await updateMediaTranscriptionJob(jobId, {
        stage: 'Building a playable 88-key acoustic-piano arrangement',
        progress: 97,
      });
      await runChild(PIANO_ARRANGER_PYTHON, [
        PIANO_ARRANGER_WORKER,
        '--input', outputPath,
        '--output', arrangedPath,
        '--mode', job.playbackMode || 'instrumental',
      ], {
        timeoutMs: 2 * 60 * 1000,
        timeoutMessage: 'The final piano arrangement took longer than two minutes.',
      });
      result = JSON.parse(fs.readFileSync(arrangedPath, 'utf8'));
      fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');
      safeRemoveUpload(arrangedPath);
    }
    db = await readDb();
    job = db.mediaTranscriptionJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.outputPath = await ARTIFACT_STORE.putFile(
      artifactKey('transcriptions', path.basename(outputPath)),
      outputPath,
      'application/json',
    );
    job.outputFilename = `${sanitizeFilename(job.title || 'polymath-transcription')}.json`;
    job.vocalMelodyNoteCount = Number(result.transcriptionCleanup?.vocalMelodyNotes || 0);
    job.noteCount = Array.isArray(result.notes) ? result.notes.length : 0;
    job.instrumentGroups = Array.isArray(result.instrumentGroups) ? result.instrumentGroups : [];
    job.status = 'completed';
    job.stage = 'Ready to play';
    job.progress = 100;
    job.completedAt = new Date().toISOString();
    attachGeneratedPersonalSong(db, job, {
      title: result.title || job.title,
      artist: result.artist || result.composer || '',
      instrument: job.instrument,
      filename: job.outputFilename,
      assetPath: job.outputPath,
      bytes: fs.readFileSync(outputPath),
      sourceJobType: 'media-transcription',
    });
    await writeDb(db);
    safeRemoveUpload(sourcePath);
    safeRemoveUpload(preparedPath);
    if (ARTIFACT_STORE.remote) safeRemoveUpload(outputPath);
  } catch (error) {
    db = await readDb();
    job = db.mediaTranscriptionJobs.find((candidate) => candidate.id === jobId);
    if (job && job.status === 'processing') {
      refundTranslationJob(db, job, error.message || 'Polymath could not transcribe this recording.');
      await writeDb(db);
    }
    safeRemoveUpload(sourcePath);
    safeRemoveUpload(preparedPath);
    safeRemoveUpload(outputPath);
    safeRemoveUpload(arrangedPath);
  } finally {
    await safeRemoveArtifact(job?.sourcePath);
  }
}

let mediaTranscriptionQueue = Promise.resolve();

function enqueueMediaTranscription(jobId) {
  mediaTranscriptionQueue = mediaTranscriptionQueue
    .then(() => processMediaTranscriptionJob(jobId))
    .catch((error) => console.error('Polymath queue error:', error));
  return mediaTranscriptionQueue;
}

async function dispatchBackgroundJob(type, jobId) {
  if (JOB_QUEUE.enabled) {
    await JOB_QUEUE.enqueue({ type, jobId });
    return;
  }
  setImmediate(() => {
    const task = type === 'media-transcription'
      ? enqueueMediaTranscription(jobId)
      : processTranslationJob(jobId);
    Promise.resolve(task).catch((error) => console.error('Background job failed:', error));
  });
}

async function runQueuedJob(message) {
  if (message?.type === 'media-transcription') return enqueueMediaTranscription(message.jobId);
  if (message?.type === 'score-translation') return processTranslationJob(message.jobId);
  throw new Error('Unknown background job type.');
}

const ASSISTANT_REQUEST_TIMES = new Map();
const COMMUNITY_REQUEST_TIMES = new Map();
const SUPPORT_REQUEST_INTERVAL_MS = Number.isFinite(Number(process.env.SUPPORT_REQUEST_INTERVAL_MS))
  ? Math.max(0, Number(process.env.SUPPORT_REQUEST_INTERVAL_MS))
  : 1400;

function requestIntervalAllowed(store, key, minimumIntervalMs) {
  const now = Date.now();
  const previous = Number(store.get(key) || 0);
  if (now - previous < minimumIntervalMs) return false;
  store.set(key, now);
  if (store.size > 10000) {
    for (const [candidate, timestamp] of store.entries()) {
      if (now - timestamp > 24 * 60 * 60 * 1000) store.delete(candidate);
    }
    while (store.size > 10000) {
      const oldestKey = store.keys().next().value;
      if (oldestKey === undefined) break;
      store.delete(oldestKey);
    }
  }
  return true;
}

app.use(cors({
  origin(origin, callback) {
    const normalizedOrigin = String(origin || '').replace(/\/+$/, '');
    if (!origin || CLIENT_ORIGINS.has(normalizedOrigin) || /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin)) {
      callback(null, true);
      return;
    }
    callback(new Error('Origin is not allowed by Polymath Musician CORS policy.'));
  },
  credentials: false,
}));
app.use(express.json({ limit: '18mb' }));
app.use('/api/model-lab', requireAuth, requireAdmin, MODEL_LAB.router);

app.get('/', async (req, res, next) => {
  if (IS_PRODUCTION) return next();
  return res.send('Polymath Musician backend is running');
});
app.get('/api/health', async (req, res) => res.json({
  ok: true,
  storage: STATE_STORE.provider,
  artifacts: ARTIFACT_STORE.provider,
  queue: JOB_QUEUE.enabled ? 'sqs' : 'in-process',
  virtualLessons: POLYMATH_ASSISTANT.capabilities().available ? 'configured' : 'unconfigured',
  region: process.env.APP_REGION || 'local',
}));
app.get('/api/health/state', async (req, res) => {
  try {
    await readDb();
    return res.json({
      ok: true,
      storage: STATE_STORE.provider,
      state: 'ready',
      region: process.env.APP_REGION || 'local',
    });
  } catch (error) {
    console.error('State dependency health check failed:', error);
    const rawCode = String(error?.code || '').trim().toUpperCase();
    const code = /^[A-Z0-9_]{2,64}$/.test(rawCode)
      ? rawCode
      : 'STATE_READ_FAILED';
    return res.status(503).json({
      ok: false,
      storage: STATE_STORE.provider,
      state: 'unavailable',
      code,
      region: process.env.APP_REGION || 'local',
    });
  }
});
app.get('/api/test', async (req, res) => res.json({
  message: 'Backend is working',
  environment: PAYPAL_ENV,
  scoreTranslation: localOmrAvailability(),
}));

app.get('/api/assistant/capabilities', requireAuth, async (req, res) => {
  const policies = sitePolicies(req.db);
  res.json({
    ...POLYMATH_ASSISTANT.capabilities(),
    support: {
      ...supportQuestionAllowance(req.user, { unlimited: isAdministrator(req.user) }),
      contact: publicSupportContact(policies),
    },
  });
});

app.post('/api/assistant/support', requireAuth, async (req, res) => {
  const policies = sitePolicies(req.db);
  const supportContact = publicSupportContact(policies);
  const unlimited = isAdministrator(req.user);
  const currentAllowance = supportQuestionAllowance(req.user, { unlimited });
  if (!currentAllowance.unlimited && currentAllowance.remainingQuestions <= 0) {
    return res.status(429).json({
      error: 'You have used today\'s 7 Help questions. Contact the Polymath helpline or return after the daily reset.',
      code: 'SUPPORT_DAILY_LIMIT_REACHED',
      support: { ...currentAllowance, contact: supportContact },
    });
  }
  if (!requestIntervalAllowed(ASSISTANT_REQUEST_TIMES, `${req.user.id}:support`, SUPPORT_REQUEST_INTERVAL_MS)) {
    res.set('Retry-After', '2');
    return res.status(429).json({
      error: 'Give Polymath Support a moment to finish the previous reply.',
      code: 'SUPPORT_REPLY_IN_PROGRESS',
      support: { ...currentAllowance, contact: supportContact },
    });
  }
  const reservation = reserveSupportQuestion(req.user, { unlimited });
  try {
    if (reservation.reserved) await writeDb(req.db);
    const safe = safeUser(req.user);
    const answer = await POLYMATH_ASSISTANT.supportChat({
      messages: req.body?.messages,
      accountContext: {
        tier: safe.subscriptionTier,
        admin: safe.admin,
        translationAllowance: safe.translationAllowance,
        readySheetAllowance: safe.readySheetAllowance,
      },
    });
    return res.json({
      ...answer,
      support: {
        ...supportQuestionAllowance(req.user, { unlimited }),
        contact: supportContact,
      },
    });
  } catch (error) {
    if (refundSupportQuestion(req.user, reservation)) {
      try {
        await writeDb(req.db);
      } catch (refundError) {
        console.error('Polymath support quota refund failed:', refundError);
      }
    }
    const unavailable = error?.code === 'ASSISTANT_UNAVAILABLE';
    const invalid = error?.code === 'INVALID_ASSISTANT_REQUEST';
    console.error('Polymath support failed:', error);
    return res.status(unavailable ? 503 : invalid ? 400 : error?.name === 'AbortError' ? 504 : 502).json({
      error: unavailable || invalid
        ? error.message
        : 'Polymath Support could not reply. Please try again.',
      support: {
        ...supportQuestionAllowance(req.user, { unlimited }),
        contact: supportContact,
      },
    });
  }
});

app.get('/api/virtual-lessons', requireAuth, async (req, res) => {
  const changed = expireVirtualLessons(req.db);
  if (changed) await writeDb(req.db);
  return res.json({
    catalog: lessonCatalog(sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins),
    assistantAvailable: POLYMATH_ASSISTANT.capabilities().available,
    session: publicVirtualLesson(activeVirtualLesson(req.db, req.user.id)),
    user: safeUser(req.user),
  });
});

app.post('/api/virtual-lessons', requireAuth, async (req, res) => {
  const clientRequestId = normalizeClientRequestId(req.body?.clientRequestId);
  if (!clientRequestId) {
    return res.status(400).json({ error: 'A valid lesson checkout reference is required.' });
  }
  const existing = req.db.virtualLessonSessions.find((session) => (
    session.userId === req.user.id && session.clientRequestId === clientRequestId
  ));
  if (existing) {
    return res.json({
      duplicate: true,
      session: publicVirtualLesson(existing),
      user: safeUser(req.user),
    });
  }
  const active = activeVirtualLesson(req.db, req.user.id);
  if (active) {
    return res.status(409).json({
      error: `Your active lesson is locked to ${active.teacher?.name || 'the selected teacher'} until it ends.`,
      code: 'VIRTUAL_LESSON_TEACHER_LOCKED',
      lockedTeacherId: active.teacher?.id || '',
      session: publicVirtualLesson(active),
    });
  }
  if (!POLYMATH_ASSISTANT.capabilities().available) {
    return res.status(503).json({
      error: 'Virtual lessons are not available on this server yet. Nothing was charged.',
      code: 'VIRTUAL_TEACHER_UNAVAILABLE',
    });
  }
  const administrator = isAdministrator(req.user);
  const conversationMode = normalizeConversationMode(req.body?.conversationMode);
  const teacher = resolvedVirtualLessonTeacher(req.db, req.body?.teacher);
  if (!teacher) {
    return res.status(409).json({
      error: 'That virtual teacher is no longer available. Choose another active teacher.',
      code: 'VIRTUAL_TEACHER_NOT_ACTIVE',
    });
  }
  const confirmedAge = Math.max(
    req.body?.adultConfirmed === true ? 18 : 0,
    clampInteger(req.body?.confirmedAge, 0, 99, 0),
  );
  if (!administrator && teacher.minimumAge > confirmedAge) {
    return res.status(403).json({
      error: `Confirm that you are at least ${teacher.minimumAge} before choosing ${teacher.name}.`,
      code: 'VIRTUAL_TEACHER_AGE_CONFIRMATION_REQUIRED',
      minimumAge: teacher.minimumAge,
    });
  }
  const globalLessonPrice = sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins;
  const quote = lessonQuote(
    req.body?.durationMinutes,
    teacher.pricePer30MinutesMcoins ?? globalLessonPrice,
  );
  if (!quote) {
    return res.status(400).json({ error: 'Enter a valid private-session duration.' });
  }
  const adultConfirmed = req.body?.adultConfirmed === true;
  const companionConsent = req.body?.companionConsent === true;
  if (conversationMode === 'adult-companion') {
    if (!teacher.adultCompanionEnabled) {
      return res.status(400).json({
        error: 'Choose an adult-eligible character for companion mode.',
        code: 'ADULT_COMPANION_TEACHER_REQUIRED',
      });
    }
    if (!adultConfirmed || !companionConsent) {
      return res.status(403).json({
        error: 'Confirm that you are 18+ and opt in before starting companion mode.',
        code: 'ADULT_COMPANION_CONFIRMATION_REQUIRED',
      });
    }
  }
  const priceMcoins = administrator ? 0 : quote.priceMcoins;
  if (!administrator && Number(req.user.mcoins || 0) < priceMcoins) {
    return res.status(402).json({
      error: `You need ${priceMcoins} Mcoins for this ${quote.durationMinutes}-minute lesson.`,
      requiredMcoins: priceMcoins,
      availableMcoins: Number(req.user.mcoins || 0),
    });
  }

  if (!administrator) {
    req.user.mcoins = Number((Number(req.user.mcoins || 0) - priceMcoins).toFixed(2));
  }
  const session = createVirtualLesson({
    id: id('virtual_lesson'),
    userId: req.user.id,
    clientRequestId,
    durationMinutes: quote.durationMinutes,
    priceMcoins,
    teacher,
    conversationMode,
    conversationPreferences: req.body?.conversationPreferences,
    adultConfirmed,
    companionConsent,
    studentName: req.user.name,
  });
  if (conversationMode === 'adult-companion') {
    req.user.adultCompanionConfirmedAt = session.adultConfirmedAt;
  }
  req.db.virtualLessonSessions.push(session);
  addLedger(
    req.db,
    req.user.id,
    -priceMcoins,
    administrator ? 'admin_virtual_lesson' : 'virtual_lesson',
    `${quote.durationMinutes}-minute private ${conversationMode === 'adult-companion' ? 'adult companion session' : 'virtual music lesson'} with ${session.teacher.name}`,
  );
  await writeDb(req.db);
  return res.status(201).json({
    session: publicVirtualLesson(session),
    user: safeUser(req.user),
    chargedMcoins: priceMcoins,
  });
});

app.post('/api/virtual-lessons/:sessionId/messages', requireAuth, async (req, res) => {
  if (!requestIntervalAllowed(ASSISTANT_REQUEST_TIMES, `${req.user.id}:teacher`, 1400)) {
    res.set('Retry-After', '2');
    return res.status(429).json({ error: 'Give your teacher a moment to finish the previous reply.' });
  }
  const session = req.db.virtualLessonSessions.find((candidate) => (
    candidate.id === req.params.sessionId && candidate.userId === req.user.id
  ));
  if (!session) return res.status(404).json({ error: 'Virtual lesson not found.' });
  if (!sessionIsActive(session)) {
    if (session.status === 'active') {
      expireVirtualLessons(req.db);
      await writeDb(req.db);
    }
    return res.status(410).json({
      error: 'This virtual lesson has ended. Choose a new duration to continue.',
      session: publicVirtualLesson(session),
    });
  }
  const userText = String(req.body?.message || '').trim().slice(0, 1600);
  if (!userText) return res.status(400).json({ error: 'Say or type a message first.' });
  const requestedAt = new Date();
  const lessonContext = req.body?.lessonContext;
  const observations = req.body?.observations;
  const action = parseTeacherDemonstration(
    userText,
    lessonContext,
    session.memory?.lastDemonstration,
  );
  appendSessionMessage(session, {
    id: id('lesson_message'),
    role: 'user',
    text: userText,
    createdAt: requestedAt.toISOString(),
  }, requestedAt);

  try {
    let result;
    if (action) {
      const start = Math.floor(action.startSeconds / 60) + ':' + String(Math.floor(action.startSeconds % 60)).padStart(2, '0');
      const end = Math.floor(action.endSeconds / 60) + ':' + String(Math.floor(action.endSeconds % 60)).padStart(2, '0');
      const hand = action.hand === 'both' ? 'both hands' : `the ${action.hand} hand`;
      const speed = action.speed ? ` at ${Math.round(action.speed * 100)}% speed` : '';
      result = {
        reply: `${req.user.name?.split(' ')[0] || 'Ready'}, watch ${hand} from ${start} to ${end}${speed}. Notice where each fingertip lands, then copy only this short phrase once.` ,
        provider: 'polymath-demonstration-engine',
        role: 'piano-teacher',
      };
    } else {
      result = await POLYMATH_ASSISTANT.teacherChat({
        messages: session.messages.map((message) => ({
          role: message.role === 'assistant' ? 'assistant' : 'user',
          content: message.text,
        })),
        teacher: session.teacher,
        conversationMode: session.conversationMode,
        conversationPreferences: session.conversationPreferences,
        accountContext: {
          studentName: req.user.name,
          sessionMemory: session.memory,
          sessionEndsAt: session.expiresAt,
        },
        lessonContext,
        observations,
      });
    }
    updateSessionMemory(session, {
      studentName: req.user.name,
      userMessage: userText,
      lessonContext,
      practiceReport: observations?.practiceReport,
      action,
    });
    const repliedAt = new Date();
    appendSessionMessage(session, {
      id: id('lesson_message'),
      role: 'assistant',
      text: result.reply,
      createdAt: repliedAt.toISOString(),
    }, repliedAt);
    await writeDb(req.db);
    return res.json({
      ...result,
      action,
      session: publicVirtualLesson(session, repliedAt),
    });
  } catch (error) {
    const unavailable = error?.code === 'ASSISTANT_UNAVAILABLE';
    const invalid = error?.code === 'INVALID_ASSISTANT_REQUEST';
    console.error('Polymath teacher chat failed:', error);
    // Paid time should not disappear while a GPU reply fails. Restore the
    // failed request time plus a small recovery allowance.
    const recoverySeconds = Math.min(120, Math.max(15, Math.ceil((Date.now() - requestedAt.getTime()) / 1000) + 10));
    session.expiresAt = new Date(new Date(session.expiresAt).getTime() + recoverySeconds * 1000).toISOString();
    session.messages.pop();
    session.aiFailureCount = Number(session.aiFailureCount || 0) + 1;
    session.lastFailureAt = new Date().toISOString();
    await writeDb(req.db);
    return res.status(unavailable ? 503 : invalid ? 400 : error?.name === 'AbortError' ? 504 : 502).json({
      error: unavailable || invalid
        ? error.message
        : 'Your virtual teacher could not reply. Please try again.',
      session: publicVirtualLesson(session),
      recoveredSeconds: recoverySeconds,
    });
  }
});

app.post('/api/virtual-lessons/:sessionId/end', requireAuth, async (req, res) => {
  const session = req.db.virtualLessonSessions.find((candidate) => (
    candidate.id === req.params.sessionId && candidate.userId === req.user.id
  ));
  if (!session) return res.status(404).json({ error: 'Virtual lesson not found.' });
  if (session.status === 'active') endVirtualLesson(session);
  await writeDb(req.db);
  return res.json({ session: publicVirtualLesson(session), user: safeUser(req.user) });
});

app.post('/api/assistant/teacher', requireAuth, async (req, res) => {
  return res.status(410).json({
    error: 'Teacher chat now runs inside a timed private virtual lesson.',
    code: 'VIRTUAL_LESSON_REQUIRED',
  });
});

app.get('/api/virtual-teachers', async (req, res) => {
  const db = await readDb();
  const globalPrice = sitePolicies(db).virtualLessonPricePer30MinutesMcoins;
  const characters = virtualTeacherCatalog(db)
    .filter((character) => !character.builtIn)
    .map((character) => publicVirtualTeacherCharacter(character, globalPrice))
    .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { sensitivity: 'base' }));
  const catalog = virtualTeacherCatalog(db)
    .map((character) => publicVirtualTeacherCharacter(character, globalPrice));
  res.json({ characters, catalog, catalogVersion: 2 });
});

app.get('/api/virtual-teachers/:characterId/image', async (req, res, next) => {
  try {
    const db = await readDb();
    const character = db.virtualTeacherCharacters.find((item) => item.id === req.params.characterId);
    if (!character?.imageKey) return res.status(404).json({ error: 'Character image not found.' });
    const bytes = await ARTIFACT_STORE.getBuffer(character.imageKey);
    res.setHeader('Content-Type', character.contentType || 'image/webp');
    res.setHeader('Content-Length', bytes.length);
    res.setHeader('Cache-Control', 'public, max-age=3600, stale-while-revalidate=86400');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    return res.send(bytes);
  } catch (error) {
    if (String(error?.code || '') === 'ENOENT' || Number(error?.$metadata?.httpStatusCode) === 404) {
      return res.status(404).json({ error: 'Character image not found.' });
    }
    return next(error);
  }
});

app.get('/api/virtual-teachers/:characterId/model', async (req, res, next) => {
  try {
    const db = await readDb();
    const character = db.virtualTeacherCharacters.find((item) => item.id === req.params.characterId);
    if (!character?.modelKey) return res.status(404).json({ error: 'Rigged character model not found.' });
    const bytes = await ARTIFACT_STORE.getBuffer(character.modelKey);
    res.setHeader('Content-Type', 'model/gltf-binary');
    res.setHeader('Content-Length', bytes.length);
    res.setHeader('Cache-Control', 'public, max-age=3600, stale-while-revalidate=86400');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    return res.send(bytes);
  } catch (error) {
    if (String(error?.code || '') === 'ENOENT' || Number(error?.$metadata?.httpStatusCode) === 404) {
      return res.status(404).json({ error: 'Rigged character model not found.' });
    }
    return next(error);
  }
});

app.get('/api/admin/virtual-teachers', requireAuth, requireAdmin, async (req, res) => {
  const globalPrice = sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins;
  res.json({
    characters: req.db.virtualTeacherCharacters
      .filter((character) => !BUILT_IN_VIRTUAL_TEACHER_IDS.has(character.id))
      .map((character) => publicVirtualTeacherCharacter(character, globalPrice))
      .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { sensitivity: 'base' })),
    catalog: virtualTeacherCatalog(req.db, { includeInactive: true })
      .map((character) => publicVirtualTeacherCharacter(character, globalPrice)),
    catalogVersion: 2,
    builtInCharacterIds: [...BUILT_IN_VIRTUAL_TEACHER_IDS],
  });
});

app.post(
  '/api/admin/virtual-teachers',
  requireAuth,
  requireAdmin,
  virtualTeacherUpload.fields([
    { name: 'image', maxCount: 1 },
    { name: 'model', maxCount: 1 },
  ]),
  async (req, res, next) => {
    let imageKey = '';
    let modelKey = '';
    try {
      const fields = virtualTeacherInput(req.body);
      const imageFile = req.files?.image?.[0];
      const modelFile = req.files?.model?.[0];
      if (!imageFile?.buffer?.length) return res.status(400).json({ error: 'Choose a character image.' });
      if (imageFile.buffer.length > VIRTUAL_TEACHER_IMAGE_MAX_BYTES) {
        return res.status(400).json({ error: 'The character image must be 8 MB or smaller.' });
      }
      const contentType = virtualTeacherImageContentType(imageFile.buffer);
      if (!contentType) return res.status(400).json({ error: 'The uploaded file is not a valid PNG, JPEG, or WebP image.' });
      const rig = modelFile?.buffer?.length ? inspectVirtualTeacherGlb(modelFile.buffer) : null;

      const characterId = id('virtual_teacher');
      const extension = contentType === 'image/png' ? 'png' : contentType === 'image/jpeg' ? 'jpg' : 'webp';
      imageKey = artifactKey(`virtual-teachers/${characterId}`, `portrait-${characterId}.${extension}`);
      await ARTIFACT_STORE.putBuffer(imageKey, imageFile.buffer, contentType);
      if (modelFile?.buffer?.length) {
        modelKey = artifactKey(`virtual-teachers/${characterId}`, `rig-${characterId}.glb`);
        await ARTIFACT_STORE.putBuffer(modelKey, modelFile.buffer, 'model/gltf-binary');
      }
      const now = new Date().toISOString();
      const character = {
        id: characterId,
        ...fields,
        builtIn: false,
        imageKey,
        contentType,
        modelKey,
        rig,
        createdBy: req.user.id,
        createdAt: now,
        updatedAt: now,
      };
      req.db.virtualTeacherCharacters.push(character);
      await writeDb(req.db);
      return res.status(201).json({
        character: publicVirtualTeacherCharacter(
          character,
          sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins,
        ),
        message: `${fields.name} is now available in the virtual teacher library.`,
      });
    } catch (error) {
      if (imageKey) await safeRemoveArtifact(imageKey).catch(() => {});
      if (modelKey) await safeRemoveArtifact(modelKey).catch(() => {});
      return next(error);
    }
  },
);

app.patch(
  '/api/admin/virtual-teachers/:characterId',
  requireAuth,
  requireAdmin,
  virtualTeacherUpload.fields([
    { name: 'image', maxCount: 1 },
    { name: 'model', maxCount: 1 },
  ]),
  async (req, res, next) => {
    let newImageKey = '';
    let newModelKey = '';
    try {
      const characterId = String(req.params.characterId || '').trim();
      const builtIn = BUILT_IN_VIRTUAL_TEACHER_IDS.has(characterId);
      const currentIndex = req.db.virtualTeacherCharacters.findIndex((item) => item.id === characterId);
      const currentRecord = currentIndex >= 0 ? req.db.virtualTeacherCharacters[currentIndex] : null;
      if (!builtIn && !currentRecord) return res.status(404).json({ error: 'Character not found.' });

      const builtInDefault = builtIn
        ? {
          ...BUILT_IN_VIRTUAL_TEACHERS[characterId],
          description: BUILT_IN_VIRTUAL_TEACHERS[characterId].style,
          builtIn: true,
          active: true,
        }
        : null;
      const current = mergedVirtualTeacher(currentRecord, builtInDefault);
      const fields = virtualTeacherInput(req.body, current);
      const imageFile = req.files?.image?.[0];
      const modelFile = req.files?.model?.[0];
      const removeImage = String(req.body.removeImage || '').trim().toLowerCase() === 'true';
      const removeModel = String(req.body.removeModel || '').trim().toLowerCase() === 'true';
      let contentType = currentRecord?.contentType || '';

      if (imageFile?.buffer?.length) {
        if (imageFile.buffer.length > VIRTUAL_TEACHER_IMAGE_MAX_BYTES) {
          return res.status(400).json({ error: 'The character image must be 8 MB or smaller.' });
        }
        contentType = virtualTeacherImageContentType(imageFile.buffer);
        if (!contentType) return res.status(400).json({ error: 'The uploaded file is not a valid PNG, JPEG, or WebP image.' });
        const extension = contentType === 'image/png' ? 'png' : contentType === 'image/jpeg' ? 'jpg' : 'webp';
        newImageKey = artifactKey(
          `virtual-teachers/${characterId}`,
          `portrait-${characterId}-${Date.now()}.${extension}`,
        );
        await ARTIFACT_STORE.putBuffer(newImageKey, imageFile.buffer, contentType);
      }
      if (modelFile?.buffer?.length) {
        const rig = inspectVirtualTeacherGlb(modelFile.buffer);
        newModelKey = artifactKey(
          `virtual-teachers/${characterId}`,
          `rig-${characterId}-${Date.now()}.glb`,
        );
        await ARTIFACT_STORE.putBuffer(newModelKey, modelFile.buffer, 'model/gltf-binary');
        current.rig = rig;
      }

      const imageKey = newImageKey || (removeImage && builtIn ? '' : currentRecord?.imageKey || '');
      if (!builtIn && !imageKey) {
        throw Object.assign(new Error('A custom character must keep a portrait image.'), { status: 400 });
      }
      const modelKey = newModelKey || (removeModel ? '' : currentRecord?.modelKey || '');
      const now = new Date().toISOString();
      const character = {
        ...(currentRecord || {}),
        id: characterId,
        ...fields,
        builtIn,
        imageKey,
        contentType: imageKey ? contentType : '',
        modelKey,
        rig: modelKey ? current.rig || currentRecord?.rig || null : null,
        createdBy: currentRecord?.createdBy || req.user.id,
        createdAt: currentRecord?.createdAt || now,
        updatedBy: req.user.id,
        updatedAt: now,
      };
      const nextRecords = [...req.db.virtualTeacherCharacters];
      if (currentIndex >= 0) nextRecords[currentIndex] = character;
      else nextRecords.push(character);
      const previewDb = { ...req.db, virtualTeacherCharacters: nextRecords };
      if (!virtualTeacherCatalog(previewDb).length) {
        throw Object.assign(new Error('Keep at least one virtual teacher active.'), { status: 409 });
      }
      req.db.virtualTeacherCharacters = nextRecords;
      await writeDb(req.db);

      if (currentRecord?.imageKey && currentRecord.imageKey !== imageKey) {
        await safeRemoveArtifact(currentRecord.imageKey).catch((error) => {
          console.error(`Could not remove replaced virtual teacher image ${currentRecord.imageKey}:`, error);
        });
      }
      if (currentRecord?.modelKey && currentRecord.modelKey !== modelKey) {
        await safeRemoveArtifact(currentRecord.modelKey).catch((error) => {
          console.error(`Could not remove replaced virtual teacher model ${currentRecord.modelKey}:`, error);
        });
      }
      return res.json({
        character: publicVirtualTeacherCharacter(
          character,
          sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins,
        ),
        message: `${fields.name} was updated.`,
      });
    } catch (error) {
      if (newImageKey) await safeRemoveArtifact(newImageKey).catch(() => {});
      if (newModelKey) await safeRemoveArtifact(newModelKey).catch(() => {});
      return next(error);
    }
  },
);

app.delete('/api/admin/virtual-teachers/:characterId', requireAuth, requireAdmin, async (req, res, next) => {
  try {
    const characterId = String(req.params.characterId || '');
    if (BUILT_IN_VIRTUAL_TEACHER_IDS.has(characterId)) {
      const currentIndex = req.db.virtualTeacherCharacters.findIndex((item) => item.id === characterId);
      const currentRecord = currentIndex >= 0 ? req.db.virtualTeacherCharacters[currentIndex] : null;
      const teacher = mergedVirtualTeacher(currentRecord, {
        ...BUILT_IN_VIRTUAL_TEACHERS[characterId],
        description: BUILT_IN_VIRTUAL_TEACHERS[characterId].style,
        builtIn: true,
        active: true,
      });
      const disabled = {
        ...(currentRecord || {}),
        ...teacher,
        id: characterId,
        builtIn: true,
        active: false,
        createdBy: currentRecord?.createdBy || req.user.id,
        createdAt: currentRecord?.createdAt || new Date().toISOString(),
        updatedBy: req.user.id,
        updatedAt: new Date().toISOString(),
      };
      const nextRecords = [...req.db.virtualTeacherCharacters];
      if (currentIndex >= 0) nextRecords[currentIndex] = disabled;
      else nextRecords.push(disabled);
      const previewDb = { ...req.db, virtualTeacherCharacters: nextRecords };
      if (!virtualTeacherCatalog(previewDb).length) {
        return res.status(409).json({ error: 'Keep at least one virtual teacher active.' });
      }
      req.db.virtualTeacherCharacters = nextRecords;
      await writeDb(req.db);
      return res.json({
        id: disabled.id,
        disabled: true,
        character: publicVirtualTeacherCharacter(
          disabled,
          sitePolicies(req.db).virtualLessonPricePer30MinutesMcoins,
        ),
        message: `${disabled.name} was removed from the public teacher library. You can enable the character again from Edit.`,
      });
    }
    const index = req.db.virtualTeacherCharacters.findIndex((item) => item.id === characterId);
    if (index < 0) return res.status(404).json({ error: 'Character not found.' });
    const [character] = req.db.virtualTeacherCharacters.splice(index, 1);
    await writeDb(req.db);
    if (character.imageKey) {
      await safeRemoveArtifact(character.imageKey).catch((error) => {
        console.error(`Could not remove deleted virtual teacher artifact ${character.imageKey}:`, error);
      });
    }
    if (character.modelKey) {
      await safeRemoveArtifact(character.modelKey).catch((error) => {
        console.error(`Could not remove deleted virtual teacher model ${character.modelKey}:`, error);
      });
    }
    return res.json({ id: character.id, message: `${character.name} was deleted.` });
  } catch (error) {
    return next(error);
  }
});

app.get('/api/media-transcriptions/capabilities', async (req, res) => {
  res.json(muscriptorAvailability());
});

app.get('/api/catalog', async (req, res) => {
  const db = await readDb();
  const { updatedBy: _updatedBy, ...publicPolicies } = sitePolicies(db);
  const withdrawalFeeRate = publicPolicies.withdrawalFeePercent / 100;
  const marketplaceFeeRate = publicPolicies.marketplaceFeePercent / 100;
  res.json({
    products: Object.values(PRODUCTS).filter((product) => !product.legacy),
    withdrawalFeeRate,
    withdrawalFeeLabel: `${publicPolicies.withdrawalFeePercent}% cash-out fee for every account`,
    marketplaceFeeRate,
    teacherMarketplace: teacherMarketplaceTerms(publicPolicies),
    mcoinsPerUsd: MCOINS_PER_USD,
    translationMcoinCosts: {
      subscriber: SUBSCRIBER_TRANSLATION_MCOIN_COST,
      free: FREE_TRANSLATION_MCOIN_COST,
    },
    policies: publicPolicies,
  });
});

function registrationContactExists(db, channel, destination) {
  if (channel === 'email') {
    return db.users.some((user) => String(user.email || '').trim().toLowerCase() === destination);
  }
  const phone = normalizePhone(destination);
  return db.users.some((user) => phone && normalizePhone(user.phone) === phone);
}

function registrationOtpFailure(res, error) {
  if (error instanceof RegistrationOtpError) {
    return res.status(error.status).json({ error: error.message });
  }
  console.error('Registration OTP failed:', error);
  return res.status(500).json({ error: 'Account verification failed. Try again later.' });
}

app.post('/api/auth/register/otp', async (req, res) => {
  const channel = String(req.body.channel || '').trim().toLowerCase();
  const destination = channel === 'email' ? req.body.email : req.body.phone;
  const db = await readDb();
  const policies = sitePolicies(db);
  if (!policies.registrationEnabled) {
    return res.status(403).json({ error: 'New account registration is temporarily closed.' });
  }

  try {
    const normalizedDestination = REGISTRATION_OTP.normalizeContact(channel, destination);
    if (registrationContactExists(db, channel, normalizedDestination)) {
      return res.status(409).json({
        error: `An account already exists for this ${channel === 'email' ? 'email address' : 'phone number'}.`,
      });
    }
    const challenge = await REGISTRATION_OTP.requestCode(db, {
      channel,
      destination: normalizedDestination,
    });
    await writeDb(db);
    return res.status(202).json({
      challengeId: challenge.challengeId,
      channel: challenge.channel,
      destinationHint: challenge.destinationHint,
      expiresInSeconds: challenge.expiresInSeconds,
      message: `We sent a six-digit code to ${challenge.destinationHint}.`,
    });
  } catch (error) {
    return registrationOtpFailure(res, error);
  }
});

app.post('/api/auth/register', async (req, res) => {
  const name = String(req.body.name || '').trim();
  const password = String(req.body.password || '');
  const birthDate = String(req.body.birthDate || '').trim();
  const challengeId = String(req.body.challengeId || '').trim();
  const verificationCode = String(req.body.verificationCode || '').trim();
  const luckyCode = String(req.body.luckyCode || '').trim();
  const db = await readDb();
  const policies = sitePolicies(db);
  if (!policies.registrationEnabled) return res.status(403).json({ error: 'New account registration is temporarily closed.' });
  if (name.length < 2) return res.status(400).json({ error: 'Name must contain at least 2 characters.' });

  let email = '';
  let phone = '';
  try {
    if (String(req.body.email || '').trim()) {
      email = REGISTRATION_OTP.normalizeContact('email', req.body.email);
    }
    if (String(req.body.phone || '').trim()) {
      phone = REGISTRATION_OTP.normalizeContact('phone', req.body.phone);
    }
  } catch (error) {
    return registrationOtpFailure(res, error);
  }
  if (!email && !phone) {
    return res.status(400).json({ error: 'Enter either an email address or a phone number.' });
  }
  if (password.length < policies.minimumPasswordLength) {
    return res.status(400).json({ error: `Password must contain at least ${policies.minimumPasswordLength} characters.` });
  }
  if (policies.minimumSignupAge > 0) {
    if (!birthDate || ageOnDate(birthDate) < policies.minimumSignupAge) {
      return res.status(400).json({ error: `You must be at least ${policies.minimumSignupAge} years old to register.` });
    }
  }
  if ((policies.minimumSignupAge > 0 || policies.policyNotice || policies.termsUrl || policies.privacyUrl)
    && req.body.termsAccepted !== true) {
    return res.status(400).json({ error: 'Accept the registration rules and policies to continue.' });
  }
  if (email && registrationContactExists(db, 'email', email)) {
    return res.status(409).json({ error: 'An account already exists for this email address.' });
  }
  if (phone && registrationContactExists(db, 'phone', phone)) {
    return res.status(409).json({ error: 'An account already exists for this phone number.' });
  }

  const luckyResult = resolveSignupLuckyCode(db, luckyCode);
  if (luckyResult.error) return res.status(400).json({ error: luckyResult.error });

  const challenge = db.registrationVerifications.find((candidate) => candidate.id === challengeId);
  if (!challenge) {
    return res.status(400).json({ error: 'Request and verify a new code before creating the account.' });
  }
  const verifiedDestination = challenge.channel === 'email' ? email : phone;
  if (!verifiedDestination) {
    return res.status(400).json({ error: `Enter the ${challenge.channel} address that received the code.` });
  }
  // Only persist the contact method proven by this challenge. This prevents a
  // caller from attaching an unverified second login identifier to the account.
  if (challenge.channel === 'email') phone = '';
  if (challenge.channel === 'phone') email = '';

  let verification;
  try {
    verification = REGISTRATION_OTP.verifyCode(db, {
      challengeId,
      channel: challenge.channel,
      destination: verifiedDestination,
      code: verificationCode,
    });
  } catch (error) {
    await writeDb(db);
    return registrationOtpFailure(res, error);
  }

  const { salt, hash } = hashPassword(password);
  const userId = id('user');
  const createdAt = new Date().toISOString();
  const user = {
    id: userId,
    friendId: createFriendId(db.users, userId),
    name,
    email,
    phone,
    passwordHash: hash,
    passwordSalt: salt,
    verifiedEmailAt: verification.channel === 'email' ? verification.verifiedAt : null,
    verifiedPhoneAt: verification.channel === 'phone' ? verification.verifiedAt : null,
    mcoins: policies.welcomeMcoins,
    withdrawableMcoins: 0,
    pro: false,
    proStatus: 'INACTIVE',
    subscriptionTier: '',
    subscriptionInterval: null,
    subscriptionStartedAt: null,
    paypalSubscriptionId: null,
    translationUsage: {
      period: currentTranslationPeriod(),
      includedUsed: 0,
    },
    readySheetUploadUsage: {
      period: currentTranslationPeriod(),
      freeUsed: 0,
    },
    ageRequirementConfirmedAt: policies.minimumSignupAge > 0 ? createdAt : null,
    policyAcceptedAt: req.body.termsAccepted === true ? createdAt : null,
    createdAt,
  };
  const token = createSession(db, user.id);
  db.users.push(user);
  if (policies.welcomeMcoins > 0) addLedger(db, user.id, policies.welcomeMcoins, 'welcome_bonus', 'Configured welcome balance');
  applySignupLuckyCode(db, user, luckyResult.claim);
  db.authEvents.push({ id: id('auth'), userId: user.id, type: 'register_verified', channel: verification.channel, createdAt });
  await writeDb(db);
  res.status(201).json({ token, user: safeUser(user) });
});

app.post('/api/auth/login', async (req, res) => {
  const identifier = String(req.body.identifier || req.body.email || '').trim();
  const email = identifier.toLowerCase();
  const phone = normalizePhone(identifier);
  const password = String(req.body.password || '');
  const db = await readDb();
  const user = db.users.find((candidate) => (
    String(candidate.email || '').toLowerCase() === email
    || (phone.length >= 7 && normalizePhone(candidate.phone) === phone)
  ));
  if (!user || !user.passwordHash) return res.status(401).json({ error: 'Email/phone or password is incorrect.' });
  const { hash } = hashPassword(password, user.passwordSalt);
  const matches = crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(user.passwordHash, 'hex'));
  if (!matches) return res.status(401).json({ error: 'Email/phone or password is incorrect.' });
  const token = createSession(db, user.id);
  db.sessions = db.sessions.filter((session) => new Date(session.expiresAt).getTime() >= Date.now());
  user.lastLoginAt = new Date().toISOString();
  user.loginCount = Number(user.loginCount || 0) + 1;
  db.authEvents.push({ id: id('auth'), userId: user.id, type: 'login', createdAt: user.lastLoginAt });
  db.authEvents = db.authEvents.slice(-5000);
  await writeDb(db);
  res.json({ token, user: safeUser(user) });
});

app.get('/api/auth/me', requireAuth, async (req, res) => {
  res.json({ user: safeUser(req.user) });
});

app.put('/api/profile/avatar', requireAuth, async (req, res) => {
  const avatarDataUrl = String(req.body.avatarDataUrl || '').trim();
  if (!avatarDataUrl) {
    req.user.avatarUrl = '';
    await writeDb(req.db);
    return res.json({ user: safeUser(req.user) });
  }
  const match = avatarDataUrl.match(/^data:image\/(jpeg|png|webp|gif);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) return res.status(400).json({ error: 'Use a JPEG, PNG, WebP, or GIF profile picture.' });
  const imageBytes = Buffer.from(match[2], 'base64');
  if (!imageBytes.length || imageBytes.length > 384 * 1024) {
    return res.status(400).json({ error: 'Profile pictures must be 384 KB or smaller after resizing.' });
  }
  req.user.avatarUrl = avatarDataUrl;
  await writeDb(req.db);
  res.json({ user: safeUser(req.user) });
});

app.post('/api/ready-sheet-uploads', requireAuth, async (req, res, next) => {
  let filename = sanitizeFilename(String(req.body.filename || 'ready-to-play-sheet'));
  let format = readySheetFormat(filename);
  const directUploadReceipt = String(req.body.directUploadReceipt || '').trim();
  const contentBase64 = String(req.body.contentBase64 || '').trim();
  let pendingKey = '';
  let finalKey = '';

  if (!format) {
    return res.status(400).json({ error: 'Only ready-to-play JSON or MIDI sheets count as direct uploads.' });
  }

  // Older clients only record allowance usage. Current clients also send the
  // normalized ready-to-play file so it can become part of the account cloud library.
  if (!directUploadReceipt && !contentBase64) {
    const charged = chargeReadySheetUpload(req.db, req.user, filename);
    if (!charged) {
      return res.status(402).json({
        error: `You need ${READY_SHEET_UPLOAD_MCOIN_COST} Mcoin to upload this ready-to-play sheet.`,
        costMcoins: READY_SHEET_UPLOAD_MCOIN_COST,
      });
    }
    await writeDb(req.db);
    return res.status(201).json({
      user: safeUser(req.user),
      costMcoins: charged.costMcoins,
      paymentMethod: charged.paymentMethod,
      personalSong: null,
    });
  }

  try {
    let bytes;
    if (directUploadReceipt) {
      const upload = await DIRECT_UPLOADS.inspect(directUploadReceipt, {
        userId: req.user.id,
        purpose: 'personal-song',
      });
      pendingKey = upload.key;
      filename = sanitizeFilename(upload.filename);
      format = readySheetFormat(filename);
      if (!format) throw Object.assign(new Error('Use a ready-to-play JSON or MIDI sheet.'), { status: 400 });
      bytes = await ARTIFACT_STORE.getBuffer(upload.key);
    } else {
      bytes = Buffer.from(contentBase64, 'base64');
    }

    validateMarketplaceAsset(format, filename, bytes);
    const metadata = readySheetMetadata(bytes, format, {
      title: req.body.title,
      artist: req.body.artist,
    });
    const instrument = String(req.body.instrument || 'piano').trim().toLowerCase();
    if (!INSTRUMENTS[instrument]) {
      throw Object.assign(new Error('Choose a supported instrument for this song.'), { status: 400 });
    }
    const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    const duplicate = req.db.personalSongs.find((song) => (
      song.userId === req.user.id && song.sha256 === sha256
    ));
    if (duplicate) {
      if (pendingKey) await safeRemoveArtifact(pendingKey);
      return res.json({
        user: safeUser(req.user),
        costMcoins: 0,
        paymentMethod: 'existing',
        personalSong: publicPersonalSong(duplicate),
        alreadySaved: true,
      });
    }

    const charged = chargeReadySheetUpload(req.db, req.user, filename);
    if (!charged) {
      if (pendingKey) await safeRemoveArtifact(pendingKey);
      return res.status(402).json({
        error: `You need ${READY_SHEET_UPLOAD_MCOIN_COST} Mcoin to upload this ready-to-play sheet.`,
        costMcoins: READY_SHEET_UPLOAD_MCOIN_COST,
      });
    }

    const songId = id('song');
    const userSegment = String(req.user.id).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80);
    finalKey = artifactKey(`personal-songs/${userSegment}`, `${songId}-${filename}`);
    if (pendingKey) await ARTIFACT_STORE.promote(pendingKey, finalKey);
    else await ARTIFACT_STORE.putBuffer(
      finalKey,
      bytes,
      format === 'JSON' ? 'application/json' : 'audio/midi',
    );
    pendingKey = '';

    const personalSong = {
      id: songId,
      userId: req.user.id,
      title: String(metadata.title || path.basename(filename, path.extname(filename)) || 'Untitled song').slice(0, 160),
      artist: String(metadata.artist || '').slice(0, 120),
      instrument,
      format,
      filename,
      assetPath: finalKey,
      size: bytes.length,
      sha256,
      createdAt: new Date().toISOString(),
    };
    req.db.personalSongs.push(personalSong);
    try {
      await writeDb(req.db);
    } catch (error) {
      await safeRemoveArtifact(finalKey);
      throw error;
    }
    finalKey = '';
    return res.status(201).json({
      user: safeUser(req.user),
      costMcoins: charged.costMcoins,
      paymentMethod: charged.paymentMethod,
      personalSong: publicPersonalSong(personalSong),
      alreadySaved: false,
    });
  } catch (error) {
    if (pendingKey) await safeRemoveArtifact(pendingKey);
    if (finalKey) await safeRemoveArtifact(finalKey);
    if (!error.status && /invalid|must contain|smaller than|filename|file type/i.test(error.message || '')) {
      error.status = 400;
    }
    return next(error);
  }
});

app.post('/api/auth/change-password', requireAuth, async (req, res) => {
  const password = String(req.body.password || '');
  const minimumLength = sitePolicies(req.db).minimumPasswordLength;
  if (password.length < minimumLength) return res.status(400).json({ error: `Your new password must contain at least ${minimumLength} characters.` });
  const { salt, hash } = hashPassword(password);
  req.user.passwordHash = hash;
  req.user.passwordSalt = salt;
  req.user.mustChangePassword = false;
  const currentToken = bearerToken(req);
  req.db.sessions = req.db.sessions.filter((session) => session.userId !== req.user.id || session.token === currentToken);
  await writeDb(req.db);
  res.json({ user: safeUser(req.user) });
});

app.post('/api/auth/logout', requireAuth, async (req, res) => {
  const token = bearerToken(req);
  const tokenHash = hashSessionToken(token);
  req.db.sessions = req.db.sessions.filter((session) => session.tokenHash !== tokenHash && session.token !== token);
  await writeDb(req.db);
  res.json({ ok: true });
});

app.get('/api/teachers', async (req, res) => {
  const db = await readDb();
  const policies = sitePolicies(db);
  const marketplace = teacherMarketplaceTerms(policies);
  if (!policies.teacherDirectoryEnabled) {
    return res.json({ teachers: [], marketplace });
  }
  const viewer = authUser(req, db);
  const query = String(req.query.query || '').trim().toLowerCase();
  const instrument = String(req.query.instrument || '').trim().toLowerCase();
  const lessonMode = String(req.query.lessonMode || '').trim().toLowerCase();
  const level = String(req.query.level || '').trim().toLowerCase();
  const teachers = db.teacherProfiles
    .filter((profile) => profile.published !== false)
    .map((profile) => publicTeacherProfile(profile, db, viewer?.id))
    .filter(Boolean)
    .filter((teacher) => !instrument || teacher.instruments.includes(instrument))
    .filter((teacher) => !lessonMode || teacher.lessonModes.includes(lessonMode))
    .filter((teacher) => !level || teacher.levels.includes(level))
    .filter((teacher) => !query || [
      teacher.name,
      teacher.headline,
      teacher.bio,
      teacher.location,
      ...teacher.instruments,
      ...teacher.languages,
    ].join(' ').toLowerCase().includes(query))
    .sort((a, b) => (
      Number(b.ranking.totalPoints) - Number(a.ranking.totalPoints)
      || Number(b.reviewSummary.averageRating) - Number(a.reviewSummary.averageRating)
      || Number(b.reviewSummary.reviewCount) - Number(a.reviewSummary.reviewCount)
      || String(b.updatedAt || b.createdAt).localeCompare(String(a.updatedAt || a.createdAt))
    ));
  return res.json({ teachers, marketplace });
});

app.get('/api/teachers/me', requireAuth, async (req, res) => {
  const profile = req.db.teacherProfiles.find((item) => item.userId === req.user.id);
  res.json({
    teacher: profile ? publicTeacherProfile(profile, req.db, req.user.id) : null,
    marketplace: teacherMarketplaceTerms(sitePolicies(req.db)),
  });
});

app.put('/api/teachers/me', requireAuth, async (req, res) => {
  const policies = sitePolicies(req.db);
  let profile = req.db.teacherProfiles.find((item) => item.userId === req.user.id);
  if (!profile && !policies.teacherApplicationsEnabled) {
    return res.status(403).json({ error: 'New teacher profiles are temporarily paused.' });
  }
  const headline = String(req.body.headline || '').trim().slice(0, 100);
  const bio = String(req.body.bio || '').trim().slice(0, 1200);
  const instruments = cleanStringList(req.body.instruments, {
    maximum: 8,
    allowed: new Set(Object.keys(MUSCRIPTOR_INSTRUMENTS)),
  });
  const levels = cleanStringList(req.body.levels, { maximum: 3, allowed: TEACHER_LEVELS });
  const lessonModes = cleanStringList(req.body.lessonModes, { maximum: 2, allowed: TEACHER_LESSON_MODES });
  const languages = cleanStringList(req.body.languages, { maximum: 8 })
    .map((language) => language.slice(0, 40));
  const location = String(req.body.location || '').trim().slice(0, 100);
  const availability = String(req.body.availability || '').trim().slice(0, 200);
  const hourlyRateMcoins = Number(req.body.hourlyRateMcoins || 0);
  const published = req.body.published !== false;

  if (headline.length < 3) return res.status(400).json({ error: 'Add a short teaching headline.' });
  if (bio.length < 10) return res.status(400).json({ error: 'Tell students a little more about your teaching.' });
  if (!instruments.length) return res.status(400).json({ error: 'Choose at least one instrument.' });
  if (!levels.length) return res.status(400).json({ error: 'Choose at least one student level.' });
  if (!lessonModes.length) return res.status(400).json({ error: 'Choose online lessons, in-person lessons, or both.' });
  if (!Number.isFinite(hourlyRateMcoins)
    || hourlyRateMcoins < policies.minimumTeacherHourlyRateMcoins
    || (policies.maximumTeacherHourlyRateMcoins > 0
      && hourlyRateMcoins > policies.maximumTeacherHourlyRateMcoins)) {
    const maximum = policies.maximumTeacherHourlyRateMcoins > 0
      ? policies.maximumTeacherHourlyRateMcoins.toLocaleString()
      : 'unlimited';
    return res.status(400).json({
      error: `Enter an hourly rate from ${policies.minimumTeacherHourlyRateMcoins.toLocaleString()} Mcoins to ${maximum}.`,
    });
  }

  const now = new Date().toISOString();
  if (profile) {
    Object.assign(profile, {
      headline,
      bio,
      instruments,
      levels,
      lessonModes,
      languages,
      location,
      availability,
      hourlyRateMcoins: Number(hourlyRateMcoins.toFixed(2)),
      published,
      updatedAt: now,
    });
  } else {
    profile = {
      id: id('teacher'),
      userId: req.user.id,
      headline,
      bio,
      instruments,
      levels,
      lessonModes,
      languages,
      location,
      availability,
      hourlyRateMcoins: Number(hourlyRateMcoins.toFixed(2)),
      published,
      createdAt: now,
    };
    req.db.teacherProfiles.push(profile);
  }
  await writeDb(req.db);
  res.status(profile.createdAt === now ? 201 : 200).json({
    teacher: publicTeacherProfile(profile, req.db, req.user.id),
  });
});

app.get('/api/teachers/:teacherProfileId/reviews', async (req, res) => {
  const db = await readDb();
  const viewer = authUser(req, db);
  const profile = db.teacherProfiles.find((item) => item.id === req.params.teacherProfileId && item.published !== false);
  if (!profile) return res.status(404).json({ error: 'Teacher profile not found.' });
  const reviews = db.teacherReviews
    .filter((review) => review.teacherProfileId === profile.id)
    .sort((a, b) => String(b.updatedAt || b.createdAt).localeCompare(String(a.updatedAt || a.createdAt)))
    .map((review) => publicTeacherReview(review, db, viewer?.id));
  return res.json({
    reviews,
    summary: teacherReviewSummary(db, profile.id),
    reviewsEnabled: sitePolicies(db).teacherReviewsEnabled,
  });
});

app.post('/api/teachers/:teacherProfileId/reviews', requireAuth, async (req, res) => {
  if (!sitePolicies(req.db).teacherReviewsEnabled) {
    return res.status(403).json({ error: 'New teacher reviews are temporarily paused.' });
  }
  const profile = req.db.teacherProfiles.find(
    (item) => item.id === req.params.teacherProfileId && item.published !== false,
  );
  if (!profile) return res.status(404).json({ error: 'Teacher profile not found.' });
  if (profile.userId === req.user.id) return res.status(403).json({ error: 'Teachers cannot review themselves.' });
  if (!hasTeacherConversation(req.db, req.user.id, profile.userId)) {
    return res.status(403).json({ error: 'Start a private conversation with this teacher before leaving a review.' });
  }

  const rating = Number(req.body.rating);
  const comment = String(req.body.comment || '').trim();
  if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
    return res.status(400).json({ error: 'Choose a star rating from 1 to 5.' });
  }
  if (comment.length < 2 || comment.length > 1000) {
    return res.status(400).json({ error: 'Write a review between 2 and 1,000 characters.' });
  }

  const now = new Date().toISOString();
  let review = req.db.teacherReviews.find(
    (item) => item.teacherProfileId === profile.id && item.studentId === req.user.id,
  );
  const created = !review;
  if (review) {
    review.rating = rating;
    review.comment = comment;
    review.updatedAt = now;
  } else {
    review = {
      id: id('teacher-review'),
      teacherProfileId: profile.id,
      teacherUserId: profile.userId,
      studentId: req.user.id,
      rating,
      comment,
      createdAt: now,
    };
    req.db.teacherReviews.push(review);
  }
  await writeDb(req.db);
  return res.status(created ? 201 : 200).json({
    review: publicTeacherReview(review, req.db, req.user.id),
    summary: teacherReviewSummary(req.db, profile.id),
  });
});

app.get('/api/listings', async (req, res) => {
  const db = await readDb();
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
    .map((listing) => publicListing(listing, db, viewer?.id))
    .sort((a, b) => (
      Number(b.seller?.ranking?.totalPoints || 0) - Number(a.seller?.ranking?.totalPoints || 0)
      || Number(b.reviewSummary?.averageRating || 0) - Number(a.reviewSummary?.averageRating || 0)
      || String(b.createdAt).localeCompare(String(a.createdAt))
    ));
  res.json({ listings });
});

app.get('/api/listings/:listingId/reviews', async (req, res) => {
  const db = await readDb();
  const viewer = authUser(req, db);
  const listing = db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Music sheet not found.' });
  const reviews = db.listingReviews
    .filter((review) => review.listingId === listing.id)
    .sort((a, b) => String(b.updatedAt || b.createdAt).localeCompare(String(a.updatedAt || a.createdAt)))
    .map((review) => publicListingReview(review, db, viewer?.id));
  return res.json({ reviews, summary: listingReviewSummary(db, listing.id) });
});

app.post('/api/listings/:listingId/reviews', requireAuth, async (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Music sheet not found.' });
  if (listing.sellerId === req.user.id) {
    return res.status(403).json({ error: 'Composers cannot review their own music sheets.' });
  }
  const purchased = req.db.purchases.some(
    (purchase) => purchase.listingId === listing.id && purchase.buyerId === req.user.id,
  );
  if (!purchased) return res.status(403).json({ error: 'Only verified buyers can review this music sheet.' });

  const rating = Number(req.body.rating);
  const comment = String(req.body.comment || '').trim();
  if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
    return res.status(400).json({ error: 'Choose a star rating from 1 to 5.' });
  }
  if (comment.length < 2 || comment.length > 1000) {
    return res.status(400).json({ error: 'Write a review between 2 and 1,000 characters.' });
  }

  const now = new Date().toISOString();
  let review = req.db.listingReviews.find(
    (item) => item.listingId === listing.id && item.userId === req.user.id,
  );
  if (review) {
    review.rating = rating;
    review.comment = comment;
    review.updatedAt = now;
  } else {
    review = {
      id: id('review'),
      listingId: listing.id,
      userId: req.user.id,
      rating,
      comment,
      createdAt: now,
    };
    req.db.listingReviews.push(review);
  }
  await writeDb(req.db);
  return res.status(review.updatedAt ? 200 : 201).json({
    review: publicListingReview(review, req.db, req.user.id),
    summary: listingReviewSummary(req.db, listing.id),
  });
});

app.delete('/api/listings/:listingId/reviews/:reviewId', requireAuth, async (req, res) => {
  return res.status(403).json({ error: 'Published reviews are permanent and cannot be deleted by composers.' });
});

app.get('/api/composers/:composerId', async (req, res) => {
  const db = await readDb();
  const viewer = authUser(req, db);
  const composer = db.users.find((item) => item.id === req.params.composerId);
  const listings = db.listings.filter((listing) => listing.sellerId === composer?.id);
  if (!composer || !listings.length) return res.status(404).json({ error: 'Composer profile not found.' });
  return res.json({
    composer: publicComposer(composer, db, viewer?.id),
    listings: listings
      .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
      .map((listing) => publicListing(listing, db, viewer?.id)),
  });
});

app.post('/api/composers/:composerId/follow', requireAuth, async (req, res) => {
  const composer = req.db.users.find((item) => item.id === req.params.composerId);
  const hasListings = req.db.listings.some((listing) => listing.sellerId === composer?.id);
  if (!composer || !hasListings) return res.status(404).json({ error: 'Composer profile not found.' });
  if (composer.id === req.user.id) return res.status(400).json({ error: 'You cannot follow your own composer profile.' });
  const existing = req.db.composerFollows.find(
    (follow) => follow.composerId === composer.id && follow.followerId === req.user.id,
  );
  if (!existing) {
    req.db.composerFollows.push({
      id: id('follow'),
      composerId: composer.id,
      followerId: req.user.id,
      createdAt: new Date().toISOString(),
    });
    await writeDb(req.db);
  }
  return res.json({ composer: publicComposer(composer, req.db, req.user.id) });
});

app.delete('/api/composers/:composerId/follow', requireAuth, async (req, res) => {
  const composer = req.db.users.find((item) => item.id === req.params.composerId);
  const hasListings = req.db.listings.some((listing) => listing.sellerId === composer?.id);
  if (!composer || !hasListings) return res.status(404).json({ error: 'Composer profile not found.' });
  req.db.composerFollows = req.db.composerFollows.filter(
    (follow) => !(follow.composerId === composer.id && follow.followerId === req.user.id),
  );
  await writeDb(req.db);
  return res.json({ composer: publicComposer(composer, req.db, req.user.id) });
});

app.get('/api/library', requireAuth, async (req, res) => {
  const backfilled = backfillGeneratedPersonalSongs(req.db, req.user.id);
  if (backfilled) await writeDb(req.db);
  const personalSongs = req.db.personalSongs
    .filter((song) => song.userId === req.user.id)
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map(publicPersonalSong);
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
  res.json({ personalSongs, purchasedSongs, sellingSongs });
});

app.get('/api/personal-songs/:songId/download', requireAuth, async (req, res) => {
  const song = req.db.personalSongs.find((item) => (
    item.id === req.params.songId && item.userId === req.user.id
  ));
  if (!song) return res.status(404).json({ error: 'Song not found in your cloud library.' });
  return ARTIFACT_STORE.sendDownload(
    res,
    song.assetPath,
    song.filename,
    song.format === 'JSON' ? 'application/json' : 'audio/midi',
  );
});

app.delete('/api/personal-songs/:songId', requireAuth, async (req, res, next) => {
  const index = req.db.personalSongs.findIndex((item) => (
    item.id === req.params.songId && item.userId === req.user.id
  ));
  if (index < 0) return res.status(404).json({ error: 'Song not found in your cloud library.' });
  const [song] = req.db.personalSongs.splice(index, 1);
  try {
    if (song.sourceJobId) {
      const sourceJob = [
        ...(req.db.mediaTranscriptionJobs || []),
        ...(req.db.scoreTranslationJobs || []),
      ].find((job) => job.id === song.sourceJobId && job.userId === req.user.id);
      if (sourceJob) {
        sourceJob.personalSongId = null;
        sourceJob.personalSongHiddenAt = new Date().toISOString();
      }
    }
    await writeDb(req.db);
    if (!song.sourceJobId) {
      await safeRemoveArtifact(song.assetPath).catch((error) => {
        console.error('Orphaned personal song artifact could not be removed:', error);
      });
    }
    return res.json({ ok: true, song: publicPersonalSong(song) });
  } catch (error) {
    return next(error);
  }
});

function listingCommercialTerms(input, policies, fallback = {}) {
  const mode = String(input.listingMode ?? fallback.listingMode ?? 'sale').trim().toLowerCase();
  if (!['sale', 'free', 'listener-reward'].includes(mode)) {
    return { error: 'Choose Sell, Free, or Reward listeners.' };
  }

  if (mode === 'listener-reward') {
    if (!policies.listenerRewardsEnabled) return { error: 'Listener rewards are disabled by the site rules.' };
    const listenerRewardMcoins = mcoinAmount(input.listenerRewardMcoins ?? fallback.listenerRewardMcoins);
    if (!Number.isFinite(listenerRewardMcoins) || listenerRewardMcoins <= 0) {
      return { error: 'Listener reward must be greater than 0 Mcoins.' };
    }
    if (policies.maximumListenerRewardMcoins > 0 && listenerRewardMcoins > policies.maximumListenerRewardMcoins) {
      return { error: `Listener reward cannot exceed ${policies.maximumListenerRewardMcoins.toLocaleString()} Mcoins per listener.` };
    }
    return { listingMode: mode, priceMcoins: 0, listenerRewardMcoins };
  }

  if (mode === 'free') return { listingMode: mode, priceMcoins: 0, listenerRewardMcoins: 0 };

  const priceMcoins = mcoinAmount(input.priceMcoins ?? fallback.priceMcoins);
  if (!Number.isFinite(priceMcoins) || priceMcoins < policies.minimumMarketplacePriceMcoins) {
    return { error: `Price must be at least ${policies.minimumMarketplacePriceMcoins.toLocaleString()} Mcoins.` };
  }
  if (policies.maximumMarketplacePriceMcoins > 0 && priceMcoins > policies.maximumMarketplacePriceMcoins) {
    return { error: `Price cannot exceed ${policies.maximumMarketplacePriceMcoins.toLocaleString()} Mcoins.` };
  }
  return { listingMode: mode, priceMcoins, listenerRewardMcoins: 0 };
}

app.post('/api/listings', requireAuth, async (req, res) => {
  const artist = String(req.body.artist || '').trim();
  const title = String(req.body.title || '').trim();
  const instrument = String(req.body.instrument || '').trim().toLowerCase();
  const format = String(req.body.format || '').trim().toUpperCase();
  const description = String(req.body.description || '').trim().slice(0, 800);
  const filename = sanitizeFilename(req.body.filename || `${title}.${format.toLowerCase()}`);
  const contentBase64 = String(req.body.contentBase64 || '');
  const rightsConfirmed = req.body.rightsConfirmed === true;
  const feeConfirmed = req.body.feeConfirmed === true;
  const policies = sitePolicies(req.db);
  const commercial = listingCommercialTerms(req.body, policies);

  if (!artist || !title) return res.status(400).json({ error: 'Artist and song title are required.' });
  if (!rightsConfirmed) return res.status(400).json({ error: 'Confirm that you own the rights or have permission to publish this file.' });
  if (!INSTRUMENTS[instrument]) return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  if (!['JSON', 'PDF', 'MIDI', 'MUSICXML'].includes(format)) return res.status(400).json({ error: 'Unsupported listing format.' });
  if (commercial.error) return res.status(400).json({ error: commercial.error });
  if (commercial.listingMode === 'sale' && policies.marketplaceFeePercent > 0 && !feeConfirmed) {
    return res.status(400).json({ error: `Confirm the ${policies.marketplaceFeePercent}% sale fee before publishing.` });
  }
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
  const storedKey = await ARTIFACT_STORE.putBuffer(
    artifactKey('marketplace', storedName),
    bytes,
    'application/octet-stream',
  );
  const listing = {
    id: listingId,
    sellerId: req.user.id,
    artist,
    title,
    instrument,
    format,
    listingMode: commercial.listingMode,
    priceMcoins: commercial.priceMcoins,
    listenerRewardMcoins: commercial.listenerRewardMcoins,
    rewardPaidMcoins: 0,
    description,
    cover: INSTRUMENTS[instrument].cover,
    filename,
    assetPath: storedKey,
    demo: false,
    rightsConfirmed: true,
    feeConfirmed: commercial.listingMode !== 'sale' || feeConfirmed,
    marketplaceFeeRate: policies.marketplaceFeePercent / 100,
    createdAt: new Date().toISOString(),
  };
  req.db.listings.push(listing);
  await writeDb(req.db);
  res.status(201).json({ listing: publicListing(listing, req.db, req.user.id) });
});

app.put('/api/listings/:listingId', requireAuth, async (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  if (listing.sellerId !== req.user.id) {
    return res.status(403).json({ error: 'Only the seller can amend this listing.' });
  }

  const artist = String(req.body.artist ?? listing.artist).trim();
  const title = String(req.body.title ?? listing.title).trim();
  const instrument = String(req.body.instrument ?? listing.instrument).trim().toLowerCase();
  const description = String(req.body.description ?? listing.description).trim().slice(0, 800);
  const policies = sitePolicies(req.db);
  const commercial = listingCommercialTerms(req.body, policies, {
    listingMode: listingMode(listing),
    priceMcoins: listing.priceMcoins,
    listenerRewardMcoins: listing.listenerRewardMcoins,
  });
  if (!artist || !title) return res.status(400).json({ error: 'Artist and song title are required.' });
  if (!INSTRUMENTS[instrument]) return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  if (commercial.error) return res.status(400).json({ error: commercial.error });

  Object.assign(listing, {
    artist,
    title,
    instrument,
    description,
    listingMode: commercial.listingMode,
    priceMcoins: commercial.priceMcoins,
    listenerRewardMcoins: commercial.listenerRewardMcoins,
    marketplaceFeeRate: policies.marketplaceFeePercent / 100,
    cover: INSTRUMENTS[instrument].cover,
    updatedAt: new Date().toISOString(),
  });
  await writeDb(req.db);
  res.json({ listing: publicListing(listing, req.db, req.user.id) });
});

app.post('/api/listings/:listingId/purchase', requireAuth, async (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  if (listing.sellerId === req.user.id) return res.status(400).json({ error: 'You already own this listing.' });
  const existing = req.db.purchases.find((purchase) => purchase.listingId === listing.id && purchase.buyerId === req.user.id);
  if (existing) return res.json({ purchase: existing, user: safeUser(req.user) });

  const mode = listingMode(listing);
  const seller = req.db.users.find((user) => user.id === listing.sellerId);
  const platform = req.db.users.find((user) => user.id === 'platform');

  if (mode === 'listener-reward') {
    const reward = listenerRewardStatus(listing, req.db);
    if (!reward.available) {
      return res.status(409).json({ error: 'This listener reward is paused, exhausted, or cannot currently be funded.' });
    }
    if (!seller) return res.status(409).json({ error: 'The composer account is unavailable.' });

    if (!hasUnlimitedMcoins(seller)) {
      seller.mcoins = Number((Number(seller.mcoins || 0) - reward.rewardMcoins).toFixed(2));
      seller.withdrawableMcoins = Math.min(Number(seller.withdrawableMcoins || 0), seller.mcoins);
    }
    req.user.mcoins = Number((Number(req.user.mcoins || 0) + reward.rewardMcoins).toFixed(2));
    req.user.withdrawableMcoins = Number((Number(req.user.withdrawableMcoins || 0) + reward.rewardMcoins).toFixed(2));
    listing.rewardPaidMcoins = Number((reward.paidMcoins + reward.rewardMcoins).toFixed(2));

    const purchase = {
      id: id('purchase'),
      listingId: listing.id,
      buyerId: req.user.id,
      sellerId: listing.sellerId,
      amount: 0,
      currency: 'MCOINS',
      amountMcoins: 0,
      grossMcoins: 0,
      buyerPaidMcoins: 0,
      listenerRewardMcoins: reward.rewardMcoins,
      paymentMethod: 'listener_reward',
      promotionDiscountMcoins: 0,
      platformFeeMcoins: 0,
      platformFeeRate: 0,
      sellerEarningsMcoins: 0,
      format: listing.format,
      instrument: listing.instrument,
      createdAt: new Date().toISOString(),
    };
    req.db.purchases.push(purchase);
    addLedger(req.db, seller.id, hasUnlimitedMcoins(seller) ? 0 : -reward.rewardMcoins, 'listener_reward_paid', `${listing.title}; rewarded ${req.user.name}`);
    addLedger(req.db, req.user.id, reward.rewardMcoins, 'listener_reward_received', `${listing.title}; paid by ${seller.name}`);
    await writeDb(req.db);
    return res.status(201).json({ purchase, user: safeUser(req.user) });
  }

  const promotionCode = String(req.body.promotionCode || '').trim();
  const requestedFriendId = String(req.body.friendId || '').trim();
  if (mode !== 'sale' && (promotionCode || requestedFriendId)) {
    return res.status(400).json({ error: 'Coupons and Friend IDs only apply to paid listings.' });
  }
  if (promotionCode && requestedFriendId) {
    return res.status(400).json({ error: 'Use either a music-sheet coupon or a Friend ID voucher, not both together.' });
  }
  const promotionResult = requestedFriendId
    ? friendPromotionForUse(req.db, requestedFriendId, req.user, listing.priceMcoins)
    : promotionForUse(
      req.db,
      promotionCode,
      req.user,
      ['marketplace_percent', 'marketplace_fixed'],
      listing.priceMcoins,
    );
  if (promotionResult.error) return res.status(400).json({ error: promotionResult.error });
  const promotion = promotionResult.promotion;
  const friendUser = promotionResult.friendUser || null;
  const discountMcoins = promotion
    ? Math.min(
      listing.priceMcoins,
      promotion.kind === 'marketplace_fixed'
        ? Number(Number(promotion.value).toFixed(2))
        : Number((listing.priceMcoins * promotion.value / 100).toFixed(2)),
    )
    : 0;
  const buyerPaidMcoins = Number((listing.priceMcoins - discountMcoins).toFixed(2));
  const administratorPurchase = hasUnlimitedMcoins(req.user);
  if (!administratorPurchase && req.user.mcoins < buyerPaidMcoins) {
    return res.status(402).json({ error: 'Not enough Mcoins.' });
  }

  const marketplaceFeeRate = Math.min(1, Math.max(0, Number(listing.marketplaceFeeRate ?? sitePolicies(req.db).marketplaceFeePercent / 100)));
  const platformFeeMcoins = Number((listing.priceMcoins * marketplaceFeeRate).toFixed(2));
  const sellerEarningsMcoins = Number((listing.priceMcoins - platformFeeMcoins).toFixed(2));

  if (!administratorPurchase) req.user.mcoins = Number((req.user.mcoins - buyerPaidMcoins).toFixed(2));
  if (seller) {
    seller.mcoins = Number((Number(seller.mcoins || 0) + sellerEarningsMcoins).toFixed(2));
    seller.withdrawableMcoins = Number((Number(seller.withdrawableMcoins || 0) + sellerEarningsMcoins).toFixed(2));
  }
  if (platform) {
    platform.mcoins = Number((Number(platform.mcoins || 0) + platformFeeMcoins - discountMcoins).toFixed(2));
  }

  const purchase = {
    id: id('purchase'),
    listingId: listing.id,
    buyerId: req.user.id,
    sellerId: listing.sellerId,
    amount: buyerPaidMcoins,
    currency: 'MCOINS',
    amountMcoins: buyerPaidMcoins,
    grossMcoins: listing.priceMcoins,
    buyerPaidMcoins,
    paymentMethod: administratorPurchase ? 'administrator_unlimited' : 'mcoins',
    promotionDiscountMcoins: discountMcoins,
    promotionId: promotion?.id || null,
    promotionCode: promotion?.code || null,
    friendUserId: friendUser?.id || null,
    friendId: friendUser?.friendId || null,
    platformFeeMcoins,
    platformFeeRate: marketplaceFeeRate,
    sellerEarningsMcoins,
    format: listing.format,
    instrument: listing.instrument,
    createdAt: new Date().toISOString(),
  };
  req.db.purchases.push(purchase);
  addLedger(
    req.db,
    req.user.id,
    administratorPurchase ? 0 : -buyerPaidMcoins,
    administratorPurchase ? 'admin_listing_purchase' : 'listing_purchase',
    `${listing.title} (${listing.format})${administratorPurchase ? '; unlimited administrator wallet' : ''}${promotion ? `; coupon ${promotion.code}: -${discountMcoins} Mcoins` : ''}`,
  );
  if (seller) addLedger(req.db, seller.id, sellerEarningsMcoins, 'listing_sale', `${listing.title}; ${Number((marketplaceFeeRate * 100).toFixed(2))}% platform fee: ${platformFeeMcoins} Mcoins`);
  if (platform) addLedger(req.db, platform.id, platformFeeMcoins - discountMcoins, 'marketplace_fee', promotion ? `${listing.title}; sponsored discount ${discountMcoins} Mcoins` : listing.title);
  if (promotion) {
    recordPromotionRedemption(req.db, promotion, req.user, {
      listingId: listing.id,
      discountMcoins,
      originalSpendMcoins: listing.priceMcoins,
      finalSpendMcoins: buyerPaidMcoins,
      friendUserId: friendUser?.id || null,
      friendId: friendUser?.friendId || null,
    });
  }
  await writeDb(req.db);
  res.status(201).json({ purchase, user: safeUser(req.user), promotion: promotion ? publicPromotion(promotion, req.db) : null });
});

app.get('/api/listings/:listingId/download', requireAuth, async (req, res) => {
  const listing = req.db.listings.find((item) => item.id === req.params.listingId);
  if (!listing) return res.status(404).json({ error: 'Listing not found.' });
  const allowed = listing.sellerId === req.user.id || req.db.purchases.some(
    (purchase) => purchase.listingId === listing.id && purchase.buyerId === req.user.id,
  );
  if (!allowed) return res.status(403).json({ error: 'Purchase this listing before downloading it.' });
  if (!listing.assetPath) {
    return res.status(409).json({ error: 'This demonstration listing does not include a downloadable asset.' });
  }
  return ARTIFACT_STORE.sendDownload(
    res,
    listing.assetPath,
    listing.filename || path.basename(listing.assetPath),
  );
});

app.get('/api/community/rooms', requireSubscriber, async (req, res) => {
  const rooms = req.db.communityRooms
    .filter((room) => canReadRoom(req.db, room, req.user.id))
    .map((room) => publicRoom(room, req.db, req.user.id, isAdministrator(req.user)))
    .sort((a, b) => {
      if (a.id === GLOBAL_ROOM_ID) return -1;
      if (b.id === GLOBAL_ROOM_ID) return 1;
      return String(b.lastMessageAt || '').localeCompare(String(a.lastMessageAt || ''));
    });
  return res.json({ rooms, access: 'subscriber' });
});

app.post('/api/community/rooms', requireSubscriber, async (req, res) => {
  const name = cleanCommunityText(req.body?.name, 60);
  const topic = cleanCommunityText(req.body?.topic, 180);
  const visibility = req.body?.visibility === 'private' ? 'private' : 'public';
  if (name.length < 2) return res.status(400).json({ error: 'Give the group a name with at least 2 characters.' });
  const ownedCount = req.db.communityRooms.filter((room) => room.ownerId === req.user.id).length;
  if (!isAdministrator(req.user) && ownedCount >= 12) {
    return res.status(400).json({ error: 'You can own up to 12 community groups.' });
  }
  const now = new Date().toISOString();
  const room = {
    id: id('community_room'),
    name,
    topic,
    visibility,
    ownerId: req.user.id,
    inviteCode: crypto.randomBytes(5).toString('hex').toUpperCase(),
    createdAt: now,
  };
  req.db.communityRooms.push(room);
  req.db.communityMemberships.push({
    id: id('community_member'),
    roomId: room.id,
    userId: req.user.id,
    role: 'owner',
    joinedAt: now,
  });
  await writeDb(req.db);
  return res.status(201).json({ room: publicRoom(room, req.db, req.user.id, isAdministrator(req.user)) });
});

app.post('/api/community/rooms/join', requireSubscriber, async (req, res) => {
  const inviteCode = cleanCommunityText(req.body?.inviteCode, 20).toUpperCase();
  const roomId = cleanCommunityText(req.body?.roomId, 100);
  const room = inviteCode
    ? req.db.communityRooms.find((candidate) => candidate.inviteCode === inviteCode)
    : req.db.communityRooms.find((candidate) => candidate.id === roomId);
  if (!room || room.id === GLOBAL_ROOM_ID) return res.status(404).json({ error: 'That community group was not found.' });
  if (room.visibility === 'private' && !inviteCode) {
    return res.status(403).json({ error: 'Enter this private group’s invite code.' });
  }
  if (!membershipFor(req.db, room.id, req.user.id)) {
    req.db.communityMemberships.push({
      id: id('community_member'),
      roomId: room.id,
      userId: req.user.id,
      role: 'member',
      joinedAt: new Date().toISOString(),
    });
    await writeDb(req.db);
  }
  return res.json({ room: publicRoom(room, req.db, req.user.id, isAdministrator(req.user)) });
});

app.delete('/api/community/rooms/:roomId/membership', requireSubscriber, async (req, res) => {
  const room = req.db.communityRooms.find((candidate) => candidate.id === req.params.roomId);
  if (!room || room.id === GLOBAL_ROOM_ID) return res.status(404).json({ error: 'That community group was not found.' });
  if (room.ownerId === req.user.id) return res.status(400).json({ error: 'The group owner can delete the group instead of leaving it.' });
  req.db.communityMemberships = req.db.communityMemberships.filter(
    (membership) => !(membership.roomId === room.id && membership.userId === req.user.id),
  );
  await writeDb(req.db);
  return res.status(204).end();
});

app.delete('/api/community/rooms/:roomId', requireSubscriber, async (req, res) => {
  const room = req.db.communityRooms.find((candidate) => candidate.id === req.params.roomId);
  if (!room || room.id === GLOBAL_ROOM_ID) return res.status(404).json({ error: 'That community group was not found.' });
  if (room.ownerId !== req.user.id && !isAdministrator(req.user)) {
    return res.status(403).json({ error: 'Only the group owner or an administrator can delete this group.' });
  }
  req.db.communityRooms = req.db.communityRooms.filter((candidate) => candidate.id !== room.id);
  req.db.communityMemberships = req.db.communityMemberships.filter((membership) => membership.roomId !== room.id);
  req.db.communityMessages = req.db.communityMessages.filter((message) => message.roomId !== room.id);
  await writeDb(req.db);
  return res.status(204).end();
});

app.get('/api/community/rooms/:roomId/messages', requireSubscriber, async (req, res) => {
  const room = req.db.communityRooms.find((candidate) => candidate.id === req.params.roomId);
  if (!room || !canReadRoom(req.db, room, req.user.id)) {
    return res.status(404).json({ error: 'That community group was not found.' });
  }
  const since = String(req.query.since || '');
  const messages = req.db.communityMessages
    .filter((message) => message.roomId === room.id && (!since || String(message.createdAt) > since))
    .sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)))
    .slice(-250)
    .map((message) => publicMessage(message, req.db, req.user.id, room, isAdministrator(req.user)));
  return res.json({
    room: publicRoom(room, req.db, req.user.id, isAdministrator(req.user)),
    messages,
  });
});

app.post('/api/community/rooms/:roomId/messages', requireSubscriber, async (req, res) => {
  const room = req.db.communityRooms.find((candidate) => candidate.id === req.params.roomId);
  if (!room || !canReadRoom(req.db, room, req.user.id)) {
    return res.status(404).json({ error: 'That community group was not found.' });
  }
  if (!canWriteRoom(req.db, room, req.user.id)) {
    return res.status(403).json({ error: 'Join this group before sending a message.' });
  }
  if (!requestIntervalAllowed(COMMUNITY_REQUEST_TIMES, req.user.id, 700)) {
    res.set('Retry-After', '1');
    return res.status(429).json({ error: 'Please wait a moment before sending another message.' });
  }
  const text = cleanCommunityText(req.body?.text);
  if (!text) return res.status(400).json({ error: 'Write a message first.' });
  const message = {
    id: id('community_message'),
    roomId: room.id,
    userId: req.user.id,
    text,
    createdAt: new Date().toISOString(),
  };
  req.db.communityMessages.push(message);
  trimRoomMessages(req.db, room.id, room.id === GLOBAL_ROOM_ID ? 1500 : 750);
  await writeDb(req.db);
  return res.status(201).json({ message: publicMessage(message, req.db, req.user.id, room, isAdministrator(req.user)) });
});

app.delete('/api/community/rooms/:roomId/messages/:messageId', requireSubscriber, async (req, res) => {
  const room = req.db.communityRooms.find((candidate) => candidate.id === req.params.roomId);
  const message = req.db.communityMessages.find(
    (candidate) => candidate.id === req.params.messageId && candidate.roomId === req.params.roomId,
  );
  if (!room || !message) return res.status(404).json({ error: 'That message was not found.' });
  const membership = membershipFor(req.db, room.id, req.user.id);
  const canDelete = isAdministrator(req.user)
    || message.userId === req.user.id
    || room.ownerId === req.user.id
    || membership?.role === 'moderator';
  if (!canDelete) return res.status(403).json({ error: 'You cannot remove that message.' });
  req.db.communityMessages = req.db.communityMessages.filter((candidate) => candidate.id !== message.id);
  await writeDb(req.db);
  return res.status(204).end();
});

app.post('/api/community/messages/:messageId/report', requireSubscriber, async (req, res) => {
  const message = req.db.communityMessages.find((candidate) => candidate.id === req.params.messageId);
  if (!message) return res.status(404).json({ error: 'That message was not found.' });
  const room = req.db.communityRooms.find((candidate) => candidate.id === message.roomId);
  if (!room || !canReadRoom(req.db, room, req.user.id)) return res.status(404).json({ error: 'That message was not found.' });
  if (message.userId === req.user.id) return res.status(400).json({ error: 'You do not need to report your own message.' });
  const existing = req.db.communityReports.find(
    (report) => report.messageId === message.id && report.reporterUserId === req.user.id,
  );
  if (!existing) {
    req.db.communityReports.push({
      id: id('community_report'),
      messageId: message.id,
      roomId: room.id,
      reporterUserId: req.user.id,
      reason: cleanCommunityText(req.body?.reason, 300) || 'Community safety review requested.',
      status: 'open',
      createdAt: new Date().toISOString(),
    });
    await writeDb(req.db);
  }
  return res.status(201).json({ message: 'Report sent privately to the moderation queue.' });
});

app.get('/api/bands', requireMusician, async (req, res) => {
  const visible = req.db.bands
    .filter((band) => band.accessMode !== 'invite')
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((band) => safeBand(band, req.db));
  res.json({ bands: visible });
});

app.get('/api/bands/me', requireMusician, async (req, res) => {
  const bands = req.db.bands
    .filter((band) => band.hostId === req.user.id || bandMembership(req.db, band.id, req.user.id))
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((band) => safeBand(band, req.db, req.user.id));
  res.json({ bands });
});

app.post('/api/bands', requireMusician, async (req, res) => {
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
    bans: [],
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
  await writeDb(req.db);
  res.status(201).json({ band: safeBand(band, req.db, req.user.id), user: safeUser(req.user) });
});

app.post('/api/bands/join-by-code', requireMusician, async (req, res) => {
  const code = String(req.body.code || '').trim().toUpperCase();
  const band = req.db.bands.find((candidate) => candidate.inviteCode === code);
  if (!band) return res.status(404).json({ error: 'That friend invite code is not valid.' });
  if (bandBan(band, req.user.id)) return res.status(403).json({ error: 'The band creator has banned this account.' });
  if (!bandMembership(req.db, band.id, req.user.id)) {
    req.db.bandMemberships.push({
      id: id('band_member'),
      bandId: band.id,
      userId: req.user.id,
      role: 'member',
      joinedAt: new Date().toISOString(),
    });
    await writeDb(req.db);
  }
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.post('/api/bands/:bandId/join', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (bandBan(band, req.user.id)) return res.status(403).json({ error: 'The band creator has banned this account.' });
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
    const administratorPayment = hasUnlimitedMcoins(req.user);
    if (!administratorPayment && req.user.mcoins < fee) {
      return res.status(402).json({ error: 'Not enough Mcoins to join this band.' });
    }
    const host = req.db.users.find((user) => user.id === band.hostId);
    if (!administratorPayment) req.user.mcoins -= fee;
    if (host && host.id !== req.user.id) {
      host.mcoins += fee;
      host.withdrawableMcoins = Number(host.withdrawableMcoins || 0) + fee;
      addLedger(req.db, host.id, fee, 'band_entry_received', `${req.user.name} joined ${band.name}`);
    }
    addLedger(
      req.db,
      req.user.id,
      administratorPayment ? 0 : -fee,
      administratorPayment ? 'admin_band_entry' : 'band_entry_paid',
      administratorPayment ? `${band.name}; unlimited administrator wallet` : band.name,
    );
  }
  req.db.bandMemberships.push({
    id: id('band_member'),
    bandId: band.id,
    userId: req.user.id,
    role: 'member',
    joinedAt: new Date().toISOString(),
  });
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id), user: safeUser(req.user) });
});

app.get('/api/bands/:bandId/chat', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (!bandMembership(req.db, band.id, req.user.id) && band.hostId !== req.user.id) {
    return res.status(403).json({ error: 'Join the band before opening its chat.' });
  }
  const messages = req.db.bandMessages
    .filter((message) => message.bandId === band.id)
    .sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)))
    .slice(-150)
    .map((message) => safeBandMessage(message, req.db));
  res.json({ messages });
});

app.post('/api/bands/:bandId/chat', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (!bandMembership(req.db, band.id, req.user.id) && band.hostId !== req.user.id) {
    return res.status(403).json({ error: 'Join the band before sending a message.' });
  }
  const text = String(req.body.text || '').trim().slice(0, 1000);
  if (!text) return res.status(400).json({ error: 'Write a message first.' });
  const recentCount = req.db.bandMessages.filter((message) => (
    message.bandId === band.id
    && message.userId === req.user.id
    && Date.now() - new Date(message.createdAt).getTime() < 60 * 1000
  )).length;
  if (recentCount >= 20) return res.status(429).json({ error: 'Slow down for a moment before sending more messages.' });
  const message = {
    id: id('band_message'),
    bandId: band.id,
    userId: req.user.id,
    text,
    createdAt: new Date().toISOString(),
  };
  req.db.bandMessages.push(message);
  req.db.bandMessages = req.db.bandMessages.slice(-5000);
  await writeDb(req.db);
  res.status(201).json({ message: safeBandMessage(message, req.db) });
});

app.delete('/api/bands/:bandId/members/:userId', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (band.hostId !== req.user.id) return res.status(403).json({ error: 'Only the band creator can remove members.' });
  if (req.params.userId === band.hostId) return res.status(400).json({ error: 'The band creator cannot be removed.' });
  const membership = bandMembership(req.db, band.id, req.params.userId);
  if (!membership) return res.status(404).json({ error: 'That account is not in this band.' });
  req.db.bandMemberships = req.db.bandMemberships.filter((item) => item.id !== membership.id);
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.post('/api/bands/:bandId/bans', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (band.hostId !== req.user.id) return res.status(403).json({ error: 'Only the band creator can ban accounts.' });
  const userId = String(req.body.userId || '');
  if (userId === band.hostId) return res.status(400).json({ error: 'The band creator cannot be banned.' });
  if (!req.db.users.some((user) => user.id === userId)) return res.status(404).json({ error: 'Account not found.' });
  if (!Array.isArray(band.bans)) band.bans = [];
  if (!bandBan(band, userId)) {
    band.bans.push({ userId, bannedBy: req.user.id, bannedAt: new Date().toISOString() });
  }
  req.db.bandMemberships = req.db.bandMemberships.filter((item) => !(item.bandId === band.id && item.userId === userId));
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.delete('/api/bands/:bandId/bans/:userId', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  if (band.hostId !== req.user.id) return res.status(403).json({ error: 'Only the band creator can unban accounts.' });
  band.bans = (Array.isArray(band.bans) ? band.bans : []).filter((ban) => ban.userId !== req.params.userId);
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.put('/api/bands/:bandId/general-score', requireMusician, async (req, res) => {
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
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.post('/api/bands/:bandId/instruments', requireMusician, async (req, res) => {
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
  await writeDb(req.db);
  res.status(201).json({ band: safeBand(band, req.db, req.user.id), part });
});

app.put('/api/bands/:bandId/instruments/:partId', requireMusician, async (req, res) => {
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
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id), part });
});

app.delete('/api/bands/:bandId/instruments/:partId', requireMusician, async (req, res) => {
  const band = req.db.bands.find((candidate) => candidate.id === req.params.bandId);
  if (!band) return res.status(404).json({ error: 'Band not found.' });
  const part = (band.instruments || []).find((candidate) => candidate.id === req.params.partId);
  if (!part) return res.status(404).json({ error: 'Instrument part not found.' });
  if (band.hostId !== req.user.id && part.addedBy !== req.user.id) {
    return res.status(403).json({ error: 'Only the host or the person who added this part can remove it.' });
  }
  band.instruments = band.instruments.filter((candidate) => candidate.id !== part.id);
  await writeDb(req.db);
  res.json({ band: safeBand(band, req.db, req.user.id) });
});

app.get('/api/messages/threads', requireAuth, async (req, res) => {
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

app.get('/api/messages/:otherUserId', requireAuth, async (req, res) => {
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

app.post('/api/messages', requireAuth, async (req, res) => {
  const toUserId = String(req.body.toUserId || '');
  const text = String(req.body.text || '').trim().slice(0, 2000);
  if (!text) return res.status(400).json({ error: 'Message cannot be empty.' });
  if (toUserId === req.user.id) return res.status(400).json({ error: 'You cannot message yourself.' });
  if (!req.db.users.some((user) => user.id === toUserId)) return res.status(404).json({ error: 'Recipient not found.' });
  const message = { id: id('message'), fromUserId: req.user.id, toUserId, text, createdAt: new Date().toISOString() };
  req.db.messages.push(message);
  await writeDb(req.db);
  res.status(201).json({ message });
});

app.get('/api/wallet', requireAuth, async (req, res) => {
  const ledger = req.db.ledger.filter((entry) => entry.userId === req.user.id).slice(-100).reverse();
  const withdrawals = req.db.withdrawals.filter((item) => item.userId === req.user.id).slice(-20).reverse();
  const policies = sitePolicies(req.db);
  res.json({ user: safeUser(req.user), ledger, withdrawals, withdrawalFeeRate: policies.withdrawalFeePercent / 100, policies });
});

app.get('/api/admin/customer-purchases', requireAuth, requireAdmin, async (req, res) => {
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

app.get('/api/admin/community/reports', requireAuth, requireAdmin, async (req, res) => {
  const reports = req.db.communityReports
    .slice()
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((report) => {
      const message = req.db.communityMessages.find((candidate) => candidate.id === report.messageId);
      const room = req.db.communityRooms.find((candidate) => candidate.id === report.roomId);
      const reporter = req.db.users.find((candidate) => candidate.id === report.reporterUserId);
      const author = message && req.db.users.find((candidate) => candidate.id === message.userId);
      return {
        id: report.id,
        status: report.status,
        reason: report.reason,
        createdAt: report.createdAt,
        resolvedAt: report.resolvedAt || null,
        room: { id: report.roomId, name: room?.name || 'Deleted group' },
        message: message ? { id: message.id, text: message.text, author: author?.name || 'Former member' } : null,
        reporter: reporter?.name || 'Former member',
      };
    });
  return res.json({ reports, openCount: reports.filter((report) => report.status === 'open').length });
});

app.patch('/api/admin/community/reports/:reportId', requireAuth, requireAdmin, async (req, res) => {
  const report = req.db.communityReports.find((candidate) => candidate.id === req.params.reportId);
  if (!report) return res.status(404).json({ error: 'Community report not found.' });
  const status = ['resolved', 'dismissed'].includes(req.body?.status) ? req.body.status : 'resolved';
  if (req.body?.removeMessage) {
    req.db.communityMessages = req.db.communityMessages.filter((message) => message.id !== report.messageId);
  }
  report.status = status;
  report.resolvedAt = new Date().toISOString();
  report.resolvedBy = req.user.id;
  await writeDb(req.db);
  return res.json({ report: { id: report.id, status: report.status, resolvedAt: report.resolvedAt } });
});

app.get('/api/admin/users', requireAuth, requireAdmin, async (req, res) => {
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
    const administrator = isAdministrator(user);
    const administratorGrant = activeAdminSubscriptionGrant(user);
    return {
      userId: user.id,
      friendId: user.friendId || '',
      name: user.name,
      email: user.email,
      phone: user.phone || '',
      mcoins,
      unlimitedMcoins: administrator,
      admin: administrator,
      mcoinUsdEquivalent: Number((mcoins / MCOINS_PER_USD).toFixed(2)),
      usdSpent: Number(usdSpent.toFixed(2)),
      marketplaceSpentMcoins,
      marketplaceSpentUsdEquivalent: Number((marketplaceSpentMcoins / MCOINS_PER_USD).toFixed(2)),
      purchaseCount: completedOrders.length + marketplacePurchases.length,
      subscriptionTier: activeSubscriptionTier(user),
      subscriptionInterval: effectiveSubscriptionInterval(user),
      subscriptionStartedAt: effectiveSubscriptionStartedAt(user),
      proStatus: administratorGrant ? 'ACTIVE' : (user.proStatus || (user.pro ? 'ACTIVE' : 'INACTIVE')),
      adminSubscriptionGrant: user.adminSubscriptionGrant ? {
        tier: String(user.adminSubscriptionGrant.tier || '').toLowerCase(),
        interval: String(user.adminSubscriptionGrant.interval || '').toUpperCase(),
        startedAt: user.adminSubscriptionGrant.startedAt || null,
        expiresAt: user.adminSubscriptionGrant.expiresAt || null,
        active: Boolean(administratorGrant),
      } : null,
      createdAt: user.createdAt,
      lastLoginAt: user.lastLoginAt || null,
      loginCount: Number(user.loginCount || 0),
      passwordResetAt: user.passwordResetAt || null,
    };
  }).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));

  const platformFeesMcoins = req.db.purchases.reduce(
    (total, purchase) => total + Number(purchase.platformFeeMcoins || 0),
    0,
  );
  const totalUsdRevenue = rows.reduce((total, row) => total + row.usdSpent, 0);
  res.json({
    rows,
    configuration: {
      paypalEnvironment: PAYPAL_ENV,
      paypalCheckoutConfigured: Boolean(process.env.PAYPAL_CLIENT_ID && process.env.PAYPAL_SECRET_KEY),
      paypalProPlanConfigured: Boolean(process.env.PAYPAL_PRO_PLAN_ID),
      paypalSubscriptionPlansConfigured: {
        chillMonthly: Boolean(process.env.PAYPAL_CHILL_MONTHLY_PLAN_ID),
        chillYearly: Boolean(process.env.PAYPAL_CHILL_YEARLY_PLAN_ID),
        musicianMonthly: Boolean(process.env.PAYPAL_MUSICIAN_MONTHLY_PLAN_ID || process.env.PAYPAL_PRO_PLAN_ID),
        musicianYearly: Boolean(process.env.PAYPAL_MUSICIAN_YEARLY_PLAN_ID),
        monthlyUpgrade: Boolean(process.env.PAYPAL_CHILL_TO_MUSICIAN_MONTHLY_PLAN_ID),
        yearlyUpgrade: Boolean(process.env.PAYPAL_CHILL_TO_MUSICIAN_YEARLY_PLAN_ID),
      },
      paypalWebhookConfigured: Boolean(process.env.PAYPAL_WEBHOOK_ID),
      persistence: 'Atomic JSON file',
      databasePath: path.relative(__dirname, DB_PATH).replace(/\\/g, '/'),
      sessionStorage: 'SHA-256 token hashes with 30-day expiry',
    },
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

app.post('/api/admin/users/:userId/mcoins', requireAuth, requireAdmin, async (req, res) => {
  const target = req.db.users.find((candidate) => candidate.id === req.params.userId && candidate.id !== 'platform');
  if (!target) return res.status(404).json({ error: 'User account not found.' });
  if (isAdministrator(target)) {
    return res.status(400).json({ error: 'Administrator accounts already have unlimited Mcoins.' });
  }
  const requestedAmount = Number(req.body.amountMcoins);
  const amountMcoins = Number.isFinite(requestedAmount)
    ? Number((Math.floor(requestedAmount * 100) / 100).toFixed(2))
    : Number.NaN;
  if (!Number.isFinite(amountMcoins) || amountMcoins <= 0 || amountMcoins > 1000000) {
    return res.status(400).json({ error: 'Enter an Mcoin gift between 0.01 and 1,000,000.' });
  }
  target.mcoins = Number((Number(target.mcoins || 0) + amountMcoins).toFixed(2));
  addLedger(
    req.db,
    target.id,
    amountMcoins,
    'admin_mcoin_grant',
    `Administrator gift from ${req.user.email}`,
  );
  await writeDb(req.db);
  return res.json({
    user: safeUser(target),
    message: `${amountMcoins.toLocaleString()} Mcoins were added to ${target.name}.`,
  });
});

app.post('/api/admin/users/:userId/subscription', requireAuth, requireAdmin, async (req, res) => {
  const target = req.db.users.find((candidate) => candidate.id === req.params.userId && candidate.id !== 'platform');
  if (!target) return res.status(404).json({ error: 'User account not found.' });
  if (isAdministrator(target)) {
    return res.status(400).json({ error: 'Administrator accounts already have unlimited platform access.' });
  }
  const tier = String(req.body.tier || '').trim().toLowerCase();
  const interval = String(req.body.interval || '').trim().toUpperCase();
  if (!['chill', 'musician'].includes(tier)) {
    return res.status(400).json({ error: 'Choose Chill or Musician.' });
  }
  if (!['MONTH', 'YEAR'].includes(interval)) {
    return res.status(400).json({ error: 'Choose a monthly or yearly access period.' });
  }

  const now = new Date();
  const existing = activeAdminSubscriptionGrant(target, now);
  const extending = Boolean(existing && existing.tier === tier && existing.interval === interval);
  const periodBase = extending ? new Date(existing.expiresAt) : now;
  const expiresAt = utcMonthAnniversary(periodBase, interval === 'YEAR' ? 12 : 1);
  target.adminSubscriptionGrant = {
    tier,
    interval,
    startedAt: extending ? existing.startedAt : now.toISOString(),
    expiresAt: expiresAt.toISOString(),
    grantedAt: now.toISOString(),
    grantedBy: req.user.id,
  };
  if (!extending) {
    target.translationUsage = {
      period: translationUsageWindow(target, now).key,
      includedUsed: 0,
    };
  }
  addLedger(
    req.db,
    target.id,
    0,
    extending ? 'admin_subscription_renewed' : 'admin_subscription_granted',
    `${tier} ${interval.toLowerCase()} access through ${expiresAt.toISOString()} by ${req.user.email}`,
  );
  await writeDb(req.db);
  return res.json({
    user: safeUser(target),
    extended: extending,
    message: `${tier === 'musician' ? 'Musician' : 'Chill'} access for ${target.name} is active through ${expiresAt.toLocaleDateString('en-SG')}.`,
  });
});

app.delete('/api/admin/users/:userId/subscription', requireAuth, requireAdmin, async (req, res) => {
  const target = req.db.users.find((candidate) => candidate.id === req.params.userId && candidate.id !== 'platform');
  if (!target) return res.status(404).json({ error: 'User account not found.' });
  if (!target.adminSubscriptionGrant) {
    return res.status(404).json({ error: 'This account has no administrator-granted subscription.' });
  }
  delete target.adminSubscriptionGrant;
  addLedger(
    req.db,
    target.id,
    0,
    'admin_subscription_grant_removed',
    `Manual access removed by ${req.user.email}; PayPal and institution access were not changed`,
  );
  await writeDb(req.db);
  return res.json({
    user: safeUser(target),
    message: `Administrator-granted access was removed from ${target.name}. PayPal or institution access, if any, remains unchanged.`,
  });
});

app.get('/api/admin/policies', requireAuth, requireAdmin, async (req, res) => {
  res.json({ policies: sitePolicies(req.db) });
});

app.get('/api/admin/withdrawals', requireAuth, requireAdmin, async (req, res) => {
  const withdrawals = req.db.withdrawals
    .slice()
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))
    .map((withdrawal) => {
      const account = req.db.users.find((user) => user.id === withdrawal.userId);
      return {
        ...withdrawal,
        account: account ? {
          userId: account.id,
          name: account.name,
          email: account.email,
          phone: account.phone || '',
        } : null,
      };
    });
  const pending = withdrawals.filter((item) => String(item.status || '').toLowerCase().startsWith('pending'));
  res.json({
    withdrawals,
    summary: {
      pendingCount: pending.length,
      pendingGrossMcoins: Number(pending.reduce((total, item) => total + Number(item.amountMcoins || 0), 0).toFixed(2)),
      pendingNetMcoins: Number(pending.reduce((total, item) => total + Number(item.netMcoins || 0), 0).toFixed(2)),
    },
  });
});

app.patch('/api/admin/withdrawals/:withdrawalId', requireAuth, requireAdmin, async (req, res) => {
  const withdrawal = req.db.withdrawals.find((item) => item.id === req.params.withdrawalId);
  if (!withdrawal) return res.status(404).json({ error: 'Withdrawal request not found.' });
  if (!String(withdrawal.status || '').toLowerCase().startsWith('pending')) {
    return res.status(409).json({ error: 'Only a pending withdrawal can be completed or rejected.' });
  }
  const nextStatus = String(req.body.status || '').trim().toLowerCase();
  if (!['paid', 'rejected'].includes(nextStatus)) {
    return res.status(400).json({ error: 'Choose paid or rejected.' });
  }

  const account = req.db.users.find((user) => user.id === withdrawal.userId);
  if (nextStatus === 'rejected' && account) {
    const amountMcoins = Number(withdrawal.amountMcoins || 0);
    const feeMcoins = Number(withdrawal.feeMcoins || 0);
    account.mcoins = Number((Number(account.mcoins || 0) + amountMcoins).toFixed(2));
    account.withdrawableMcoins = Math.min(
      account.mcoins,
      Number((Number(account.withdrawableMcoins || 0) + amountMcoins).toFixed(2)),
    );
    const platform = req.db.users.find((user) => user.id === 'platform');
    if (platform) {
      platform.mcoins = Number((Number(platform.mcoins || 0) - feeMcoins).toFixed(2));
      addLedger(req.db, platform.id, -feeMcoins, 'cashout_fee_reversed', `${account.name} rejected cash-out`);
    }
    addLedger(req.db, account.id, amountMcoins, 'withdrawal_rejected_refund', `Withdrawal ${withdrawal.id} rejected by administrator`);
  }

  withdrawal.status = nextStatus;
  withdrawal.reviewedAt = new Date().toISOString();
  withdrawal.reviewedBy = req.user.id;
  await writeDb(req.db);
  res.json({ withdrawal, message: nextStatus === 'paid' ? 'Withdrawal marked as paid.' : 'Withdrawal rejected and refunded.' });
});

app.put('/api/admin/policies', requireAuth, requireAdmin, async (req, res) => {
  const current = sitePolicies(req.db);
  const next = {
    registrationEnabled: req.body.registrationEnabled !== false,
    minimumSignupAge: clampInteger(req.body.minimumSignupAge, 0, 120, current.minimumSignupAge),
    minimumPasswordLength: clampInteger(req.body.minimumPasswordLength, 1, 256, current.minimumPasswordLength),
    minimumMarketplacePriceMcoins: clampDecimal(req.body.minimumMarketplacePriceMcoins, 0, 1000000000, current.minimumMarketplacePriceMcoins),
    maximumMarketplacePriceMcoins: clampDecimal(req.body.maximumMarketplacePriceMcoins, 0, 1000000000, current.maximumMarketplacePriceMcoins),
    marketplaceFeePercent: clampDecimal(req.body.marketplaceFeePercent, 0, 100, current.marketplaceFeePercent),
    teacherDirectoryEnabled: typeof req.body.teacherDirectoryEnabled === 'boolean'
      ? req.body.teacherDirectoryEnabled
      : current.teacherDirectoryEnabled,
    teacherApplicationsEnabled: typeof req.body.teacherApplicationsEnabled === 'boolean'
      ? req.body.teacherApplicationsEnabled
      : current.teacherApplicationsEnabled,
    teacherReviewsEnabled: typeof req.body.teacherReviewsEnabled === 'boolean'
      ? req.body.teacherReviewsEnabled
      : current.teacherReviewsEnabled,
    minimumTeacherHourlyRateMcoins: clampDecimal(
      req.body.minimumTeacherHourlyRateMcoins,
      0,
      1000000000,
      current.minimumTeacherHourlyRateMcoins,
    ),
    maximumTeacherHourlyRateMcoins: clampDecimal(
      req.body.maximumTeacherHourlyRateMcoins,
      0,
      1000000000,
      current.maximumTeacherHourlyRateMcoins,
    ),
    teacherMarketplaceFeePercent: clampDecimal(
      req.body.teacherMarketplaceFeePercent,
      0,
      100,
      current.teacherMarketplaceFeePercent,
    ),
    teacherMarketplaceNotice: String(
      req.body.teacherMarketplaceNotice ?? current.teacherMarketplaceNotice ?? '',
    ).trim().slice(0, 600),
    listenerRewardsEnabled: req.body.listenerRewardsEnabled !== false,
    maximumListenerRewardMcoins: clampDecimal(req.body.maximumListenerRewardMcoins, 0, 1000000000, current.maximumListenerRewardMcoins),
    maximumRewardOutflowPerListingMcoins: clampDecimal(req.body.maximumRewardOutflowPerListingMcoins, 0, 1000000000, current.maximumRewardOutflowPerListingMcoins),
    minimumWithdrawalMcoins: clampDecimal(req.body.minimumWithdrawalMcoins, 0, 1000000000, current.minimumWithdrawalMcoins),
    maximumWithdrawalMcoins: clampDecimal(req.body.maximumWithdrawalMcoins, 0, 1000000000, current.maximumWithdrawalMcoins),
    dailyWithdrawalLimitMcoins: clampDecimal(req.body.dailyWithdrawalLimitMcoins, 0, 1000000000, current.dailyWithdrawalLimitMcoins),
    maximumPendingWithdrawalOutflowMcoins: clampDecimal(req.body.maximumPendingWithdrawalOutflowMcoins, 0, 1000000000, current.maximumPendingWithdrawalOutflowMcoins),
    withdrawalFeePercent: clampDecimal(req.body.withdrawalFeePercent, 0, 100, current.withdrawalFeePercent),
    minimumWithdrawal20MigrationApplied: true,
    welcomeMcoins: clampDecimal(req.body.welcomeMcoins, 0, 1000000000, current.welcomeMcoins),
    virtualLessonPricePer30MinutesMcoins: clampDecimal(
      req.body.virtualLessonPricePer30MinutesMcoins,
      0,
      1000000000,
      current.virtualLessonPricePer30MinutesMcoins,
    ),
    policyNotice: String(req.body.policyNotice || '').trim().slice(0, 1000),
    termsUrl: String(req.body.termsUrl || '').trim().slice(0, 500),
    privacyUrl: String(req.body.privacyUrl || '').trim().slice(0, 500),
    supportEmail: String(req.body.supportEmail || '').trim().toLowerCase().slice(0, 254),
    supportPhone: String(req.body.supportPhone || '').trim().slice(0, 40),
    updatedAt: new Date().toISOString(),
    updatedBy: req.user.id,
  };
  if (next.maximumMarketplacePriceMcoins > 0 && next.maximumMarketplacePriceMcoins < next.minimumMarketplacePriceMcoins) {
    return res.status(400).json({ error: 'Maximum listing price must be 0 (unlimited) or at least the minimum listing price.' });
  }
  if (next.maximumTeacherHourlyRateMcoins > 0
    && next.maximumTeacherHourlyRateMcoins < next.minimumTeacherHourlyRateMcoins) {
    return res.status(400).json({
      error: 'Maximum teacher hourly rate must be 0 (unlimited) or at least the minimum teacher hourly rate.',
    });
  }
  if (next.maximumWithdrawalMcoins > 0 && next.maximumWithdrawalMcoins < next.minimumWithdrawalMcoins) {
    return res.status(400).json({ error: 'Maximum withdrawal must be 0 (unlimited) or at least the minimum withdrawal.' });
  }
  if (next.supportEmail && !/^\S+@\S+\.\S+$/.test(next.supportEmail)) {
    return res.status(400).json({ error: 'Enter a valid support email or leave it blank.' });
  }
  if (next.supportPhone) {
    const phoneDigits = next.supportPhone.replace(/\D/g, '');
    if (!/^\+?[\d() .-]+$/.test(next.supportPhone) || phoneDigits.length < 7 || phoneDigits.length > 18) {
      return res.status(400).json({ error: 'Enter a valid helpline number or leave it blank.' });
    }
  }
  for (const [label, value] of [['Terms', next.termsUrl], ['Privacy', next.privacyUrl]]) {
    if (value && !/^https?:\/\//i.test(value)) {
      return res.status(400).json({ error: `${label} URL must begin with http:// or https://.` });
    }
  }
  req.db.settings = next;
  await writeDb(req.db);
  res.json({ policies: sitePolicies(req.db), message: 'Rules and policies saved.' });
});

app.get('/api/admin/promotions', requireAuth, requireAdmin, async (req, res) => {
  const promotions = req.db.promotions
    .slice()
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((promotion) => publicPromotion(promotion, req.db));
  res.json({ promotions });
});

app.post('/api/admin/promotions', requireAuth, requireAdmin, async (req, res) => {
  const code = normalizePromotionCode(req.body.code);
  const name = String(req.body.name || '').trim().slice(0, 100);
  const kind = String(req.body.kind || '');
  const fixedMcoinDiscount = kind === 'marketplace_fixed';
  const rawValue = Number(req.body.value);
  const value = fixedMcoinDiscount
    ? clampDecimal(rawValue, 0.01, 1000000000, 0.01)
    : clampInteger(rawValue, 1, 100, 1);
  const startsAtDate = req.body.startsAt ? new Date(req.body.startsAt) : null;
  const expiresAtDate = req.body.expiresAt ? new Date(req.body.expiresAt) : null;
  if (code.length < 3) return res.status(400).json({ error: 'Promotion code must contain at least 3 letters or numbers.' });
  if (!name) return res.status(400).json({ error: 'Give the promotion a short name.' });
  if (!['marketplace_percent', 'marketplace_fixed', 'friend_id_percent', 'subscription_percent'].includes(kind)) {
    return res.status(400).json({ error: 'Choose a supported percentage or fixed-Mcoin discount.' });
  }
  if (!Number.isFinite(rawValue) || rawValue <= 0 || (!fixedMcoinDiscount && rawValue > 100)) {
    return res.status(400).json({ error: fixedMcoinDiscount
      ? 'Fixed discounts must be greater than 0 Mcoins.'
      : 'Percentage discounts must be between 1% and 100%.' });
  }
  if (req.db.promotions.some((promotion) => promotion.code === code)) {
    return res.status(409).json({ error: 'That promotion code already exists.' });
  }
  if ((startsAtDate && Number.isNaN(startsAtDate.getTime()))
    || (expiresAtDate && Number.isNaN(expiresAtDate.getTime()))) {
    return res.status(400).json({ error: 'Enter valid promotion dates.' });
  }
  if (startsAtDate && expiresAtDate && expiresAtDate <= startsAtDate) {
    return res.status(400).json({ error: 'Expiry must be later than the start date.' });
  }
  const startsAt = startsAtDate ? startsAtDate.toISOString() : null;
  const expiresAt = expiresAtDate ? expiresAtDate.toISOString() : null;
  const promotion = {
    id: id('promotion'),
    code,
    name,
    kind,
    value,
    minimumSpendMcoins: clampInteger(req.body.minimumSpendMcoins, 0, 100000, 0),
    minimumAccountAgeDays: clampInteger(req.body.minimumAccountAgeDays, 0, 3650, 0),
    maxRedemptions: clampInteger(req.body.maxRedemptions, 0, 1000000, 0),
    perUserLimit: clampInteger(req.body.perUserLimit, 0, 100, 1),
    startsAt,
    expiresAt,
    active: true,
    createdBy: req.user.id,
    createdAt: new Date().toISOString(),
  };
  if (kind === 'friend_id_percent') {
    req.db.promotions
      .filter((item) => item.kind === 'friend_id_percent' && item.active !== false)
      .forEach((item) => {
        item.active = false;
        item.updatedAt = promotion.createdAt;
        item.updatedBy = req.user.id;
      });
  }
  req.db.promotions.push(promotion);
  await writeDb(req.db);
  res.status(201).json({ promotion: publicPromotion(promotion, req.db), message: `${promotion.code} is ready to use.` });
});

app.patch('/api/admin/promotions/:promotionId', requireAuth, requireAdmin, async (req, res) => {
  const promotion = req.db.promotions.find((item) => item.id === req.params.promotionId);
  if (!promotion) return res.status(404).json({ error: 'Promotion not found.' });
  if (promotion.retired && req.body.active === true) {
    return res.status(400).json({ error: 'This legacy promotion is permanently retired.' });
  }
  if (req.body.active !== undefined) promotion.active = Boolean(req.body.active);
  promotion.updatedAt = new Date().toISOString();
  promotion.updatedBy = req.user.id;
  await writeDb(req.db);
  res.json({
    promotion: publicPromotion(promotion, req.db),
    message: `${promotion.code} is now ${promotion.active ? 'active' : 'paused'}.`,
  });
});

app.post('/api/promotions/redeem', requireAuth, async (req, res) => {
  res.status(410).json({
    error: 'Mcoin-credit vouchers are retired. Promotion codes now provide percentage discounts only.',
  });
});

app.post('/api/admin/users/:userId/reset-password', requireAuth, requireAdmin, async (req, res) => {
  const user = req.db.users.find((candidate) => candidate.id === req.params.userId && candidate.id !== 'platform');
  if (!user) return res.status(404).json({ error: 'User account not found.' });
  if (user.id === req.user.id) {
    return res.status(400).json({ error: 'Use your own account password-change flow instead of an administrator reset.' });
  }
  const minimumLength = sitePolicies(req.db).minimumPasswordLength;
  const generatedPassword = `PM-${crypto.randomBytes(Math.ceil(minimumLength * 0.8) + 4).toString('base64url')}`
    .slice(0, Math.max(minimumLength, 15));
  const password = String(req.body.password || generatedPassword);
  if (password.length < minimumLength) {
    return res.status(400).json({ error: `The temporary password must contain at least ${minimumLength} characters.` });
  }
  const { salt, hash } = hashPassword(password);
  user.passwordHash = hash;
  user.passwordSalt = salt;
  user.mustChangePassword = true;
  user.passwordResetAt = new Date().toISOString();
  user.passwordResetBy = req.user.id;
  req.db.passwordResetEvents.push({
    id: id('password_reset'),
    userId: user.id,
    adminUserId: req.user.id,
    createdAt: user.passwordResetAt,
  });
  req.db.sessions = req.db.sessions.filter((session) => session.userId !== user.id);
  await writeDb(req.db);
  res.json({
    userId: user.id,
    temporaryPassword: password,
    message: `Temporary password issued for ${user.name}. Existing sessions were signed out and a private password change is required at next sign-in.`,
  });
});

app.post('/api/wallet/buy-pro', requireAuth, async (req, res) => {
  res.status(410).json({
    error: 'Subscriptions use recurring PayPal billing. Mcoins are reserved for one-time purchases and translation overages.',
  });
});

app.post('/api/wallet/withdraw', requireAuth, async (req, res) => {
  const requestedMcoins = Number(req.body.amountMcoins);
  const amountMcoins = Number.isFinite(requestedMcoins)
    ? Math.floor(requestedMcoins * 100) / 100
    : Number.NaN;
  const payoutEmail = String(req.body.payoutEmail || '').trim().toLowerCase();
  const policies = sitePolicies(req.db);
  const minimumWithdrawalMcoins = policies.minimumWithdrawalMcoins;
  if (!Number.isFinite(amountMcoins) || amountMcoins <= 0 || amountMcoins < minimumWithdrawalMcoins) return res.status(400).json({ error: `Withdrawal must be greater than 0 and at least ${minimumWithdrawalMcoins.toLocaleString()} Mcoins.` });
  if (policies.maximumWithdrawalMcoins > 0 && amountMcoins > policies.maximumWithdrawalMcoins) {
    return res.status(400).json({ error: `Maximum withdrawal is ${policies.maximumWithdrawalMcoins.toLocaleString()} Mcoins per request.` });
  }
  if (req.user.mcoins < amountMcoins) return res.status(402).json({ error: 'Insufficient Mcoin balance.' });
  if (!/^\S+@\S+\.\S+$/.test(payoutEmail)) return res.status(400).json({ error: 'Enter a valid payout email.' });
  const todayUtc = new Date().toISOString().slice(0, 10);
  const userOutflowToday = req.db.withdrawals
    .filter((item) => item.userId === req.user.id && String(item.createdAt || '').startsWith(todayUtc) && !['rejected', 'cancelled'].includes(String(item.status || '').toLowerCase()))
    .reduce((total, item) => total + Number(item.amountMcoins || 0), 0);
  if (policies.dailyWithdrawalLimitMcoins > 0 && userOutflowToday + amountMcoins > policies.dailyWithdrawalLimitMcoins) {
    const remaining = Math.max(0, Number((policies.dailyWithdrawalLimitMcoins - userOutflowToday).toFixed(2)));
    return res.status(400).json({ error: `Daily withdrawal limit reached. ${remaining.toLocaleString()} Mcoins remain today.` });
  }
  const withdrawalFeeRate = policies.withdrawalFeePercent / 100;
  const feeMcoins = Number((amountMcoins * withdrawalFeeRate).toFixed(2));
  const netMcoins = Number((amountMcoins - feeMcoins).toFixed(2));
  const pendingOutflow = req.db.withdrawals
    .filter((item) => String(item.status || '').toLowerCase().startsWith('pending'))
    .reduce((total, item) => total + Number(item.netMcoins || 0), 0);
  if (policies.maximumPendingWithdrawalOutflowMcoins > 0 && pendingOutflow + netMcoins > policies.maximumPendingWithdrawalOutflowMcoins) {
    return res.status(409).json({ error: 'The platform pending payout limit has been reached. Please try again after an administrator processes existing payouts.' });
  }
  req.user.mcoins = Number((req.user.mcoins - amountMcoins).toFixed(2));
  req.user.withdrawableMcoins = Math.min(
    Number(req.user.withdrawableMcoins || 0),
    req.user.mcoins,
  );
  const platform = req.db.users.find((user) => user.id === 'platform');
  if (platform) {
    platform.mcoins = Number((Number(platform.mcoins || 0) + feeMcoins).toFixed(2));
    addLedger(req.db, platform.id, feeMcoins, 'cashout_fee', `${req.user.name} cash-out fee`);
  }
  const withdrawal = {
    id: id('withdrawal'),
    userId: req.user.id,
    amountMcoins,
    feeMcoins,
    netMcoins,
    feeRate: withdrawalFeeRate,
    payoutEmail,
    status: 'pending_manual_review',
    createdAt: new Date().toISOString(),
  };
  req.db.withdrawals.push(withdrawal);
  addLedger(req.db, req.user.id, -amountMcoins, 'withdrawal_requested', `${policies.withdrawalFeePercent}% cash-out fee: ${feeMcoins} Mcoins; net: ${netMcoins} Mcoins`);
  await writeDb(req.db);
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

function configuredSubscriptionPlanId(product, upgrade = false) {
  if (!product || product.kind !== 'subscription') return '';
  if (upgrade && product.tier === 'musician') {
    const key = product.interval === 'YEAR'
      ? 'PAYPAL_CHILL_TO_MUSICIAN_YEARLY_PLAN_ID'
      : 'PAYPAL_CHILL_TO_MUSICIAN_MONTHLY_PLAN_ID';
    return String(process.env[key] || '').trim();
  }
  const keys = {
    'polymath-chill-monthly': 'PAYPAL_CHILL_MONTHLY_PLAN_ID',
    'polymath-chill-yearly': 'PAYPAL_CHILL_YEARLY_PLAN_ID',
    'polymath-musician-monthly': 'PAYPAL_MUSICIAN_MONTHLY_PLAN_ID',
    'polymath-musician-yearly': 'PAYPAL_MUSICIAN_YEARLY_PLAN_ID',
    'polymath-institution-class-monthly': 'PAYPAL_INSTITUTION_CLASS_MONTHLY_PLAN_ID',
    'polymath-institution-class-yearly': 'PAYPAL_INSTITUTION_CLASS_YEARLY_PLAN_ID',
    'polymath-institution-cohort-monthly': 'PAYPAL_INSTITUTION_COHORT_MONTHLY_PLAN_ID',
    'polymath-institution-cohort-yearly': 'PAYPAL_INSTITUTION_COHORT_YEARLY_PLAN_ID',
    'polymath-institution-school-monthly': 'PAYPAL_INSTITUTION_SCHOOL_MONTHLY_PLAN_ID',
    'polymath-institution-school-yearly': 'PAYPAL_INSTITUTION_SCHOOL_YEARLY_PLAN_ID',
    'polymath-pro': 'PAYPAL_PRO_PLAN_ID',
  };
  const configured = String(process.env[keys[product.id]] || '').trim();
  if (product.id === 'polymath-musician-monthly') {
    return configured || String(process.env.PAYPAL_PRO_PLAN_ID || '').trim();
  }
  return configured;
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
    const active = subscriptionStatusGrantsPro(normalizedStatus);
    const product = PRODUCTS[record?.productId] || PRODUCTS['polymath-pro'];
    if (active) {
      const firstActivation = user.paypalSubscriptionId !== subscriptionId || !record?.activatedAt;
      user.paypalSubscriptionId = subscriptionId;
      user.proStatus = normalizedStatus;
      user.pro = true;
      user.subscriptionTier = product.tier || 'musician';
      user.subscriptionInterval = product.interval || 'MONTH';
      if (firstActivation) {
        const activatedAt = new Date().toISOString();
        user.subscriptionStartedAt = activatedAt;
        if (record && !record.activatedAt) record.activatedAt = activatedAt;
        user.translationUsage = {
          period: translationUsageWindow(user).key,
          includedUsed: 0,
        };
      }
    } else if (user.paypalSubscriptionId === subscriptionId) {
      user.proStatus = normalizedStatus;
      user.pro = false;
      user.subscriptionTier = '';
      user.subscriptionInterval = null;
    }
    syncInstitutionSubscription(db, record, user, product, active);
  }

  return user || null;
}

async function cancelPreviousSubscriptionForUpgrade(db, record) {
  if (!record?.isUpgrade || !record.upgradeFromSubscriptionId || record.upgradeCancelledAt) return;
  const cancellation = await paypalRequest(
    `/v1/billing/subscriptions/${encodeURIComponent(record.upgradeFromSubscriptionId)}/cancel`,
    {
      method: 'POST',
      body: JSON.stringify({ reason: 'Upgraded to Polymath Musician with a new billing period.' }),
    },
  );
  if (!cancellation.response.ok && cancellation.response.status !== 204) {
    record.upgradeCancellationError = cancellation.data;
    const error = new Error(
      'Musician was approved, but the previous Chill renewal could not be cancelled automatically.',
    );
    error.status = 502;
    error.details = cancellation.data;
    throw error;
  }
  record.upgradeCancelledAt = new Date().toISOString();
  record.upgradeCancellationError = null;
  const previous = db.subscriptions.find(
    (item) => item.subscriptionId === record.upgradeFromSubscriptionId,
  );
  if (previous) {
    previous.status = 'CANCELLED';
    previous.cancelledAt = record.upgradeCancelledAt;
  }
}

async function validatePayPalSubscriptionPlan(product, planId, upgradeFrom = null) {
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
      `The configured PayPal plan must be an active ${product.currency} ${expectedPrice} ${product.interval.toLowerCase()} plan.`,
    );
    error.status = 409;
    throw error;
  }

  if (upgradeFrom) {
    const trialCycle = (data.billing_cycles || []).find((cycle) => cycle.tenure_type === 'TRIAL');
    const setupFee = data.payment_preferences?.setup_fee;
    const expectedDifference = (Number(product.price) - Number(upgradeFrom.price)).toFixed(2);
    const actualSetupFee = Number(setupFee?.value).toFixed(2);
    if (
      !trialCycle
      || Number(trialCycle.total_cycles || 0) !== 1
      || trialCycle.frequency?.interval_unit !== product.interval
      || setupFee?.currency_code !== product.currency
      || actualSetupFee !== expectedDifference
    ) {
      const error = new Error(
        `The PayPal upgrade plan must charge a ${product.currency} ${expectedDifference} setup fee, then start regular ${product.name} billing after one ${product.interval.toLowerCase()}.`,
      );
      error.status = 409;
      throw error;
    }
  }

  return data;
}

app.post('/api/paypal/create-order', requireAuth, async (req, res) => {
  try {
    const product = PRODUCTS[String(req.body.productId || '')];
    if (!product) return res.status(404).json({ error: 'Product not found.' });
    if (product.kind !== 'mcoins') {
      return res.status(400).json({ error: 'Recurring plan access must use the subscription checkout.' });
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
    await writeDb(req.db);
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
    await writeDb(req.db);
    res.json({ user: safeUser(req.user), product, paypalStatus: paymentData.status });
  } catch (error) {
    console.error('PayPal capture order failed:', error.message);
    res.status(error.status || 500).json({ error: error.message, ...(error.details ? { details: error.details } : {}) });
  }
});

app.post('/api/paypal/create-subscription', requireAuth, async (req, res) => {
  try {
    const product = PRODUCTS[String(req.body.productId || 'polymath-musician-monthly')];
    if (!product || product.kind !== 'subscription' || product.legacy) {
      return res.status(404).json({ error: 'Subscription product not found.' });
    }

    const currentTier = activeSubscriptionTier(req.user);
    const currentInterval = String(req.user.subscriptionInterval || 'MONTH').toUpperCase();
    const institutionPurchase = Boolean(product.institutionTier);
    const isUpgrade = !institutionPurchase && currentTier === 'chill' && product.tier === 'musician';
    if (institutionPurchase && req.user.institutionRole === 'owner' && req.user.institutionStatus === 'ACTIVE') {
      return res.status(409).json({ error: 'An institution plan is already active for this account.' });
    }
    if (!institutionPurchase && currentTier === 'musician') {
      return res.status(409).json({ error: 'Musician is already active for this account.' });
    }
    if (!institutionPurchase && currentTier === 'chill' && !isUpgrade) {
      return res.status(409).json({ error: 'Chill is already active. Choose Musician to upgrade.' });
    }
    if (isUpgrade && product.interval !== currentInterval) {
      return res.status(409).json({
        error: `Choose the ${currentInterval === 'YEAR' ? 'yearly' : 'monthly'} Musician plan to upgrade and reset the same billing period.`,
      });
    }
    const upgradeFrom = isUpgrade
      ? PRODUCTS[currentInterval === 'YEAR' ? 'polymath-chill-yearly' : 'polymath-chill-monthly']
      : null;
    const planId = configuredSubscriptionPlanId(product, isUpgrade);
    if (!planId) {
      return res.status(503).json({
        error: `The PayPal ${isUpgrade ? 'upgrade' : product.name} ${product.interval.toLowerCase()} plan is not configured on the server.`,
      });
    }
    const pricing = subscriptionPriceForUser(product, req.user);

    const reusablePending = req.db.subscriptions.find((item) => (
      item.userId === req.user.id
      && item.productId === product.id
      && String(item.checkoutPrice || product.price) === pricing.price
      && String(item.luckyCode || '') === pricing.luckyCode
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

    const planDetails = await validatePayPalSubscriptionPlan(product, planId, upgradeFrom);
    const regularCycle = (planDetails.billing_cycles || []).find((cycle) => cycle.tenure_type === 'REGULAR');
    const planOverride = pricing.discountPercent > 0 ? {
      billing_cycles: [{
        sequence: Number(regularCycle.sequence),
        total_cycles: Number(regularCycle.total_cycles || 0),
        pricing_scheme: {
          fixed_price: { currency_code: product.currency, value: pricing.price },
        },
      }],
    } : null;

    const requestId = id('paypal_subscription');
    const { response, data } = await paypalRequest('/v1/billing/subscriptions', {
      method: 'POST',
      headers: { 'PayPal-Request-Id': requestId },
      body: JSON.stringify({
        plan_id: planId,
        custom_id: req.user.id,
        ...(planOverride ? { plan: planOverride } : {}),
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
      checkoutPrice: pricing.price,
      discountPercent: pricing.discountPercent,
      luckyCode: pricing.luckyCode,
      isUpgrade,
      upgradeFromSubscriptionId: isUpgrade ? req.user.paypalSubscriptionId : null,
      upgradeFromProductId: upgradeFrom?.id || null,
      createdAt: new Date().toISOString(),
    });
    if (!isUpgrade) {
      req.user.paypalSubscriptionId = data.id;
      req.user.proStatus = data.status || 'APPROVAL_PENDING';
      req.user.pro = false;
    }
    await writeDb(req.db);
    res.status(201).json({
      subscriptionId: data.id,
      approveUrl,
      product,
      status: data.status,
      checkoutPrice: pricing.price,
      discountPercent: pricing.discountPercent,
      upgrade: isUpgrade,
      upgradePrice: isUpgrade ? (Number(product.price) - Number(upgradeFrom.price)).toFixed(2) : null,
    });
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
      return res.status(409).json({ error: 'PayPal subscription plan does not match the selected plan.' });
    }

    const product = PRODUCTS[record.productId] || PRODUCTS['polymath-pro'];
    const activating = subscriptionStatusGrantsPro(data.status);
    const firstActivation = activating && !record.activatedAt;
    if (activating) await cancelPreviousSubscriptionForUpgrade(req.db, record);
    const user = applySubscriptionStatus(req.db, subscriptionId, data.status, req.user.id);
    if (user?.pro && firstActivation) {
      addLedger(
        req.db,
        user.id,
        0,
        record.isUpgrade ? 'subscription_upgraded' : 'subscription_active',
        `${product.name} ${product.interval.toLowerCase()}`,
      );
    }
    await writeDb(req.db);

    res.json({
      user: safeUser(req.user),
      product,
      subscriptionStatus: data.status,
      active: subscriptionStatusGrantsPro(data.status),
      upgraded: Boolean(record.isUpgrade && activating),
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

    const db = await readDb();
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
      const record = db.subscriptions.find((item) => item.subscriptionId === subscriptionId);
      if (subscriptionStatusGrantsPro(webhookStatus) && record?.isUpgrade) {
        try {
          await cancelPreviousSubscriptionForUpgrade(db, record);
        } catch (error) {
          await writeDb(db);
          throw error;
        }
      }
      applySubscriptionStatus(db, subscriptionId, webhookStatus, resource.custom_id || '');
    }

    db.webhookEvents.push({
      eventId: eventId || id('webhook'),
      eventType,
      resourceId: subscriptionId || String(resource.id || ''),
      receivedAt: new Date().toISOString(),
    });
    if (db.webhookEvents.length > 1000) db.webhookEvents = db.webhookEvents.slice(-1000);
    await writeDb(db);
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
    engine: job.omrEngine || '',
    confidence: Number(job.confidence || 0),
    warnings: Array.isArray(job.warnings) ? job.warnings.slice(0, 12) : [],
    pianoPerformance: job.pianoPerformance && typeof job.pianoPerformance === 'object'
      ? job.pianoPerformance
      : null,
    outputFilename: job.status === 'completed' ? job.outputFilename : undefined,
    personalSongId: job.status === 'completed' ? job.personalSongId : undefined,
  };
}

const TEACHER_LEVELS = new Set(['beginner', 'intermediate', 'advanced']);
const TEACHER_LESSON_MODES = new Set(['online', 'in-person']);

function teacherMarketplaceTerms(policies) {
  const feePercent = clampDecimal(
    policies?.teacherMarketplaceFeePercent,
    0,
    100,
    DEFAULT_SITE_POLICIES.teacherMarketplaceFeePercent,
  );
  return {
    directoryEnabled: policies?.teacherDirectoryEnabled !== false,
    applicationsEnabled: policies?.teacherApplicationsEnabled !== false,
    reviewsEnabled: policies?.teacherReviewsEnabled !== false,
    minimumHourlyRateMcoins: clampDecimal(
      policies?.minimumTeacherHourlyRateMcoins,
      0,
      1000000000,
      DEFAULT_SITE_POLICIES.minimumTeacherHourlyRateMcoins,
    ),
    maximumHourlyRateMcoins: clampDecimal(
      policies?.maximumTeacherHourlyRateMcoins,
      0,
      1000000000,
      DEFAULT_SITE_POLICIES.maximumTeacherHourlyRateMcoins,
    ),
    platformFeePercent: feePercent,
    teacherKeepsPercent: Number((100 - feePercent).toFixed(2)),
    withdrawalFeePercent: clampDecimal(
      policies?.withdrawalFeePercent,
      0,
      100,
      DEFAULT_SITE_POLICIES.withdrawalFeePercent,
    ),
    notice: String(policies?.teacherMarketplaceNotice || '').trim().slice(0, 600),
    checkoutAvailable: false,
  };
}

function teacherReviewSummary(db, teacherProfileId) {
  const reviews = db.teacherReviews.filter((review) => review.teacherProfileId === teacherProfileId);
  const total = reviews.reduce((sum, review) => sum + Number(review.rating || 0), 0);
  return {
    averageRating: reviews.length ? Number((total / reviews.length).toFixed(2)) : 0,
    reviewCount: reviews.length,
  };
}

function hasTeacherConversation(db, studentId, teacherUserId) {
  if (!studentId || !teacherUserId || studentId === teacherUserId) return false;
  return db.messages.some((message) => (
    (message.fromUserId === studentId && message.toUserId === teacherUserId)
    || (message.fromUserId === teacherUserId && message.toUserId === studentId)
  ));
}

function teacherStudentCount(db, teacherUserId) {
  return new Set(
    db.messages
      .filter((message) => message.fromUserId === teacherUserId || message.toUserId === teacherUserId)
      .map((message) => message.fromUserId === teacherUserId ? message.toUserId : message.fromUserId)
      .filter((userId) => userId && userId !== teacherUserId),
  ).size;
}

function publicTeacherProfile(profile, db, viewerId = null) {
  const teacher = db.users.find((user) => user.id === profile.userId);
  if (!teacher) return null;
  const reviewSummary = teacherReviewSummary(db, profile.id);
  const studentCount = teacherStudentCount(db, teacher.id);
  return {
    id: profile.id,
    user_id: teacher.id,
    name: teacher.name || 'Music teacher',
    avatarUrl: teacher.avatarUrl || '',
    headline: profile.headline,
    bio: profile.bio,
    instruments: profile.instruments,
    levels: profile.levels,
    lessonModes: profile.lessonModes,
    location: profile.location || '',
    languages: profile.languages || [],
    availability: profile.availability || '',
    hourlyRateMcoins: Number(profile.hourlyRateMcoins || 0),
    published: profile.published !== false,
    reviewSummary,
    studentCount,
    ranking: marketplaceRanking(reviewSummary.averageRating, studentCount),
    isSelf: viewerId === teacher.id,
    canReview: hasTeacherConversation(db, viewerId, teacher.id),
    createdAt: profile.createdAt,
    updatedAt: profile.updatedAt || null,
  };
}

function publicTeacherReview(review, db, viewerId = null) {
  const author = db.users.find((user) => user.id === review.studentId);
  return {
    id: review.id,
    teacherProfileId: review.teacherProfileId,
    rating: Number(review.rating),
    comment: review.comment,
    createdAt: review.createdAt,
    updatedAt: review.updatedAt || null,
    mine: viewerId === review.studentId,
    connectedStudent: true,
    author: {
      user_id: review.studentId,
      name: author?.name || 'Former student',
      avatarUrl: author?.avatarUrl || '',
    },
  };
}

function cleanStringList(value, { maximum = 8, allowed = null } = {}) {
  const entries = Array.isArray(value) ? value : [];
  return [...new Set(entries
    .map((entry) => String(entry || '').trim().toLowerCase())
    .filter((entry) => entry && (!allowed || allowed.has(entry))))]
    .slice(0, maximum);
}

function refundTranslationJob(db, job, reason) {
  if (job.refundedAt) return;
  const user = db.users.find((candidate) => candidate.id === job.userId);
  if (user) {
    if (job.paymentMethod === 'mcoins') {
      const refundMcoins = Math.max(0, Number(job.costMcoins || 0));
      user.mcoins += refundMcoins;
      addLedger(db, user.id, refundMcoins, 'translation_refund', job.filename);
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
    throw new Error('The local music reader did not return a readable music-sheet result.');
  }

  if (rawResult.isInstrumentalMusicSheet !== true) {
    throw new Error(
      rawResult.rejectionReason
      || 'Invalid PDF music sheet. Please upload a readable instrumental music sheet.',
    );
  }

  const notes = (Array.isArray(rawResult.notes) ? rawResult.notes : [])
    .map((note) => {
      const duration = clampNumber(note.duration, 0.01, 60 * 60, 0.25);
      const normalized = {
        note: normalizePitchName(note.note),
        time: clampNumber(note.time, 0, 24 * 60 * 60, 0),
        duration,
        scoreDuration: clampNumber(note.scoreDuration, 0.01, 60 * 60, duration),
        visualDuration: clampNumber(note.visualDuration, 0.01, 60 * 60, duration),
        audioDuration: clampNumber(note.audioDuration, 0.01, 60 * 60, duration),
        velocity: clampNumber(note.velocity, 0.01, 1, 0.75),
        hand: ['left', 'right', 'both', ''].includes(note.hand) ? note.hand : '',
        voice: String(note.voice || '').slice(0, 80),
        articulation: String(note.articulation || '').slice(0, 80),
        dynamic: String(note.dynamic || '').slice(0, 24),
        measure: Math.round(clampNumber(note.measure, 0, 100_000, 0)),
        measureBeat: clampNumber(note.measureBeat, 0, 128, 0),
      };
      if (Number.isFinite(Number(note.releaseSeconds))) {
        normalized.releaseSeconds = clampNumber(note.releaseSeconds, 0.01, 8, 0.58);
      }
      return normalized;
    })
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
      value: pedal.down ? Math.round(clampNumber(pedal.value, 64, 127, 127)) : 0,
      controller: 64,
      source: String(pedal.source || '').slice(0, 80),
      inferred: pedal.inferred === true,
      confidence: clampNumber(pedal.confidence, 0, 1, pedal.inferred === true ? 0.5 : 1),
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
    performance: rawResult.performance && typeof rawResult.performance === 'object'
      ? {
          profile: String(rawResult.performance.profile || '').slice(0, 80),
          preserveScoreDurations: rawResult.performance.preserveScoreDurations === true,
          preserveScoreTiming: rawResult.performance.preserveScoreTiming === true,
          durationFieldPolicy: String(rawResult.performance.durationFieldPolicy || '').slice(0, 80),
          sameKeyRetriggerGapSeconds: clampNumber(
            rawResult.performance.sameKeyRetriggerGapSeconds, 0.01, 0.2, 0.038,
          ),
          defaultAutoplayReleaseSeconds: clampNumber(
            rawResult.performance.defaultAutoplayReleaseSeconds, 0.1, 2, 0.58,
          ),
        }
      : undefined,
    pianoPerformance: rawResult.pianoPerformance && typeof rawResult.pianoPerformance === 'object'
      ? {
          voices: Math.round(clampNumber(rawResult.pianoPerformance.voices, 0, 64, 0)),
          restrikesGivenReleaseGap: Math.round(clampNumber(
            rawResult.pianoPerformance.restrikesGivenReleaseGap, 0, 1_000_000, 0,
          )),
          legatoConnections: Math.round(clampNumber(
            rawResult.pianoPerformance.legatoConnections, 0, 1_000_000, 0,
          )),
          pedalSource: String(rawResult.pianoPerformance.pedalSource || 'none').slice(0, 80),
          pedalEvents: Math.round(clampNumber(rawResult.pianoPerformance.pedalEvents, 0, 1_000_000, 0)),
          writtenAndPhysicalDurationsSeparated:
            rawResult.pianoPerformance.writtenAndPhysicalDurationsSeparated === true,
        }
      : undefined,
  };
}

async function processTranslationJob(jobId) {
  let job = await claimBackgroundJob('scoreTranslationJobs', jobId);
  if (!job) return;
  let db;
  let sourcePath = '';
  let outputPath = '';

  try {
    db = await readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.stage = 'Rendering PDF pages locally';
    job.progress = 18;
    await writeDb(db);

    const sourceWorkPath = path.join(UPLOAD_DIR, `${job.id}-source.pdf`);
    sourcePath = await ARTIFACT_STORE.materialize(job.sourcePath, sourceWorkPath);
    const bytes = fs.readFileSync(sourcePath);

    db = await readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.stage = 'Detecting staffs, symbols, pitch, and rhythm';
    job.progress = 42;
    await writeDb(db);

    // The bytes read above prove that the materialized artifact is present and
    // non-empty before Python begins a potentially expensive page render.
    if (!bytes.length) throw new Error('The stored PDF is empty.');
    const outputName = `${job.id}-${sanitizeFilename(job.filename.replace(/\.pdf$/i, '') || 'ready-to-play-sheet')}.json`;
    outputPath = path.join(UPLOAD_DIR, outputName);
    const localOmr = await runLocalOmr({
      sourcePath,
      outputPath,
      filename: job.filename,
      instrument: job.instrument,
    });
    const result = normalizeReadyToPlaySong(localOmr.result, job.instrument);

    db = await readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.stage = 'Validating local notation evidence';
    job.progress = 82;
    job.omrEngine = String(localOmr.summary?.engine || localOmr.result?.omrDiagnostics?.engine || 'polymath-local-omr');
    job.confidence = Number(result.confidence || 0);
    job.warnings = result.warnings;
    job.pianoPerformance = result.pianoPerformance || null;
    await writeDb(db);

    fs.writeFileSync(outputPath, JSON.stringify({
      ...result,
      sourcePdf: job.filename,
      readyToPlayFormat: 'polymath-musician-json-v1',
      translationProvider: 'Polymath Local OMR',
      omrEngine: job.omrEngine,
      omrDiagnostics: localOmr.result?.omrDiagnostics || {},
      translatedAt: new Date().toISOString(),
    }, null, 2));

    db = await readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    job.outputPath = await ARTIFACT_STORE.putFile(
      artifactKey('score-translations', outputName),
      outputPath,
      'application/json',
    );
    job.outputFilename = `${sanitizeFilename(job.filename.replace(/\.pdf$/i, '') || 'ready-to-play-sheet')}.json`;
    job.status = 'completed';
    job.stage = 'Ready to download';
    job.progress = 100;
    job.completedAt = new Date().toISOString();
    attachGeneratedPersonalSong(db, job, {
      title: result.title || path.basename(job.filename, path.extname(job.filename)),
      artist: result.artist || result.composer || '',
      instrument: job.instrument,
      filename: job.outputFilename,
      assetPath: job.outputPath,
      bytes: fs.readFileSync(outputPath),
      sourceJobType: 'score-translation',
    });
    await writeDb(db);
    if (ARTIFACT_STORE.remote) {
      safeRemoveUpload(sourcePath);
      safeRemoveUpload(outputPath);
    }
  } catch (error) {
    db = await readDb();
    job = db.scoreTranslationJobs.find((candidate) => candidate.id === jobId);
    if (!job || job.status !== 'processing') return;
    refundTranslationJob(db, job, error.message);
    await writeDb(db);
  } finally {
    safeRemoveUpload(sourcePath);
    if (ARTIFACT_STORE.remote) safeRemoveUpload(outputPath);
    await safeRemoveArtifact(job?.sourcePath);
  }
}

app.post('/api/artifact-upload-intents', requireAuth, async (req, res) => {
  if (!DIRECT_UPLOADS.enabled) return res.json({ direct: false });

  const purpose = String(req.body.purpose || '').trim().toLowerCase();
  const filename = sanitizeFilename(String(req.body.filename || 'upload'));
  const extension = path.extname(filename).toLowerCase();
  const size = Math.floor(Number(req.body.size));
  if (!Number.isSafeInteger(size) || size <= 0 || size > DIRECT_UPLOAD_MAX_BYTES) {
    return res.status(400).json({ error: 'The selected file size is invalid or exceeds the 5 GB object-storage limit.' });
  }

  if (purpose === 'score-translation') {
    if (!localOmrAvailability().enabled) {
      return res.status(503).json({ error: 'The local PDF music reader is disabled. Nothing was charged.' });
    }
    if (extension !== '.pdf' || size > MAX_PDF_BYTES) {
      return res.status(400).json({ error: 'PDF music sheets must be valid PDF files smaller than 10 MB.' });
    }
  } else if (purpose === 'media-transcription') {
    const capability = muscriptorAvailability();
    if (!capability.enabled) return res.status(503).json({ error: capability.reason, capability });
    if (MUSCRIPTOR_ADMIN_ONLY && !isAdministrator(req.user)) {
      return res.status(403).json({
        error: 'Polymath is currently available only to administrators for model testing.',
        capability,
      });
    }
    if (!MEDIA_EXTENSIONS.has(extension)) {
      return res.status(400).json({ error: 'Use MP3, WAV, FLAC, M4A, MP4, MOV, WebM, MKV, or AVI.' });
    }
  } else if (purpose === 'personal-song') {
    if (!['.json', '.mid', '.midi'].includes(extension) || size > MARKETPLACE_MAX_BYTES) {
      return res.status(400).json({ error: 'Ready-to-play JSON or MIDI songs must be smaller than 8 MB.' });
    }
  } else {
    return res.status(400).json({ error: 'Choose a supported upload purpose.' });
  }

  const userSegment = String(req.user.id).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80);
  const key = `pending/${purpose}/${userSegment}/${crypto.randomUUID()}-${filename}`;
  const intent = await DIRECT_UPLOADS.create({
    userId: req.user.id,
    purpose,
    key,
    filename,
    contentType: uploadContentType(filename, req.body.contentType),
    size,
  });
  return res.status(201).json(intent);
});

async function submitMediaTranscription(req, res, {
  fields,
  originalName,
  mimeType,
  sourceKey = '',
  persistSource,
  cleanup,
}) {
  const capability = muscriptorAvailability();
  const cleanupAndReject = async (status, message, details = {}) => {
    await cleanup?.();
    return res.status(status).json({ error: message, ...details });
  };
  if (!capability.enabled) {
    return cleanupAndReject(503, capability.reason, { capability });
  }
  if (MUSCRIPTOR_ADMIN_ONLY && !isAdministrator(req.user)) {
    return cleanupAndReject(403, 'Polymath is currently available only to administrators for model testing.', { capability });
  }

  const extension = path.extname(originalName || '').toLowerCase();
  if (!MEDIA_EXTENSIONS.has(extension)) {
    return cleanupAndReject(400, 'Use MP3, WAV, FLAC, M4A, MP4, MOV, WebM, MKV, or AVI.');
  }
  const instrument = String(fields.instrument || '').trim().toLowerCase();
  if (instrument !== 'band' && !validBandInstrument(instrument)) {
    return cleanupAndReject(400, 'Choose a supported instrument or band transcription.');
  }
  if (String(fields.rightsConfirmed || '').toLowerCase() !== 'true') {
    return cleanupAndReject(400, 'Confirm that you have permission to transcribe this recording.');
  }
  const playbackMode = String(fields.playbackMode || 'instrumental').trim().toLowerCase();
  if (!['full', 'instrumental'].includes(playbackMode)) {
    return cleanupAndReject(400, 'Choose full song or pure instrumental playback.');
  }

  if (sourceKey) {
    const duplicate = req.db.mediaTranscriptionJobs.find((job) => (
      job.userId === req.user.id && job.sourcePath === sourceKey
    ));
    if (duplicate) {
      return res.json({
        job: publicMediaTranscriptionJob(duplicate),
        capability,
        user: safeUser(req.user),
        duplicate: true,
      });
    }
  }

  const administratorAccess = isAdministrator(req.user);
  const requestedPaymentMethod = String(fields.paymentMethod || '').trim().toLowerCase();
  const paymentMethod = administratorAccess ? 'admin' : requestedPaymentMethod;
  const mcoinCost = translationMcoinCost(req.user);
  if (!administratorAccess && !['allowance', 'mcoins'].includes(paymentMethod)) {
    return cleanupAndReject(400, `Choose an included translation or ${mcoinCost}-Mcoin payment.`);
  }
  let allowanceBucket = null;
  if (paymentMethod === 'allowance') {
    allowanceBucket = deductTranslationAllowance(req.user);
    if (!allowanceBucket) {
      return cleanupAndReject(402, `No included translations remain. Pay ${mcoinCost} Mcoins to continue.`);
    }
  } else if (paymentMethod === 'mcoins') {
    if (req.user.mcoins < mcoinCost) {
      return cleanupAndReject(402, `You need ${mcoinCost} Mcoins for this translation.`);
    }
    req.user.mcoins -= mcoinCost;
  }

  const filename = sanitizeFilename(originalName || `recording${extension}`);
  const title = String(fields.title || path.basename(filename, extension) || 'Uploaded recording')
    .trim()
    .slice(0, 120);
  const jobId = id('media_tx');
  const persistedSourceKey = sourceKey
    ? await ARTIFACT_STORE.promote(sourceKey, `media/${jobId}-${filename}`)
    : await persistSource({ filename, mimeType });
  const job = {
    id: jobId,
    userId: req.user.id,
    filename,
    title,
    instrument,
    playbackMode,
    paymentMethod,
    allowanceBucket,
    costMcoins: paymentMethod === 'mcoins' ? mcoinCost : 0,
    model: `polymath-${MUSCRIPTOR_MODEL}`,
    modelLicense: 'CC-BY-NC-4.0',
    sourcePath: persistedSourceKey,
    status: 'processing',
    stage: 'Queued for Polymath',
    progress: 5,
    startedAt: new Date().toISOString(),
  };
  req.db.mediaTranscriptionJobs.push(job);
  if (paymentMethod === 'mcoins') {
    addLedger(req.db, req.user.id, -mcoinCost, 'audio_translation', filename);
  } else if (paymentMethod === 'allowance') {
    addLedger(req.db, req.user.id, 0, 'translation_allowance_used', `${filename} (${allowanceBucket})`);
  } else {
    addLedger(req.db, req.user.id, 0, 'admin_audio_translation', `${filename} (unlimited administrator access)`);
  }
  try {
    await writeDb(req.db);
  } catch (error) {
    await safeRemoveArtifact(persistedSourceKey);
    throw error;
  }
  await dispatchBackgroundJob('media-transcription', job.id);
  return res.status(202).json({
    job: publicMediaTranscriptionJob(job),
    capability,
    user: safeUser(req.user),
    translationMcoinCost: mcoinCost,
  });
}

app.get('/api/media-transcriptions', requireAuth, async (req, res) => {
  const jobs = req.db.mediaTranscriptionJobs
    .filter((job) => job.userId === req.user.id)
    .sort((a, b) => String(b.startedAt).localeCompare(String(a.startedAt)))
    .slice(0, 20)
    .map(publicMediaTranscriptionJob);
  res.json({ jobs, capability: muscriptorAvailability(), user: safeUser(req.user) });
});

app.post('/api/media-transcriptions', requireAuth, async (req, res, next) => {
  return mediaUpload.single('media')(req, res, async (uploadError) => {
    if (uploadError) {
      if (req.file?.path) safeRemoveUpload(req.file.path);
      const message = uploadError.code === 'LIMIT_FIELD_COUNT'
        ? 'The transcription form contained more fields than the server accepts. Refresh the page and try again.'
        : uploadError.message || 'The audio or video upload failed.';
      return res.status(400).json({ error: message });
    }
    if (!req.file) {
      return res.status(400).json({ error: 'Choose an MP3, audio file, or music video.' });
    }

    try {
      return await submitMediaTranscription(req, res, {
        fields: req.body,
        originalName: req.file.originalname,
        mimeType: req.file.mimetype,
        cleanup: async () => safeRemoveUpload(req.file.path),
        persistSource: async () => {
          const key = artifactKey('media', path.basename(req.file.path));
          await ARTIFACT_STORE.putFile(key, req.file.path, req.file.mimetype);
          if (ARTIFACT_STORE.remote) safeRemoveUpload(req.file.path);
          return key;
        },
      });
    } catch (error) {
      safeRemoveUpload(req.file.path);
      return next(error);
    }
  });
});

app.post('/api/media-transcriptions/direct', requireAuth, async (req, res) => {
  const upload = await DIRECT_UPLOADS.inspect(req.body.uploadReceipt, {
    userId: req.user.id,
    purpose: 'media-transcription',
  });
  return submitMediaTranscription(req, res, {
    fields: req.body,
    originalName: upload.filename,
    mimeType: upload.contentType,
    sourceKey: upload.key,
    cleanup: async () => safeRemoveArtifact(upload.key),
  });
});

app.get('/api/media-transcriptions/:jobId', requireAuth, async (req, res) => {
  const job = req.db.mediaTranscriptionJobs.find((candidate) => (
    candidate.id === req.params.jobId && candidate.userId === req.user.id
  ));
  if (!job) return res.status(404).json({ error: 'Music transcription job not found.' });
  return res.json({
    job: publicMediaTranscriptionJob(job),
    capability: muscriptorAvailability(),
    user: safeUser(req.user),
  });
});

app.get('/api/media-transcriptions/:jobId/download', requireAuth, async (req, res) => {
  const job = req.db.mediaTranscriptionJobs.find((candidate) => (
    candidate.id === req.params.jobId && candidate.userId === req.user.id
  ));
  if (!job) return res.status(404).json({ error: 'Music transcription job not found.' });
  if (job.status !== 'completed' || !job.outputPath) {
    return res.status(409).json({ error: 'The ready-to-play transcription is not available yet.' });
  }
  return ARTIFACT_STORE.sendDownload(
    res,
    job.outputPath,
    job.outputFilename || 'polymath-ready-to-play.json',
    'application/json',
  );
});

app.get('/api/score-translations/usage', requireAuth, async (req, res) => {
  await writeDb(req.db);
  res.json({
    user: safeUser(req.user),
    translationMcoinCost: translationMcoinCost(req.user),
    mcoinsPerUsd: MCOINS_PER_USD,
  });
});

app.get('/api/score-translations', requireAuth, async (req, res) => {
  let changed = false;
  const jobs = req.db.scoreTranslationJobs
    .filter((job) => job.userId === req.user.id)
    .sort((a, b) => b.startedAt.localeCompare(a.startedAt))
    .slice(0, 20)
    .map((job) => {
      if (extendTranslationEstimate(job)) changed = true;
      return publicTranslationJob(job);
    });
  if (changed) await writeDb(req.db);
  res.json({ jobs, user: safeUser(req.user) });
});

app.post('/api/score-translations', requireAuth, async (req, res) => {
  let directUpload = null;
  if (req.body.uploadReceipt) {
    directUpload = await DIRECT_UPLOADS.inspect(req.body.uploadReceipt, {
      userId: req.user.id,
      purpose: 'score-translation',
    });
    const receiptDuplicate = req.db.scoreTranslationJobs.find((job) => (
      job.userId === req.user.id && job.sourcePath === directUpload.key
    ));
    if (receiptDuplicate) {
      return res.json({
        job: publicTranslationJob(receiptDuplicate),
        user: safeUser(req.user),
        duplicate: true,
      });
    }
  }
  const cleanupDirectUpload = async () => safeRemoveArtifact(directUpload?.key);
  if (!localOmrAvailability().enabled) {
    await cleanupDirectUpload();
    return res.status(503).json({
      error: 'PDF translation is temporarily unavailable because the local reader is disabled. Nothing was charged.',
      setup: 'Set OMR_ENABLED=true and install server/omr/requirements.txt on the backend.',
    });
  }

  const filename = sanitizeFilename(String(directUpload?.filename || req.body.filename || 'music-sheet.pdf'));
  const instrument = String(req.body.instrument || '').trim().toLowerCase();
  const administratorAccess = isAdministrator(req.user);
  const requestedPaymentMethod = String(req.body.paymentMethod || '').trim().toLowerCase();
  const paymentMethod = administratorAccess ? 'admin' : requestedPaymentMethod;
  const mcoinCost = translationMcoinCost(req.user);

  if (!filename.toLowerCase().endsWith('.pdf')) {
    await cleanupDirectUpload();
    return res.status(400).json({ error: 'Invalid PDF music sheet. Please upload a PDF music sheet.' });
  }
  if (!INSTRUMENTS[instrument]) {
    await cleanupDirectUpload();
    return res.status(400).json({ error: 'Choose a supported Polymath Musician instrument.' });
  }
  if (!administratorAccess && !['allowance', 'mcoins'].includes(paymentMethod)) {
    await cleanupDirectUpload();
    return res.status(400).json({ error: `Choose an included translation or ${mcoinCost}-Mcoin payment.` });
  }

  let bytes;
  let directWorkPath = '';
  try {
    if (directUpload) {
      directWorkPath = path.join(UPLOAD_DIR, `direct-${crypto.randomUUID()}.pdf`);
      await ARTIFACT_STORE.materialize(directUpload.key, directWorkPath);
      bytes = fs.readFileSync(directWorkPath);
      if (!bytes.length || bytes.length > MAX_PDF_BYTES || bytes.subarray(0, 5).toString('ascii') !== '%PDF-') {
        throw new Error('Invalid PDF music sheet. The selected file is not a valid PDF smaller than 10 MB.');
      }
    } else {
      bytes = decodePdfBase64(String(req.body.contentBase64 || ''));
    }
    rejectClearlyNonMusicPdf(bytes, filename);
  } catch (error) {
    safeRemoveUpload(directWorkPath);
    await cleanupDirectUpload();
    return res.status(400).json({ error: error.message });
  }
  safeRemoveUpload(directWorkPath);

  const fileHash = crypto.createHash('sha256').update(bytes).digest('hex');
  const recentDuplicate = req.db.scoreTranslationJobs.find((job) => (
    job.userId === req.user.id
    && job.fileHash === fileHash
    && ['processing', 'completed'].includes(job.status)
    && Date.now() - new Date(job.startedAt).getTime() < 10 * 60 * 1000
  ));
  if (recentDuplicate) {
    await cleanupDirectUpload();
    extendTranslationEstimate(recentDuplicate);
    await writeDb(req.db);
    return res.json({ job: publicTranslationJob(recentDuplicate), user: safeUser(req.user), duplicate: true });
  }

  let allowanceBucket = null;
  if (paymentMethod === 'admin') {
    // Backend-authorized administrators have unlimited translation access.
  } else if (paymentMethod === 'allowance') {
    allowanceBucket = deductTranslationAllowance(req.user);
    if (!allowanceBucket) {
      await cleanupDirectUpload();
      return res.status(402).json({
        error: `No included translations remain. Pay ${mcoinCost} Mcoins to continue.`,
      });
    }
  } else {
    if (req.user.mcoins < mcoinCost) {
      await cleanupDirectUpload();
      return res.status(402).json({ error: `You need ${mcoinCost} Mcoins for this translation.` });
    }
    req.user.mcoins -= mcoinCost;
    addLedger(req.db, req.user.id, -mcoinCost, 'pdf_translation', filename);
  }

  const jobId = id('translation');
  const sourceName = `${jobId}-${filename}`;
  const finalSourceKey = artifactKey('score-sources', sourceName);
  const sourceKey = directUpload
    ? await ARTIFACT_STORE.promote(directUpload.key, finalSourceKey)
    : await ARTIFACT_STORE.putBuffer(finalSourceKey, bytes, 'application/pdf');
  const now = Date.now();
  const job = {
    id: jobId,
    userId: req.user.id,
    filename,
    instrument,
    paymentMethod,
    allowanceBucket,
    costMcoins: paymentMethod === 'mcoins' ? mcoinCost : 0,
    fileHash,
    sourcePath: sourceKey,
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
  } else if (paymentMethod === 'admin') {
    addLedger(req.db, req.user.id, 0, 'admin_pdf_translation', `${filename} (unlimited administrator access)`);
  }
  try {
    await writeDb(req.db);
  } catch (error) {
    await safeRemoveArtifact(sourceKey);
    throw error;
  }

  await dispatchBackgroundJob('score-translation', job.id);

  res.status(202).json({
    job: publicTranslationJob(job),
    user: safeUser(req.user),
    translationMcoinCost: mcoinCost,
  });
});

app.get('/api/score-translations/:jobId', requireAuth, async (req, res) => {
  const job = req.db.scoreTranslationJobs.find((candidate) => candidate.id === req.params.jobId && candidate.userId === req.user.id);
  if (!job) return res.status(404).json({ error: 'Translation job not found.' });
  const changed = extendTranslationEstimate(job);
  if (changed) await writeDb(req.db);
  res.json({ job: publicTranslationJob(job), user: safeUser(req.user) });
});

app.get('/api/score-translations/:jobId/download', requireAuth, async (req, res) => {
  const job = req.db.scoreTranslationJobs.find((candidate) => candidate.id === req.params.jobId && candidate.userId === req.user.id);
  if (!job) return res.status(404).json({ error: 'Translation job not found.' });
  if (job.status !== 'completed' || !job.outputPath) {
    return res.status(409).json({ error: 'The ready-to-play sheet is not available yet.' });
  }
  return ARTIFACT_STORE.sendDownload(
    res,
    job.outputPath,
    job.outputFilename || 'ready-to-play-sheet.json',
    'application/json',
  );
});

app.post('/api/score-import', async (req, res) => {
  res.status(410).json({
    error: 'Direct PDF conversion has been replaced by the user-facing translation queue. Sign in and use /api/score-translations.',
  });
});

if (IS_PRODUCTION) {
  const frontendDir = path.resolve(__dirname, '..', 'dist');

  app.use('/samples', express.static(path.join(frontendDir, 'samples'), {
    etag: true,
    setHeaders(res, filename) {
      const immutable = path.extname(filename).toLowerCase() === '.wav';
      res.setHeader(
        'Cache-Control',
        immutable
          ? 'public, max-age=31536000, immutable'
          : 'public, max-age=300, must-revalidate',
      );
    },
  }));

  app.use(express.static(frontendDir, {
    etag: true,
    maxAge: '1h',
  }));

  // React owns browser routes. Unknown API routes must still return JSON 404s.
  app.use(async (req, res, next) => {
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({ error: 'API route not found.' });
    }
    if (!['GET', 'HEAD'].includes(req.method) || !req.accepts('html')) {
      return next();
    }
    return res.sendFile(path.join(frontendDir, 'index.html'));
  });
}

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    const message = error.code === 'LIMIT_FILE_SIZE'
      ? 'The upload is too large. Images may be 8 MB and rigged GLB models may be 25 MB.'
      : 'The character upload form was not accepted. Choose one image and try again.';
    return res.status(400).json({ error: message, code: error.code });
  }
  if (error instanceof StateConflictError) {
    return res.status(409).json({
      error: 'Another request updated the same account data. Please retry.',
      code: 'STATE_CONFLICT',
    });
  }
  if (res.headersSent) return next(error);
  console.error('Request failed:', error);
  return res.status(Number(error.status || 500)).json({
    error: Number(error.status) < 500 ? error.message : 'The server could not complete this request.',
  });
});

async function resumePendingTranslationJobs() {
  if (!localOmrAvailability().enabled) return;
  const db = await readDb();
  for (const job of db.scoreTranslationJobs.filter((candidate) => candidate.status === 'processing')) {
    await dispatchBackgroundJob('score-translation', job.id);
  }
}

async function resumePendingMediaTranscriptionJobs() {
  if (!muscriptorAvailability().enabled) return;
  const db = await readDb();
  for (const job of db.mediaTranscriptionJobs.filter((candidate) => (
    candidate.status === 'processing' && candidate.sourcePath
  ))) {
    await dispatchBackgroundJob('media-transcription', job.id);
  }
}

let virtualLessonExpiryTimer = null;
let virtualLessonExpirySweepRunning = false;

function startVirtualLessonExpirySweep() {
  if (virtualLessonExpiryTimer) return;
  virtualLessonExpiryTimer = setInterval(async () => {
    if (virtualLessonExpirySweepRunning) return;
    virtualLessonExpirySweepRunning = true;
    try {
      const db = await readDb();
      if (expireVirtualLessons(db)) await writeDb(db);
    } catch (error) {
      console.error('Virtual lesson expiry sweep failed:', error.message);
    } finally {
      virtualLessonExpirySweepRunning = false;
    }
  }, 30000);
  virtualLessonExpiryTimer.unref?.();
}

async function startServer() {
  ensureStorage();
  await bootstrapAdminAccounts();
  await writeDb(await readDb());
  if (JOB_QUEUE.enabled) {
    JOB_QUEUE.start(runQueuedJob).catch((error) => console.error('Job queue stopped:', error));
  }
  await resumePendingTranslationJobs();
  await resumePendingMediaTranscriptionJobs();
  startVirtualLessonExpirySweep();
  return app.listen(PORT, () => {
    console.log(
      `Polymath Musician backend running on http://localhost:${PORT} `
      + `(${STATE_STORE.provider}, region ${process.env.APP_REGION || 'local'})`,
    );
  });
}

if (require.main === module) {
  startServer().catch((error) => {
    console.error('Polymath Musician failed to start:', error);
    process.exitCode = 1;
  });
}

module.exports = {
  app,
  ensureStorage,
  bootstrapAdminAccounts,
  readDb,
  writeDb,
  startServer,
  selectMuscriptorExecution,
  muscriptorConstraints,
  normalizeReadyToPlaySong,
  subscriptionRules: {
    products: PRODUCTS,
    activeSubscriptionTier,
    applySubscriptionStatus,
    resolveSignupLuckyCode,
    applySignupLuckyCode,
    institutionSeatCount,
    subscriptionPriceForUser,
    hasMusicianAccess,
    chargeReadySheetUpload,
    readySheetAllowance,
    readySheetUploadCost,
    translationAllowance,
    translationMcoinCost,
    translationUsageWindow,
  },
};
