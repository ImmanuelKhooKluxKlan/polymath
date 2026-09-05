'use strict';

const crypto = require('crypto');
const { PollyClient, SynthesizeSpeechCommand } = require('@aws-sdk/client-polly');

const MAX_SPEECH_CHARACTERS = 2800;
const CACHE_TTL_MS = 15 * 60 * 1000;
const CACHE_LIMIT = 200;
const GENERATIVE_REGIONS = new Set([
  'us-east-1',
  'us-west-2',
  'ap-northeast-1',
  'ap-northeast-2',
  'ap-southeast-1',
  'ap-southeast-2',
  'ca-central-1',
  'eu-central-1',
  'eu-west-2',
  'eu-central-2',
]);

// These are original Amazon Polly voices, not imitations of actors or other
// real people. A fixed server-side map keeps each teacher recognisable on every
// phone and prevents a client from turning the endpoint into arbitrary TTS.
const PROFILE_BY_TEACHER = Object.freeze({
  aria: Object.freeze({ voiceId: 'Ruth', fallbackVoiceId: 'Ruth', gender: 'feminine', character: 'Warm and reassuring' }),
  nova: Object.freeze({ voiceId: 'Salli', fallbackVoiceId: 'Salli', gender: 'feminine', character: 'Youthful adult, light, and playful' }),
  anakin: Object.freeze({ voiceId: 'Matthew', fallbackVoiceId: 'Matthew', gender: 'masculine', character: 'Confident and energetic' }),
  taylor: Object.freeze({ voiceId: 'Tiffany', fallbackVoiceId: 'Salli', gender: 'feminine', character: 'Bright and thoughtful' }),
  mace: Object.freeze({ voiceId: 'Stephen', fallbackVoiceId: 'Stephen', gender: 'masculine', character: 'Deep, measured, and exact' }),
});

function clean(value) {
  return String(value || '').trim();
}

function enabledFromEnvironment(environment) {
  const configured = clean(environment.TEACHER_TTS_ENABLED).toLowerCase();
  if (configured) return configured === 'true';
  return clean(environment.NODE_ENV).toLowerCase() === 'production';
}

function speechRegion(environment) {
  const requested = clean(environment.TEACHER_TTS_REGION || environment.APP_REGION || environment.AWS_REGION);
  if (GENERATIVE_REGIONS.has(requested)) return requested;
  return requested.startsWith('ap-') ? 'ap-southeast-1' : 'us-east-1';
}

function teacherSpeechProfile(teacher = {}) {
  const id = clean(teacher.id).toLowerCase();
  const known = PROFILE_BY_TEACHER[id];
  if (known) return { id, ...known, engine: 'generative', languageCode: 'en-US' };
  const gender = clean(teacher.voiceType).toLowerCase() === 'masculine' ? 'masculine' : 'feminine';
  return {
    id: id || 'custom',
    voiceId: gender === 'masculine' ? 'Stephen' : 'Ruth',
    fallbackVoiceId: gender === 'masculine' ? 'Stephen' : 'Ruth',
    gender,
    character: gender === 'masculine' ? 'Natural and measured' : 'Natural and expressive',
    engine: 'generative',
    languageCode: 'en-US',
  };
}

function speechText(value) {
  return clean(value)
    .replace(/```[\s\S]*?```/g, ' code example ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)]\([^\s)]+\)/g, '$1')
    .replace(/\b([A-G])#(-?\d+)\b/g, '$1 sharp $2')
    .replace(/\b([A-G])b(-?\d+)\b/g, '$1 flat $2')
    .replace(/[•*_#>]/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .slice(0, MAX_SPEECH_CHARACTERS)
    .trim();
}

function teacherGreeting({ teacher = {}, studentName = '', conversationMode = 'music-coach' } = {}) {
  const firstName = clean(studentName).split(/\s+/)[0] || 'there';
  const teacherId = clean(teacher.id).toLowerCase();
  if (teacherId === 'nova' && conversationMode === 'adult-companion') {
    return `Oh, hi, sweetheart. I'm Padme. Come sit with me. What kind of mood are you in today?`;
  }
  if (teacherId === 'nova') {
    return `Hi, ${firstName}. I'm Padme. Tell me what you want to play, and we'll make it sound beautiful together.`;
  }
  if (teacherId === 'anakin') {
    return `Hey, ${firstName}. I'm Anakin. Show me the passage you want to strengthen, and we'll make the movement confident.`;
  }
  if (teacherId === 'taylor') {
    return `Hi, ${firstName}. I'm Taylor. Tell me what you want the music to say, and we'll shape it together.`;
  }
  if (teacherId === 'mace') {
    return `Hello, ${firstName}. Sit comfortably, level your wrists, and show me the passage that needs work.`;
  }
  return `Hi, ${firstName}. I'm Aria. Tell me what you want to improve, and we'll work through it one step at a time.`;
}

async function audioStreamToBuffer(stream) {
  if (!stream) throw new Error('Amazon Polly returned no audio stream.');
  if (Buffer.isBuffer(stream)) return stream;
  if (stream instanceof Uint8Array) return Buffer.from(stream);
  if (typeof stream.transformToByteArray === 'function') {
    return Buffer.from(await stream.transformToByteArray());
  }
  const chunks = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

function cacheKey(profile, text) {
  return crypto.createHash('sha256')
    .update(`${profile.engine}\n${profile.voiceId}\n${text}`)
    .digest('hex');
}

function publicProfile(profile) {
  return {
    characterId: profile.id,
    quality: 'generative',
    character: profile.character,
  };
}

function createTeacherSpeechService(environment = process.env, options = {}) {
  const enabled = Boolean(options.client) || enabledFromEnvironment(environment);
  const region = clean(options.region) || speechRegion(environment);
  const client = enabled ? (options.client || new PollyClient({ region })) : null;
  const cache = new Map();
  const now = options.now || (() => Date.now());

  function pruneCache() {
    const timestamp = now();
    for (const [key, entry] of cache) {
      if (entry.expiresAt <= timestamp) cache.delete(key);
    }
    while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  }

  async function synthesize({ teacher, text }) {
    if (!enabled || !client) {
      const error = new Error('Natural teacher speech is not configured on this server.');
      error.code = 'TEACHER_SPEECH_UNAVAILABLE';
      throw error;
    }
    const normalizedText = speechText(text);
    if (!normalizedText) {
      const error = new Error('There is no teacher reply to speak.');
      error.code = 'INVALID_TEACHER_SPEECH';
      throw error;
    }
    const profile = teacherSpeechProfile(teacher);
    const key = cacheKey(profile, normalizedText);
    pruneCache();
    const cached = cache.get(key);
    if (cached && cached.expiresAt > now()) {
      return { ...cached.value, audio: Buffer.from(cached.value.audio), cached: true };
    }

    const input = {
      Engine: profile.engine,
      LanguageCode: profile.languageCode,
      OutputFormat: 'mp3',
      SampleRate: '24000',
      Text: normalizedText,
      TextType: 'text',
      VoiceId: profile.voiceId,
    };
    let response;
    try {
      response = await client.send(new SynthesizeSpeechCommand(input));
    } catch (error) {
      // Neural is still substantially better than the old browser voice and
      // keeps lessons audible during a temporary generative-engine issue.
      if (!['EngineNotSupportedException', 'InvalidParameterValueException'].includes(error?.name)) throw error;
      response = await client.send(new SynthesizeSpeechCommand({
        ...input,
        Engine: 'neural',
        VoiceId: profile.fallbackVoiceId,
      }));
    }
    const audio = await audioStreamToBuffer(response.AudioStream);
    if (!audio.length) throw new Error('Amazon Polly returned an empty audio stream.');
    const value = {
      audio,
      contentType: response.ContentType || 'audio/mpeg',
      requestCharacters: Number(response.RequestCharacters || normalizedText.length),
      profile: publicProfile(profile),
      cached: false,
    };
    cache.set(key, { expiresAt: now() + CACHE_TTL_MS, value });
    pruneCache();
    return value;
  }

  return Object.freeze({
    capabilities: (teacher) => ({
      available: enabled,
      provider: enabled ? 'amazon-polly' : null,
      engine: enabled ? 'generative' : null,
      region: enabled ? region : null,
      profile: teacher ? publicProfile(teacherSpeechProfile(teacher)) : null,
      fallback: 'device-voice',
    }),
    synthesize,
  });
}

module.exports = {
  PROFILE_BY_TEACHER,
  audioStreamToBuffer,
  createTeacherSpeechService,
  speechRegion,
  speechText,
  teacherGreeting,
  teacherSpeechProfile,
};
