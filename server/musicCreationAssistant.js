'use strict';

const {
  DEFAULT_MUSIC_MODEL,
  createOpenAiResponsesClient,
} = require('./openAiResponses');

const PROMPT_VERSION = 'polymath-song-architect-v002-openai';
const FINISHED_STATUSES = new Set(['COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED']);
const LYRIC_SECTIONS = ['verse-1', 'pre-chorus', 'chorus', 'verse-2', 'bridge', 'final-chorus'];
const stringArray = (maximumItems, maximumLength) => ({
  type: 'array',
  items: { type: 'string', maxLength: maximumLength },
  maxItems: maximumItems,
});
const SONG_BLUEPRINT_SCHEMA = Object.freeze({
  type: 'object',
  additionalProperties: false,
  required: [
    'title', 'summary', 'genre', 'mood', 'energy', 'bpm', 'key', 'timeSignature',
    'referenceTraits', 'chordDegrees', 'structure', 'lyrics', 'vocalCoach',
  ],
  properties: {
    title: { type: 'string', maxLength: 120 },
    summary: { type: 'string', maxLength: 300 },
    genre: { type: 'string', maxLength: 80 },
    mood: { type: 'string', maxLength: 80 },
    energy: { type: 'string', enum: ['low', 'medium', 'high'] },
    bpm: { type: 'integer', minimum: 40, maximum: 220 },
    key: { type: 'string', maxLength: 20 },
    timeSignature: { type: 'string', enum: ['3/4', '4/4', '6/8'] },
    referenceTraits: stringArray(8, 120),
    chordDegrees: stringArray(12, 12),
    structure: stringArray(16, 40),
    lyrics: {
      type: 'object',
      additionalProperties: false,
      required: LYRIC_SECTIONS,
      properties: Object.fromEntries(LYRIC_SECTIONS.map((section) => [section, stringArray(12, 90)])),
    },
    vocalCoach: {
      type: 'object',
      additionalProperties: false,
      required: ['comfortableRange', 'delivery', 'breathing', 'practiceSteps'],
      properties: {
        comfortableRange: { type: 'string', maxLength: 30 },
        delivery: stringArray(8, 180),
        breathing: stringArray(8, 180),
        practiceSteps: stringArray(10, 180),
      },
    },
  },
});

function clean(value, maximum = 4000) {
  return String(value || '').trim().slice(0, maximum);
}

function clampNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

function sanitizeBrief(input = {}) {
  const instruments = Array.isArray(input.instruments)
    ? input.instruments.map((item) => clean(item, 50)).filter(Boolean).slice(0, 8)
    : [];
  const sections = Array.isArray(input.sections)
    ? input.sections.map((item) => clean(item, 40)).filter(Boolean).slice(0, 12)
    : [];
  return {
    idea: clean(input.idea, 1800),
    title: clean(input.title, 120),
    genre: clean(input.genre, 80),
    mood: clean(input.mood, 80),
    energy: clean(input.energy, 30),
    bpm: Math.round(clampNumber(input.bpm, 40, 220, 100)),
    key: clean(input.key, 12),
    timeSignature: ['3/4', '4/4', '6/8'].includes(input.timeSignature) ? input.timeSignature : '4/4',
    vocalRange: clean(input.vocalRange, 30),
    reference: clean(input.reference, 300),
    referenceNotes: clean(input.referenceNotes, 800),
    instruments,
    sections,
    existingLyrics: clean(input.existingLyrics, 12000),
    revisionRequest: clean(input.revisionRequest, 1200),
  };
}

function systemPrompt(kind) {
  return [
    'You are Polymath Song Architect, an expert songwriter, arranger, vocal coach, and music-theory assistant.',
    'Your job is to help a human create and perform an original song. The human remains the lead artist.',
    'A reference artist or song is only a source of high-level properties such as tempo range, groove, form, texture, energy, and vocal difficulty.',
    'Never copy, closely paraphrase, or continue protected lyrics. Never reproduce a recognizable melody, hook, voice, or signature passage.',
    'Return only the song blueprint required by the provided structured-output schema.',
    'Use singable lines, deliberate repetition, natural stresses, and a clear emotional progression.',
    'Keep every lyric line under 90 characters and every coaching instruction practical.',
    `Task: ${kind === 'revise' ? 'revise the supplied original draft while preserving unchanged strengths' : 'create an original song blueprint and lyric draft'}.`,
    'Include all lyric sections even when a section contains no lines. Keep chord degrees in Roman-numeral form.',
  ].join('\n');
}

function userPrompt(brief) {
  return [
    'Create from this user brief:',
    JSON.stringify(brief),
    'If information is missing, make restrained musical decisions that fit the stated idea.',
    'Treat reference material as abstract attributes only and keep the composition original.',
  ].join('\n');
}

function textFromOutput(output) {
  if (typeof output === 'string') return output.trim();
  if (Array.isArray(output)) return output.map(textFromOutput).filter(Boolean).join('').trim();
  if (!output || typeof output !== 'object') return '';
  if (typeof output.text === 'string') return output.text.trim();
  if (typeof output?.choices?.[0]?.message?.content === 'string') return output.choices[0].message.content.trim();
  if (output.output !== undefined) return textFromOutput(output.output);
  return '';
}

function extractJson(text) {
  const raw = clean(text, 100000)
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '');
  const first = raw.indexOf('{');
  const last = raw.lastIndexOf('}');
  if (first < 0 || last <= first) throw new Error('The music assistant returned no JSON blueprint.');
  return JSON.parse(raw.slice(first, last + 1));
}

function sanitizeStringArray(value, maximumItems, maximumChars) {
  return Array.isArray(value)
    ? value.map((item) => clean(item, maximumChars)).filter(Boolean).slice(0, maximumItems)
    : [];
}

function sanitizeBlueprint(value, fallbackBrief = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('The music assistant returned an invalid blueprint.');
  }
  const rawLyrics = value.lyrics && typeof value.lyrics === 'object' && !Array.isArray(value.lyrics)
    ? value.lyrics
    : {};
  const lyrics = {};
  Object.entries(rawLyrics).slice(0, 16).forEach(([section, lines]) => {
    const key = clean(section, 40).toLowerCase().replace(/[^a-z0-9-]+/g, '-');
    const safeLines = sanitizeStringArray(lines, 12, 90);
    if (key && safeLines.length) lyrics[key] = safeLines;
  });
  if (!Object.keys(lyrics).length) throw new Error('The music assistant returned no usable lyric sections.');
  const coach = value.vocalCoach && typeof value.vocalCoach === 'object' ? value.vocalCoach : {};
  return {
    title: clean(value.title || fallbackBrief.title || 'Untitled song', 120),
    summary: clean(value.summary, 300),
    genre: clean(value.genre || fallbackBrief.genre || 'Pop', 80),
    mood: clean(value.mood || fallbackBrief.mood || 'Hopeful', 80),
    energy: ['low', 'medium', 'high'].includes(String(value.energy).toLowerCase())
      ? String(value.energy).toLowerCase()
      : clean(fallbackBrief.energy || 'medium', 20),
    bpm: Math.round(clampNumber(value.bpm, 40, 220, fallbackBrief.bpm || 100)),
    key: clean(value.key || fallbackBrief.key || 'C major', 20),
    timeSignature: ['3/4', '4/4', '6/8'].includes(value.timeSignature)
      ? value.timeSignature
      : fallbackBrief.timeSignature || '4/4',
    referenceTraits: sanitizeStringArray(value.referenceTraits, 8, 120),
    chordDegrees: sanitizeStringArray(value.chordDegrees, 12, 12).length
      ? sanitizeStringArray(value.chordDegrees, 12, 12)
      : ['I', 'V', 'vi', 'IV'],
    structure: sanitizeStringArray(value.structure, 16, 40).length
      ? sanitizeStringArray(value.structure, 16, 40)
      : Object.keys(lyrics),
    lyrics,
    vocalCoach: {
      comfortableRange: clean(coach.comfortableRange || fallbackBrief.vocalRange || 'C3-G4', 30),
      delivery: sanitizeStringArray(coach.delivery, 8, 180),
      breathing: sanitizeStringArray(coach.breathing, 8, 180),
      practiceSteps: sanitizeStringArray(coach.practiceSteps, 10, 180),
    },
  };
}

function createMusicCreationAssistant(env = process.env, options = {}) {
  const configured = Boolean(options.client || clean(env.OPENAI_API_KEY, 1000));
  const client = options.client || (configured ? createOpenAiResponsesClient({
    apiKey: env.OPENAI_API_KEY,
    baseUrl: env.OPENAI_BASE_URL,
    organization: env.OPENAI_ORGANIZATION,
    project: env.OPENAI_PROJECT,
    model: env.OPENAI_MUSIC_MODEL || DEFAULT_MUSIC_MODEL,
    reasoningEffort: env.OPENAI_MUSIC_REASONING_EFFORT || 'medium',
    timeoutMs: env.OPENAI_TIMEOUT_MS,
    fetch: options.fetch,
  }) : null);

  function capabilities() {
    return {
      configured,
      provider: configured ? 'OpenAI Responses API' : 'Local song architect',
      model: clean(env.OPENAI_MUSIC_MODEL, 120) || client?.model || DEFAULT_MUSIC_MODEL,
      servedModel: clean(env.OPENAI_MUSIC_MODEL, 120) || client?.model || DEFAULT_MUSIC_MODEL,
      promptVersion: PROMPT_VERSION,
      modelPolicy: 'managed-api-no-project-weights',
    };
  }

  async function submit(kind, input) {
    if (!client) {
      const error = new Error('The Polymath music assistant is not connected to OpenAI yet.');
      error.code = 'MUSIC_CREATION_UNAVAILABLE';
      throw error;
    }
    const brief = sanitizeBrief(input);
    if (!brief.idea && !brief.existingLyrics) {
      const error = new Error('Describe the song idea or add lyrics before asking for AI help.');
      error.code = 'INVALID_MUSIC_CREATION_REQUEST';
      throw error;
    }
    const result = await client.submit([
      { role: 'system', content: systemPrompt(kind) },
      { role: 'user', content: userPrompt(brief) },
    ], {
      reasoning_effort: clean(env.OPENAI_MUSIC_REASONING_EFFORT, 20) || 'medium',
      max_output_tokens: 5200,
      prompt_cache_key: `polymath-song-architect-${PROMPT_VERSION}`,
      metadata: { workload: 'create-music', prompt_version: PROMPT_VERSION, task: kind },
      text: {
        format: {
          type: 'json_schema',
          name: 'polymath_song_blueprint',
          strict: true,
          schema: SONG_BLUEPRINT_SCHEMA,
        },
      },
    });
    return { ...result, brief, promptVersion: PROMPT_VERSION };
  }

  async function status(jobId, fallbackBrief = {}) {
    if (!client) {
      const error = new Error('The Polymath music assistant is not connected to OpenAI yet.');
      error.code = 'MUSIC_CREATION_UNAVAILABLE';
      throw error;
    }
    const body = await client.status(jobId);
    const jobStatus = clean(body?.status, 30).toUpperCase() || 'UNKNOWN';
    let blueprint = null;
    let error = '';
    if (jobStatus === 'COMPLETED') {
      try {
        blueprint = sanitizeBlueprint(extractJson(textFromOutput(body?.output)), fallbackBrief);
      } catch (parseError) {
        error = `The draft finished, but its structure was invalid: ${parseError.message}`;
      }
    } else if (['FAILED', 'TIMED_OUT'].includes(jobStatus)) {
      error = clean(body?.error || body?.output?.error, 600) || 'OpenAI could not complete this music draft.';
    }
    return {
      id: clean(body?.id || jobId, 160),
      status: jobStatus,
      active: ['IN_QUEUE', 'IN_PROGRESS'].includes(jobStatus),
      finished: FINISHED_STATUSES.has(jobStatus),
      blueprint,
      error,
      delayTime: Number(body?.delayTime) || 0,
      executionTime: Number(body?.executionTime) || 0,
    };
  }

  async function cancel(jobId) {
    if (!client) {
      const error = new Error('The Polymath music assistant is not connected to OpenAI yet.');
      error.code = 'MUSIC_CREATION_UNAVAILABLE';
      throw error;
    }
    return client.cancel(jobId);
  }

  return Object.freeze({ capabilities, submit, status, cancel });
}

module.exports = {
  PROMPT_VERSION,
  SONG_BLUEPRINT_SCHEMA,
  createMusicCreationAssistant,
  extractJson,
  sanitizeBlueprint,
  sanitizeBrief,
};
