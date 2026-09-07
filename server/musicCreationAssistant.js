'use strict';

const { createChatBossRunpodClient } = require('./chatBossRunpod');

const PROMPT_VERSION = 'polymath-song-architect-v001';
const FINISHED_STATUSES = new Set(['COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED']);

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
    'Return one strict JSON object only. Do not add markdown, analysis, preambles, or text outside JSON.',
    'Use singable lines, deliberate repetition, natural stresses, and a clear emotional progression.',
    'Keep every lyric line under 90 characters and every coaching instruction practical.',
    `Task: ${kind === 'revise' ? 'revise the supplied original draft while preserving unchanged strengths' : 'create an original song blueprint and lyric draft'}.`,
    'Required JSON schema:',
    JSON.stringify({
      title: 'Original song title',
      summary: 'One-sentence artistic direction',
      genre: 'genre',
      mood: 'mood',
      energy: 'low|medium|high',
      bpm: 100,
      key: 'C major',
      timeSignature: '4/4',
      referenceTraits: ['high-level trait only'],
      chordDegrees: ['I', 'V', 'vi', 'IV'],
      structure: ['intro', 'verse-1', 'pre-chorus', 'chorus', 'verse-2', 'chorus', 'bridge', 'final-chorus', 'outro'],
      lyrics: {
        'verse-1': ['line one', 'line two', 'line three', 'line four'],
        'pre-chorus': ['line one', 'line two'],
        chorus: ['line one', 'line two', 'line three', 'line four'],
        'verse-2': ['line one', 'line two', 'line three', 'line four'],
        bridge: ['line one', 'line two', 'line three', 'line four'],
        'final-chorus': ['line one', 'line two', 'line three', 'line four'],
      },
      vocalCoach: {
        comfortableRange: 'C3-G4',
        delivery: ['short performance direction'],
        breathing: ['where or how to breathe'],
        practiceSteps: ['specific rehearsal step'],
      },
    }),
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
  const endpointId = clean(env.RUNPOD_POLYMATH_CREATE_ENDPOINT_ID, 160);
  const configured = Boolean(options.client || (endpointId && clean(env.RUNPOD_API_KEY, 1000)));
  const client = options.client || (configured ? createChatBossRunpodClient({
    endpointId,
    apiKey: env.RUNPOD_API_KEY,
    model: env.RUNPOD_POLYMATH_CREATE_MODEL || 'polymath-create-deepseek-v4',
    timeoutMs: env.RUNPOD_POLYMATH_CREATE_TIMEOUT_MS || 20 * 60 * 1000,
    fetch: options.fetch,
  }) : null);

  function capabilities() {
    return {
      configured,
      provider: configured ? 'RunPod Serverless' : 'Local song architect',
      model: clean(env.RUNPOD_POLYMATH_CREATE_DISPLAY_MODEL, 120) || 'DeepSeek V4 (private inference source)',
      servedModel: clean(env.RUNPOD_POLYMATH_CREATE_MODEL, 120) || 'polymath-create-deepseek-v4',
      promptVersion: PROMPT_VERSION,
      originalCheckpointPolicy: 'read-only',
    };
  }

  async function submit(kind, input) {
    if (!client) {
      const error = new Error('The Polymath music assistant is not connected to its RunPod endpoint yet.');
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
      temperature: kind === 'revise' ? 0.65 : 0.82,
      top_p: 0.9,
      max_tokens: 3600,
    });
    return { ...result, brief, promptVersion: PROMPT_VERSION };
  }

  async function status(jobId, fallbackBrief = {}) {
    if (!client) {
      const error = new Error('The Polymath music assistant is not connected to its RunPod endpoint yet.');
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
      error = clean(body?.error || body?.output?.error, 600) || 'RunPod could not complete this music draft.';
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
      const error = new Error('The Polymath music assistant is not connected to its RunPod endpoint yet.');
      error.code = 'MUSIC_CREATION_UNAVAILABLE';
      throw error;
    }
    return client.cancel(jobId);
  }

  return Object.freeze({ capabilities, submit, status, cancel });
}

module.exports = {
  PROMPT_VERSION,
  createMusicCreationAssistant,
  extractJson,
  sanitizeBlueprint,
  sanitizeBrief,
};
