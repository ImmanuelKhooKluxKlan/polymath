const NATURAL_QUALITY = /\b(natural|neural|premium|enhanced|studio|online)\b/i;
const ROBOTIC_OR_NOVELTY = /\b(albert|bad news|bahh|bells|boing|bubbles|cellos|deranged|hysterical|pipe organ|trinoids|whisper|zarvox)\b/i;
const FEMININE_NAMES = /\b(aria|ava|emma|jenny|karen|moira|samantha|serena|susan|tessa|victoria|zira|female)\b/i;
const MASCULINE_NAMES = /\b(aaron|alex|daniel|david|fred|george|guy|mark|oliver|ryan|male)\b/i;

const PROFILE_BY_TEACHER = Object.freeze({
  aria: Object.freeze({ voiceType: 'feminine', rate: 0.94, pitch: 1.02 }),
  nova: Object.freeze({ voiceType: 'feminine', rate: 0.96, pitch: 1.05 }),
  anakin: Object.freeze({ voiceType: 'masculine', rate: 0.96, pitch: 0.92 }),
  taylor: Object.freeze({ voiceType: 'feminine', rate: 0.98, pitch: 1.04 }),
  mace: Object.freeze({ voiceType: 'masculine', rate: 0.88, pitch: 0.82 }),
});

function languageParts(locale) {
  const normalized = String(locale || 'en-US').replace('_', '-').toLowerCase();
  return { exact: normalized, base: normalized.split('-')[0] || 'en' };
}

export function teacherVoiceProfile(teacher = {}) {
  const fallback = PROFILE_BY_TEACHER[teacher.id] || PROFILE_BY_TEACHER.aria;
  const suppliedType = String(teacher.voiceType || '').toLowerCase();
  return {
    ...fallback,
    voiceType: ['feminine', 'masculine', 'neutral'].includes(suppliedType)
      ? suppliedType
      : fallback.voiceType,
  };
}

export function teacherVoiceScore(voice, teacher = {}, locale = 'en-US', preferredVoiceUri = '') {
  if (!voice) return Number.NEGATIVE_INFINITY;
  const profile = teacherVoiceProfile(teacher);
  const desired = languageParts(locale);
  const candidate = languageParts(voice.lang);
  const name = `${voice.name || ''} ${voice.voiceURI || ''}`;
  let score = 0;
  if (preferredVoiceUri && voice.voiceURI === preferredVoiceUri) score += 1000;
  if (candidate.exact === desired.exact) score += 90;
  else if (candidate.base === desired.base) score += 55;
  else score -= 80;
  if (NATURAL_QUALITY.test(name)) score += 70;
  if (/\b(google|microsoft|apple)\b/i.test(name)) score += 18;
  if (voice.localService) score += 5;
  if (voice.default) score += 4;
  if (profile.voiceType === 'feminine' && FEMININE_NAMES.test(name)) score += 32;
  if (profile.voiceType === 'masculine' && MASCULINE_NAMES.test(name)) score += 32;
  if (profile.voiceType === 'feminine' && MASCULINE_NAMES.test(name)) score -= 16;
  if (profile.voiceType === 'masculine' && FEMININE_NAMES.test(name)) score -= 16;
  if (ROBOTIC_OR_NOVELTY.test(name)) score -= 120;
  return score;
}

export function selectTeacherVoice(voices, teacher, locale, preferredVoiceUri = '') {
  return [...(Array.isArray(voices) ? voices : [])]
    .sort((left, right) => (
      teacherVoiceScore(right, teacher, locale, preferredVoiceUri)
      - teacherVoiceScore(left, teacher, locale, preferredVoiceUri)
    ))[0] || null;
}

export function speechReadyText(value) {
  return String(value || '')
    .replace(/```[\s\S]*?```/g, ' code example ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)]\([^\s)]+\)/g, '$1')
    .replace(/\b([A-G])#(-?\d+)\b/g, '$1 sharp $2')
    .replace(/\b([A-G])b(-?\d+)\b/g, '$1 flat $2')
    .replace(/[•*_#>]/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .trim();
}

export function speechSegments(value, maximumCharacters = 220) {
  const text = speechReadyText(value);
  if (!text) return [];
  const maximum = Math.max(80, Math.min(500, Number(maximumCharacters) || 220));
  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g)?.map((part) => part.trim()).filter(Boolean) || [text];
  const segments = [];
  for (const sentence of sentences) {
    if (sentence.length <= maximum) {
      const previous = segments[segments.length - 1];
      if (previous && `${previous} ${sentence}`.length <= maximum) segments[segments.length - 1] = `${previous} ${sentence}`;
      else segments.push(sentence);
      continue;
    }
    const words = sentence.split(/\s+/);
    let current = '';
    words.forEach((word) => {
      if (current && `${current} ${word}`.length > maximum) {
        segments.push(current);
        current = word;
      } else current = current ? `${current} ${word}` : word;
    });
    if (current) segments.push(current);
  }
  return segments;
}
