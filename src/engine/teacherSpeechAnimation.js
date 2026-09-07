const REST_FRAME = Object.freeze({
  active: false,
  charIndex: 0,
  viseme: 'rest',
  intensity: 0,
});

const VISEME_INTENSITY = Object.freeze({
  rest: 0,
  closed: 0.12,
  neutral: 0.4,
  teeth: 0.48,
  narrow: 0.56,
  wide: 0.68,
  round: 0.78,
  open: 0.92,
});

function boundedIndex(text, value) {
  if (!text.length) return 0;
  const number = Number(value);
  return Math.max(0, Math.min(text.length - 1, Number.isFinite(number) ? Math.floor(number) : 0));
}

function nextSpokenIndex(text, startIndex) {
  let index = boundedIndex(text, startIndex);
  while (index < text.length && /\s/.test(text[index])) index += 1;
  return Math.min(index, Math.max(0, text.length - 1));
}

export function spokenTokenLength(text, startIndex = 0) {
  const value = String(text || '');
  if (!value) return 0;
  const start = nextSpokenIndex(value, startIndex);
  const token = value.slice(start).match(/^[^\s]+/u)?.[0] || '';
  return token.length;
}

export function teacherVisemeAt(text, requestedIndex = 0) {
  const value = String(text || '').toLowerCase();
  if (!value) return 'rest';
  const index = nextSpokenIndex(value, requestedIndex);
  const character = value[index] || '';
  const pair = value.slice(index, index + 2);

  if (!/[a-z0-9]/i.test(character)) return 'rest';
  if (/^(m|b|p)/.test(pair)) return 'closed';
  if (/^(f|v|th)/.test(pair)) return 'teeth';
  if (/^(sh|ch|j|zh)/.test(pair)) return 'narrow';
  if (/^(oo|ou|ow|o|u|w|q|r)/.test(pair)) return 'round';
  if (/^(ee|ea|ie|e|i|y)/.test(pair)) return 'wide';
  if (/^(a|ah|ai)/.test(pair)) return 'open';
  if (/^(s|z|x|t|d|n|l|k|g)/.test(pair)) return 'teeth';
  return 'neutral';
}

/**
 * Predict the frame between Web Speech boundary events. Every real boundary
 * resets the clock, so prediction drift cannot accumulate across words.
 */
export function speechAnimationFrame({
  text,
  speaking = true,
  boundaryIndex = 0,
  boundaryLength = 0,
  boundaryAtMs = 0,
  nowMs = 0,
  rate = 1,
} = {}) {
  const value = String(text || '');
  if (!speaking || !value) return REST_FRAME;
  const start = nextSpokenIndex(value, boundaryIndex);
  const suppliedLength = Math.max(0, Math.floor(Number(boundaryLength) || 0));
  const length = suppliedLength || spokenTokenLength(value, start) || value.length;
  const elapsed = Math.max(0, Number(nowMs) - Number(boundaryAtMs));
  const characterMilliseconds = Math.max(38, Math.min(105, 68 / Math.max(0.55, Number(rate) || 1)));
  const advance = Math.floor(elapsed / characterMilliseconds);
  if (advance >= length) return { ...REST_FRAME, charIndex: Math.min(value.length - 1, start + length) };
  const charIndex = boundedIndex(value, start + advance);
  const viseme = teacherVisemeAt(value, charIndex);
  return {
    active: viseme !== 'rest',
    charIndex,
    viseme,
    intensity: VISEME_INTENSITY[viseme] ?? VISEME_INTENSITY.neutral,
  };
}

export function restingSpeechFrame() {
  return { ...REST_FRAME };
}
