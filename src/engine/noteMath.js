export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const FLAT_NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
export const WHITE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];

export const BLACK_NOTE_OFFSETS = {
  'C#': 0.72,
  'D#': 1.72,
  'F#': 3.72,
  'G#': 4.72,
  'A#': 5.72,
};

const NATURAL_MIDI_INDEX = {
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
  A: 9,
  B: 11,
};

function positiveMod(value, mod) {
  return ((value % mod) + mod) % mod;
}

export function parseNote(note) {
  const text = String(note).trim().replace('♭', 'b').replace('♯', '#');
  const match = /^([A-G])(#|b)?(-?\d+)$/.exec(text);

  if (!match) {
    throw new Error(`Invalid note: ${note}`);
  }

  const [, letter, accidental = '', octaveText] = match;
  const octave = Number(octaveText);

  if (!Number.isFinite(octave)) {
    throw new Error(`Invalid note octave: ${note}`);
  }

  let semitone = NATURAL_MIDI_INDEX[letter];

  if (accidental === '#') {
    semitone += 1;
  } else if (accidental === 'b') {
    semitone -= 1;
  }

  const midi = (octave + 1) * 12 + semitone;
  const normalizedIndex = positiveMod(midi, 12);
  const normalizedOctave = Math.floor(midi / 12) - 1;
  const normalizedName = `${NOTE_NAMES[normalizedIndex]}${normalizedOctave}`;

  return {
    letter,
    accidental,
    octave,
    midi,
    name: normalizedName,
  };
}

export function midiToNote(midi) {
  const roundedMidi = Math.round(Number(midi));

  if (!Number.isFinite(roundedMidi)) {
    throw new Error(`Invalid midi note: ${midi}`);
  }

  const octave = Math.floor(roundedMidi / 12) - 1;
  const name = NOTE_NAMES[positiveMod(roundedMidi, 12)];

  return `${name}${octave}`;
}

export function noteToFrequency(note) {
  const { midi } = parseNote(note);

  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function buildPianoRange(start = 'C2', end = 'C6') {
  const startMidi = parseNote(start).midi;
  const endMidi = parseNote(end).midi;
  const notes = [];

  for (let midi = startMidi; midi <= endMidi; midi += 1) {
    const label = midiToNote(midi);
    const pitch = NOTE_NAMES[positiveMod(midi, 12)];

    notes.push({
      note: label,
      midi,
      pitch,
      isBlack: pitch.includes('#'),
    });
  }

  return notes;
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function midiToFlatName(midi) {
  const roundedMidi = Math.round(Number(midi));

  if (!Number.isFinite(roundedMidi)) {
    throw new Error(`Invalid midi note: ${midi}`);
  }

  const octave = Math.floor(roundedMidi / 12) - 1;
  const name = FLAT_NOTE_NAMES[positiveMod(roundedMidi, 12)];
  return `${name}${octave}`;
}

export function noteToDisplayName(noteOrMidi, preferFlats = true) {
  const midi = typeof noteOrMidi === 'number' ? noteOrMidi : parseNote(noteOrMidi).midi;
  return preferFlats ? midiToFlatName(midi) : midiToNote(midi);
}

export function isBlackMidi(midi) {
  return NOTE_NAMES[positiveMod(midi, 12)].includes('#');
}
