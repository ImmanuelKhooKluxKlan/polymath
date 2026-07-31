const NOTE_TO_SEMITONE = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11,
};

const SEMITONE_TO_NOTE = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
];

const CHORD_PATTERNS = {
  major: [0, 4, 7],
  minor: [0, 3, 7],
  diminished: [0, 3, 6],
  augmented: [0, 4, 8],
  sus2: [0, 2, 7],
  sus4: [0, 5, 7],
  maj7: [0, 4, 7, 11],
  minor7: [0, 3, 7, 10],
  dominant7: [0, 4, 7, 10],
};

const CHORD_REGEX = /\b([A-G](?:#|b)?)(m|maj7|m7|7|dim|aug|sus2|sus4)?\b/g;

function normalizeChordType(type = '') {
  if (type === 'm') return 'minor';
  if (type === 'm7') return 'minor7';
  if (type === 'maj7') return 'maj7';
  if (type === '7') return 'dominant7';
  if (type === 'dim') return 'diminished';
  if (type === 'aug') return 'augmented';
  if (type === 'sus2') return 'sus2';
  if (type === 'sus4') return 'sus4';
  return 'major';
}

function noteFromSemitone(semitone, octave = 4) {
  const wrapped = ((semitone % 12) + 12) % 12;
  return `${SEMITONE_TO_NOTE[wrapped]}${octave}`;
}

function chordToNotes(root, type) {
  const rootValue = NOTE_TO_SEMITONE[root];

  if (rootValue === undefined) {
    return [];
  }

  const chordType = normalizeChordType(type);
  const intervals = CHORD_PATTERNS[chordType] || CHORD_PATTERNS.major;

  const baseOctave = rootValue >= 9 ? 3 : 4;

  return intervals.map((interval) => {
    const semitone = rootValue + interval;
    const octave = baseOctave + Math.floor(semitone / 12);
    return noteFromSemitone(semitone, octave);
  });
}

export function parseLyricsAndChordsToSong(text, options = {}) {
  const secondsPerChord = options.secondsPerChord || 1.5;
  const noteDuration = options.noteDuration || 1.25;
  const velocity = options.velocity || 0.8;

  const lines = text.split(/\r?\n/);
  const notes = [];
  const detectedChords = [];

  let currentTime = 0;

  for (const line of lines) {
    const matches = [...line.matchAll(CHORD_REGEX)];

    if (matches.length === 0) {
      continue;
    }

    for (const match of matches) {
      const root = match[1];
      const type = match[2] || '';
      const chordName = `${root}${type}`;
      const chordNotes = chordToNotes(root, type);

      if (chordNotes.length === 0) {
        continue;
      }

      detectedChords.push(chordName);

      chordNotes.forEach((note) => {
        notes.push({
          note,
          time: Number(currentTime.toFixed(2)),
          duration: noteDuration,
          velocity,
          source: chordName,
        });
      });

      currentTime += secondsPerChord;
    }
  }

  return {
    title: 'Uploaded Lyrics Chord Sheet',
    tempo: 80,
    timeSignature: '4/4',
    detectedChords,
    notes,
  };
}