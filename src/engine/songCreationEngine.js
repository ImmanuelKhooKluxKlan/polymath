const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const ROOTS = {
  C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, F: 5,
  'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11,
};
const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10];
const DEGREE_INDEX = { I: 0, II: 1, III: 2, IV: 3, V: 4, VI: 5, VII: 6 };
const DEFAULT_STRUCTURE = ['intro', 'verse-1', 'pre-chorus', 'chorus', 'verse-2', 'chorus', 'bridge', 'final-chorus', 'outro'];

export const CREATE_MUSIC_GENRES = [
  'Pop', 'Acoustic pop', 'R&B', 'Indie', 'Rock', 'Country', 'Dance pop',
  'Hip-hop', 'Soul', 'Jazz pop', 'Cinematic', 'Electronic',
];

export const CREATE_MUSIC_MOODS = [
  'Hopeful', 'Romantic', 'Bittersweet', 'Confident', 'Dreamy', 'Joyful',
  'Heartbroken', 'Peaceful', 'Playful', 'Dark', 'Triumphant',
];

export const CREATE_MUSIC_STYLES = Object.freeze([
  {
    id: 'catchy-acoustic',
    label: 'Catchy acoustic',
    description: 'Warm strums, a clear hook and a piano melody that leaves room for your voice.',
    genre: 'Acoustic pop', mood: 'Hopeful', energy: 'medium', bpm: 106,
    instruments: ['acoustic-guitar', 'piano', 'upright-bass', 'drums'],
    pianoPattern: 'flowing', groove: 0.035,
  },
  {
    id: 'intimate-piano',
    label: 'Intimate piano',
    description: 'Soft felt-style piano, spacious phrases and restrained dynamics.',
    genre: 'Pop', mood: 'Bittersweet', energy: 'low', bpm: 76,
    instruments: ['piano', 'upright-bass', 'violin'],
    pianoPattern: 'ballad', groove: 0.012,
  },
  {
    id: 'bright-pop',
    label: 'Bright pop hook',
    description: 'Punchy rhythm, short memorable phrases and a lifted chorus.',
    genre: 'Dance pop', mood: 'Joyful', energy: 'high', bpm: 122,
    instruments: ['piano', 'synth', 'upright-bass', 'drums'],
    pianoPattern: 'syncopated', groove: 0.026,
  },
  {
    id: 'dreamy-indie',
    label: 'Dreamy indie',
    description: 'Wide arpeggios, gentle motion and an airy late-night atmosphere.',
    genre: 'Indie', mood: 'Dreamy', energy: 'low', bpm: 88,
    instruments: ['piano', 'electric-guitar', 'upright-bass', 'violin'],
    pianoPattern: 'open', groove: 0.02,
  },
  {
    id: 'soulful-rnb',
    label: 'Soulful R&B',
    description: 'Laid-back pocket, warmer voicings and room for expressive vocals.',
    genre: 'R&B', mood: 'Romantic', energy: 'medium', bpm: 82,
    instruments: ['piano', 'synth', 'upright-bass', 'drums'],
    pianoPattern: 'pocket', groove: 0.055,
  },
  {
    id: 'cinematic-rise',
    label: 'Cinematic rise',
    description: 'A spacious opening that grows into a broad, emotional final chorus.',
    genre: 'Cinematic', mood: 'Triumphant', energy: 'high', bpm: 96,
    instruments: ['piano', 'upright-bass', 'drums', 'violin', 'flute'],
    pianoPattern: 'cinematic', groove: 0.014,
  },
]);

export const CREATE_MUSIC_VOICES = Object.freeze([
  {
    id: 'airy-soprano', label: 'Airy soprano', description: 'Light and floating',
    defaultRange: 'C4-G5',
    instrument: 'vocal-airy', speechPitch: 1.16, speechRate: 0.94,
    preferredNames: ['samantha', 'aria', 'jenny', 'karen', 'tessa', 'zira'],
  },
  {
    id: 'warm-alto', label: 'Warm alto', description: 'Close and expressive',
    defaultRange: 'G3-D5',
    instrument: 'vocal-warm', speechPitch: 1.02, speechRate: 0.91,
    preferredNames: ['ava', 'serena', 'sonia', 'salli', 'moira', 'victoria'],
  },
  {
    id: 'clear-tenor', label: 'Clear tenor', description: 'Bright and direct',
    defaultRange: 'C3-G4',
    instrument: 'vocal-clear', speechPitch: 0.92, speechRate: 0.96,
    preferredNames: ['daniel', 'ryan', 'aaron', 'guy', 'alex'],
  },
  {
    id: 'rich-baritone', label: 'Rich baritone', description: 'Deep and grounded',
    defaultRange: 'G2-D4',
    instrument: 'vocal-rich', speechPitch: 0.78, speechRate: 0.88,
    preferredNames: ['george', 'brian', 'tom', 'fred', 'ralph'],
  },
]);

const STYLE_BY_ID = Object.fromEntries(CREATE_MUSIC_STYLES.map((style) => [style.id, style]));
const VOICE_BY_ID = Object.fromEntries(CREATE_MUSIC_VOICES.map((voice) => [voice.id, voice]));

export const CREATE_MUSIC_INSTRUMENTS = [
  { id: 'piano', label: 'Piano', role: 'harmony' },
  { id: 'synth', label: 'Synth', role: 'harmony' },
  { id: 'acoustic-guitar', label: 'Acoustic guitar', role: 'harmony' },
  { id: 'electric-guitar', label: 'Electric guitar', role: 'harmony' },
  { id: 'upright-bass', label: 'Bass', role: 'bass' },
  { id: 'drums', label: 'Drums', role: 'rhythm' },
  { id: 'violin', label: 'Strings', role: 'texture' },
  { id: 'flute', label: 'Flute', role: 'texture' },
];

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function hashString(value) {
  let hash = 2166136261;
  for (const character of String(value || '')) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed) {
  let state = seed || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function titleFromIdea(idea) {
  const words = String(idea || '')
    .replace(/[^a-z0-9' -]/gi, ' ')
    .split(/\s+/)
    .filter((word) => word.length > 2)
    .slice(0, 4);
  return words.length
    ? words.map((word) => word[0].toUpperCase() + word.slice(1).toLowerCase()).join(' ')
    : 'Untitled Song';
}

function defaultLyrics() {
  return {
    'verse-1': ['Write the scene that starts your story', 'Show us one detail only you would know'],
    'pre-chorus': ['Raise the question that pulls us forward', 'Leave a breath before the answer'],
    chorus: ['Write the line you want people to remember', 'Repeat the heart of your song'],
    'verse-2': ['Change the scene and raise the stakes', 'Reveal what the first verse could not say'],
    bridge: ['Turn the meaning in a new direction', 'Say the truth with different words'],
    'final-chorus': ['Return to your unforgettable line', 'Change one detail so the ending feels earned'],
  };
}

export function createBlankMusicProject() {
  return {
    id: '',
    title: '',
    status: 'draft',
    brief: {
      idea: '',
      genre: 'Pop',
      mood: 'Hopeful',
      energy: 'medium',
      bpm: 104,
      key: 'C major',
      timeSignature: '4/4',
      vocalRange: 'G3-D5',
      reference: '',
      referenceNotes: '',
      stylePreset: 'catchy-acoustic',
      singerVoice: 'warm-alto',
      instruments: ['acoustic-guitar', 'piano', 'upright-bass', 'drums'],
      structure: DEFAULT_STRUCTURE,
    },
    lyrics: defaultLyrics(),
    vocalCoach: {
      comfortableRange: 'C3-G4',
      delivery: ['Speak the lyric naturally before singing it.'],
      breathing: ['Mark a breath at the end of each complete thought.'],
      practiceSteps: ['Clap the rhythm.', 'Hum the guide melody.', 'Sing with the guide.', 'Mute the guide and perform.'],
    },
    referenceTraits: [],
    chordDegrees: ['I', 'V', 'vi', 'IV'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

export function applySongStyle(projectInput, styleId) {
  const project = cloneJson(projectInput || createBlankMusicProject());
  const style = STYLE_BY_ID[styleId] || STYLE_BY_ID['catchy-acoustic'];
  project.brief = {
    ...project.brief,
    stylePreset: style.id,
    genre: style.genre,
    mood: style.mood,
    energy: style.energy,
    bpm: style.bpm,
    instruments: [...style.instruments],
  };
  project.updatedAt = new Date().toISOString();
  return project;
}

export function musicVoiceProfile(voiceId) {
  return VOICE_BY_ID[voiceId] || VOICE_BY_ID['warm-alto'];
}

export function applySongBlueprint(currentProject, blueprint) {
  const next = cloneJson(currentProject || createBlankMusicProject());
  next.title = blueprint.title || next.title || titleFromIdea(next.brief.idea);
  next.brief = {
    ...next.brief,
    genre: blueprint.genre || next.brief.genre,
    mood: blueprint.mood || next.brief.mood,
    energy: blueprint.energy || next.brief.energy,
    bpm: clamp(Math.round(Number(blueprint.bpm) || next.brief.bpm), 40, 220),
    key: blueprint.key || next.brief.key,
    timeSignature: blueprint.timeSignature || next.brief.timeSignature,
    structure: Array.isArray(blueprint.structure) && blueprint.structure.length
      ? blueprint.structure
      : next.brief.structure,
  };
  next.summary = blueprint.summary || next.summary || '';
  next.referenceTraits = Array.isArray(blueprint.referenceTraits) ? blueprint.referenceTraits : [];
  next.chordDegrees = Array.isArray(blueprint.chordDegrees) && blueprint.chordDegrees.length
    ? blueprint.chordDegrees
    : next.chordDegrees;
  next.lyrics = blueprint.lyrics && typeof blueprint.lyrics === 'object'
    ? blueprint.lyrics
    : next.lyrics;
  next.vocalCoach = blueprint.vocalCoach || next.vocalCoach;
  next.updatedAt = new Date().toISOString();
  return next;
}

function midiToNote(midi) {
  const safe = clamp(Math.round(midi), 0, 127);
  return `${NOTE_NAMES[safe % 12]}${Math.floor(safe / 12) - 1}`;
}

function parseKey(key) {
  const [rootText = 'C', modeText = 'major'] = String(key || 'C major').trim().split(/\s+/);
  const normalizedRoot = rootText.length > 1
    ? `${rootText[0].toUpperCase()}${rootText.slice(1)}`
    : rootText.toUpperCase();
  return {
    root: ROOTS[normalizedRoot] ?? 0,
    label: normalizedRoot,
    minor: modeText.toLowerCase().startsWith('min'),
  };
}

function romanDegree(value) {
  const raw = String(value || 'I').replace(/[^ivIV]/g, '');
  const upper = raw.toUpperCase();
  return { index: DEGREE_INDEX[upper] ?? 0, minor: raw && raw === raw.toLowerCase() };
}

function chordForDegree(key, degreeText, octave = 3) {
  const scale = key.minor ? MINOR_SCALE : MAJOR_SCALE;
  const degree = romanDegree(degreeText);
  const root = 12 * (octave + 1) + key.root + scale[degree.index];
  const naturalThird = scale[(degree.index + 2) % 7] + ((degree.index + 2) >= 7 ? 12 : 0);
  const naturalFifth = scale[(degree.index + 4) % 7] + ((degree.index + 4) >= 7 ? 12 : 0);
  const rootScale = scale[degree.index];
  let third = root + (naturalThird - rootScale);
  const fifth = root + (naturalFifth - rootScale);
  if (degree.minor && third - root === 4) third -= 1;
  return { root, notes: [root, third, fifth], label: `${key.label}${degreeText}` };
}

function barsForSection(section, lyricLines) {
  if (/intro|outro/i.test(section)) return 4;
  if (/pre/i.test(section)) return 4;
  if (/chorus|bridge/i.test(section)) return Math.max(4, Math.min(8, lyricLines.length * 2));
  return Math.max(4, Math.min(8, lyricLines.length * 2));
}

function sectionLyrics(project, section) {
  const exact = project.lyrics?.[section];
  if (Array.isArray(exact)) return exact.filter(Boolean);
  if (section === 'final-chorus' && Array.isArray(project.lyrics?.chorus)) return project.lyrics.chorus.filter(Boolean);
  return [];
}

function lyricUnits(line) {
  const words = String(line || '').trim().split(/\s+/).filter(Boolean);
  return words.flatMap((word) => {
    const cleaned = word.replace(/[^a-z']/gi, '') || word;
    const vowelGroups = cleaned.match(/[aeiouy]+/gi)?.length || 1;
    return Array.from({ length: clamp(vowelGroups, 1, 4) }, (_, index) => ({
      text: index === 0 ? word : '·',
      word,
      continuation: index > 0,
    }));
  });
}

function addTrackNote(track, midi, time, duration, velocity, extra = {}) {
  track.events.push({
    note: midiToNote(midi),
    midi: Math.round(midi),
    time: Number(time.toFixed(4)),
    duration: Number(Math.max(0.05, duration).toFixed(4)),
    audioDuration: Number(Math.max(0.05, duration).toFixed(4)),
    velocity: Number(clamp(velocity, 0.05, 1).toFixed(3)),
    ...extra,
  });
}

function makeTrack(id, label, instrument, role, colour) {
  return { id, label, instrument, role, colour, events: [] };
}

function activeStyle(project) {
  const selected = STYLE_BY_ID[project.brief?.stylePreset];
  if (selected) return selected;
  const genre = String(project.brief?.genre || '').toLowerCase();
  if (genre.includes('acoustic') || genre.includes('country')) return STYLE_BY_ID['catchy-acoustic'];
  if (genre.includes('r&b') || genre.includes('soul')) return STYLE_BY_ID['soulful-rnb'];
  if (genre.includes('indie')) return STYLE_BY_ID['dreamy-indie'];
  if (genre.includes('cinematic')) return STYLE_BY_ID['cinematic-rise'];
  return STYLE_BY_ID['bright-pop'];
}

function sectionFamily(sectionName) {
  const value = String(sectionName || '').toLowerCase();
  if (value.includes('chorus')) return 'chorus';
  if (value.includes('pre')) return 'pre-chorus';
  if (value.includes('bridge')) return 'bridge';
  if (value.includes('intro') || value.includes('outro')) return 'instrumental';
  return 'verse';
}

function voiceLedChord(chord, previousVoicing = null) {
  const candidates = [];
  for (let inversion = 0; inversion < chord.notes.length; inversion += 1) {
    const rotated = [
      ...chord.notes.slice(inversion),
      ...chord.notes.slice(0, inversion).map((midi) => midi + 12),
    ];
    for (const shift of [-12, 0, 12]) {
      const notes = rotated.map((midi) => midi + shift);
      if (notes[0] < 48 || notes[notes.length - 1] > 76) continue;
      const movement = previousVoicing
        ? notes.reduce((total, midi, index) => total + Math.abs(midi - previousVoicing[index]), 0)
        : Math.abs(notes.reduce((total, midi) => total + midi, 0) / notes.length - 61);
      candidates.push({ notes, movement });
    }
  }
  return candidates.sort((left, right) => left.movement - right.movement)[0]?.notes
    || chord.notes.map((midi) => midi + (midi < 48 ? 12 : 0));
}

function humanizedTime(time, random, amount, floor = 0) {
  return Math.max(floor, time + (random() - 0.5) * amount);
}

function pianoPatternOffsets(pattern, beatsPerBar) {
  if (pattern === 'syncopated') return [0, 1.5, 2, 3.5].filter((beat) => beat < beatsPerBar);
  if (pattern === 'pocket') return [0, 0.75, 2, 2.75].filter((beat) => beat < beatsPerBar);
  if (pattern === 'cinematic') return [0, Math.max(1, beatsPerBar / 2)];
  if (pattern === 'ballad') return Array.from({ length: beatsPerBar }, (_, index) => index);
  return Array.from({ length: beatsPerBar * 2 }, (_, index) => index / 2);
}

function addPianoBar(track, chord, voicing, time, barSeconds, beatUnit, beatsPerBar, style, sectionName, random) {
  const family = sectionFamily(sectionName);
  const lift = family === 'chorus' ? 0.09 : family === 'bridge' ? 0.045 : 0;
  const leftRoot = chord.root - 12;
  const leftFifth = chord.notes[2] - 12;
  addTrackNote(track, leftRoot, time, barSeconds * 0.72, 0.43 + lift, {
    hand: 'left', scoreRole: 'accompaniment', articulation: 'sustain',
  });
  if (style.pianoPattern !== 'ballad' && beatsPerBar >= 4) {
    addTrackNote(track, leftFifth, time + beatUnit * 2, beatUnit * 1.45, 0.34 + lift, {
      hand: 'left', scoreRole: 'accompaniment', articulation: 'sustain',
    });
  }

  const offsets = pianoPatternOffsets(style.pianoPattern, beatsPerBar);
  if (['syncopated', 'pocket', 'cinematic'].includes(style.pianoPattern)) {
    offsets.forEach((beat, hitIndex) => {
      voicing.forEach((midi, voiceIndex) => addTrackNote(
        track,
        midi,
        humanizedTime(time + beat * beatUnit + voiceIndex * 0.012, random, style.groove, time),
        beatUnit * (style.pianoPattern === 'cinematic' ? 1.65 : 0.58),
        0.37 + lift + (hitIndex === 0 ? 0.055 : 0) + voiceIndex * 0.012,
        { hand: 'right', scoreRole: 'accompaniment', articulation: 'connected' },
      ));
    });
    return;
  }

  offsets.forEach((beat, noteIndex) => {
    const sequence = style.pianoPattern === 'ballad' ? [0, 2, 1, 2] : [0, 1, 2, 1, 0, 1, 2, 1];
    const midi = voicing[sequence[noteIndex % sequence.length] % voicing.length];
    addTrackNote(
      track,
      midi,
      humanizedTime(time + beat * beatUnit, random, style.groove, time),
      beatUnit * (style.pianoPattern === 'ballad' ? 0.78 : 0.42),
      0.34 + lift + (noteIndex % Math.max(1, beatsPerBar) === 0 ? 0.065 : 0) + (random() - 0.5) * 0.035,
      { hand: 'right', scoreRole: 'accompaniment', articulation: 'legato' },
    );
  });
}

function addHarmonyBar(track, chord, time, barSeconds, beatUnit, beatsPerBar, style, sectionName, random) {
  const plucked = track.instrument.includes('guitar');
  const family = sectionFamily(sectionName);
  const level = (family === 'chorus' ? 0.54 : 0.42) + (style.energy === 'high' ? 0.04 : 0);
  const hitBeats = plucked
    ? (style.id === 'catchy-acoustic' ? [0, 1, 2, 2.5, 3].filter((beat) => beat < beatsPerBar) : [0, 2].filter((beat) => beat < beatsPerBar))
    : [0, Math.max(1, beatsPerBar / 2)];
  hitBeats.forEach((beat, hitIndex) => {
    const direction = hitIndex % 2 === 0 ? 1 : -1;
    chord.notes.forEach((unusedMidi, index) => {
      const voiceIndex = direction > 0 ? index : chord.notes.length - 1 - index;
      const playedMidi = chord.notes[voiceIndex] + (family === 'chorus' && voiceIndex === 2 ? 12 : 0);
      const stringIndex = voiceIndex + 1;
      const guitarOpenMidi = [40, 45, 50, 55, 59, 64][stringIndex];
      addTrackNote(
        track,
        playedMidi,
        humanizedTime(time + beat * beatUnit + index * (plucked ? 0.018 : 0.01), random, style.groove, time),
        plucked ? beatUnit * 0.72 : barSeconds * 0.43,
        level + (hitIndex === 0 ? 0.05 : 0) + index * 0.015,
        {
          hand: playedMidi < 60 ? 'left' : 'right', scoreRole: 'harmony',
          articulation: plucked ? (direction > 0 ? 'down-strum' : 'up-strum') : 'connected',
          ...(plucked ? { stringIndex, fret: clamp(playedMidi - guitarOpenMidi, 0, 24) } : {}),
        },
      );
    });
  });
}

const MELODY_DEGREES = Object.freeze({
  verse: [3, 2, 1, 2, 3, 5, 3, 2],
  'pre-chorus': [2, 3, 4, 5, 4, 5, 6, 5],
  chorus: [5, 5, 6, 5, 3, 2, 3, 1],
  bridge: [6, 5, 4, 3, 4, 2, 5, 3],
});

const RHYTHM_WEIGHTS = Object.freeze({
  'catchy-acoustic': [0.5, 0.5, 1, 0.5, 0.5, 1],
  'intimate-piano': [1, 0.5, 1.5, 0.5, 1, 1.5],
  'bright-pop': [0.5, 0.5, 0.5, 1, 0.5, 1],
  'dreamy-indie': [1, 0.5, 1, 1.5, 0.5, 1],
  'soulful-rnb': [0.75, 0.5, 1.25, 0.5, 0.75, 1.25],
  'cinematic-rise': [1, 1, 0.5, 1.5, 1, 2],
});

function nearestScalePitch(scaleMidis, pitchClass, target) {
  const matching = scaleMidis.filter((midi) => ((midi % 12) + 12) % 12 === pitchClass);
  const choices = matching.length ? matching : scaleMidis;
  return choices.reduce((best, midi) => (
    Math.abs(midi - target) < Math.abs(best - target) ? midi : best
  ), choices[0]);
}

function phraseMelodyPitch({ key, scaleMidis, chord, family, index, count, lineIndex, previous }) {
  const scale = key.minor ? MINOR_SCALE : MAJOR_SCALE;
  const motif = MELODY_DEGREES[family] || MELODY_DEGREES.verse;
  let degree = motif[(index + lineIndex * 2) % motif.length] - 1;
  if (index === count - 1) degree = family === 'chorus' ? 0 : romanDegree(chord.degree).index;
  const pitchClass = (key.root + scale[degree % scale.length]) % 12;
  const centre = scaleMidis[Math.floor(scaleMidis.length * (family === 'chorus' ? 0.6 : 0.48))];
  const contour = Math.sin((index / Math.max(1, count - 1)) * Math.PI) * (family === 'chorus' ? 4 : 2);
  const target = Number.isFinite(previous)
    ? previous + clamp(centre + contour - previous, -5, 5)
    : centre + contour;
  return nearestScalePitch(scaleMidis, pitchClass, target);
}

function polishPianoPerformance(events) {
  const sorted = events
    .map((event) => ({ ...event }))
    .sort((left, right) => left.time - right.time || left.midi - right.midi);
  const output = [];
  const lastByMidi = new Map();
  sorted.forEach((event) => {
    const previous = lastByMidi.get(event.midi);
    if (previous && event.time - previous.time < 0.065) {
      if (event.scoreRole === 'vocal-melody' && previous.scoreRole !== 'vocal-melody') {
        const previousIndex = output.indexOf(previous);
        if (previousIndex >= 0) output.splice(previousIndex, 1);
      } else {
        previous.duration = Math.max(previous.duration, event.duration);
        previous.audioDuration = previous.duration;
        previous.velocity = Math.max(previous.velocity, event.velocity);
        return;
      }
    } else if (previous && previous.time + previous.duration > event.time - 0.018) {
      previous.duration = Number(Math.max(0.06, event.time - previous.time - 0.018).toFixed(4));
      previous.audioDuration = previous.duration;
    }
    output.push(event);
    lastByMidi.set(event.midi, event);
  });
  return output.sort((left, right) => left.time - right.time || left.midi - right.midi);
}

function vocalScale(key, range = 'C3-G4') {
  const matches = String(range).match(/([A-G](?:#|b)?)(-?\d)\s*[-–]\s*([A-G](?:#|b)?)(-?\d)/i);
  const noteMidi = (name, octave) => 12 * (Number(octave) + 1) + (ROOTS[name[0].toUpperCase() + name.slice(1)] ?? 0);
  const low = matches ? noteMidi(matches[1], matches[2]) : 60;
  const high = matches ? noteMidi(matches[3], matches[4]) : 79;
  const scale = key.minor ? MINOR_SCALE : MAJOR_SCALE;
  const pitches = [];
  for (let midi = Math.max(36, low); midi <= Math.min(96, high); midi += 1) {
    const relative = ((midi - key.root) % 12 + 12) % 12;
    if (scale.includes(relative)) pitches.push(midi);
  }
  return pitches.length >= 4 ? pitches : [60, 62, 64, 67, 69, 72];
}

export function buildSongArrangement(projectInput) {
  const project = cloneJson(projectInput || createBlankMusicProject());
  const bpm = clamp(Math.round(Number(project.brief?.bpm) || 104), 40, 220);
  const beatSeconds = 60 / bpm;
  const beatsPerBar = project.brief?.timeSignature === '3/4' ? 3 : project.brief?.timeSignature === '6/8' ? 6 : 4;
  const beatUnit = project.brief?.timeSignature === '6/8' ? beatSeconds / 2 : beatSeconds;
  const barSeconds = beatsPerBar * beatUnit;
  const key = parseKey(project.brief?.key);
  const style = activeStyle(project);
  const voice = musicVoiceProfile(project.brief?.singerVoice);
  const progression = Array.isArray(project.chordDegrees) && project.chordDegrees.length
    ? project.chordDegrees.slice(0, 12)
    : ['I', 'V', 'vi', 'IV'];
  const structure = Array.isArray(project.brief?.structure) && project.brief.structure.length
    ? project.brief.structure
    : DEFAULT_STRUCTURE;
  const instruments = new Set(project.brief?.instruments || []);
  if (!instruments.size) instruments.add('piano');
  const harmonyLabels = {
    piano: 'Piano harmony',
    synth: 'Synth harmony',
    'acoustic-guitar': 'Acoustic guitar',
    'electric-guitar': 'Electric guitar',
  };
  const harmonyColours = {
    piano: '#8174ff',
    synth: '#a86cff',
    'acoustic-guitar': '#f5a862',
    'electric-guitar': '#ef6aa8',
  };
  const harmonyInstruments = Object.keys(harmonyLabels)
    .filter((instrument) => instrument !== 'piano' && instruments.has(instrument));
  const harmonyTracks = harmonyInstruments.map((instrument) => makeTrack(
    `harmony-${instrument}`,
    harmonyLabels[instrument],
    instrument,
    'harmony',
    harmonyColours[instrument],
  ));
  const pianoTrack = makeTrack('harmony-piano', 'Piano accompaniment', 'piano', 'harmony', harmonyColours.piano);
  pianoTrack.enabled = instruments.has('piano');
  const bassTrack = makeTrack('bass', 'Bass', 'upright-bass', 'bass', '#5ce1d7');
  const drumTrack = makeTrack('drums', 'Drums', 'drums', 'rhythm', '#ff6c9f');
  const melodyTrack = makeTrack('guide-melody', `${voice.label} guide`, voice.instrument, 'melody', '#ffd85c');
  const textureTracks = ['violin', 'flute']
    .filter((instrument) => instruments.has(instrument))
    .map((instrument) => makeTrack(
      `texture-${instrument}`,
      instrument === 'violin' ? 'Strings' : 'Flute',
      instrument,
      'texture',
      instrument === 'violin' ? '#b88cff' : '#6cbcff',
    ));
  const tracks = [pianoTrack, ...harmonyTracks, bassTrack, drumTrack, ...textureTracks, melodyTrack];
  bassTrack.enabled = instruments.has('upright-bass');
  drumTrack.enabled = instruments.has('drums');
  melodyTrack.enabled = true;
  const seed = hashString(JSON.stringify({
    idea: project.brief?.idea,
    title: project.title,
    lyrics: project.lyrics,
    bpm,
    key: project.brief?.key,
    style: style.id,
    voice: voice.id,
  }));
  const random = seededRandom(seed);
  const sections = [];
  const chords = [];
  const lyricCues = [];
  const guideSyllables = [];
  const scaleMidis = vocalScale(key, project.brief?.vocalRange || project.vocalCoach?.comfortableRange);
  let cursor = 0;
  let previousMelody = null;
  let previousPianoVoicing = null;

  structure.forEach((sectionName, sectionIndex) => {
    const lines = sectionLyrics(project, sectionName);
    const bars = barsForSection(sectionName, lines);
    const sectionStart = cursor;
    const sectionDuration = bars * barSeconds;
    sections.push({
      id: `${sectionName}-${sectionIndex}`,
      type: sectionName,
      label: sectionName.replace(/-/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
      start: Number(sectionStart.toFixed(4)),
      end: Number((sectionStart + sectionDuration).toFixed(4)),
      bars,
    });

    for (let bar = 0; bar < bars; bar += 1) {
      const degree = progression[(bar + sectionIndex) % progression.length];
      const chord = chordForDegree(key, degree, 3);
      chord.degree = degree;
      const time = sectionStart + bar * barSeconds;
      chords.push({ time: Number(time.toFixed(4)), duration: Number(barSeconds.toFixed(4)), degree, label: chord.label });
      const pianoVoicing = voiceLedChord(chord, previousPianoVoicing);
      addPianoBar(
        pianoTrack,
        chord,
        pianoVoicing,
        time,
        barSeconds,
        beatUnit,
        beatsPerBar,
        style,
        sectionName,
        random,
      );
      previousPianoVoicing = pianoVoicing;
      harmonyTracks.forEach((harmonyTrack, trackIndex) => {
        addHarmonyBar(
          harmonyTrack,
          chord,
          time + trackIndex * 0.004,
          barSeconds,
          beatUnit,
          beatsPerBar,
          style,
          sectionName,
          random,
        );
      });
      textureTracks.forEach((textureTrack, textureIndex) => {
        const textureMidi = chord.notes[(bar + textureIndex + 1) % chord.notes.length] + 12;
        addTrackNote(
          textureTrack,
          textureMidi,
          time + beatUnit * 0.08,
          barSeconds * 0.82,
          sectionName.includes('chorus') ? 0.48 : 0.34,
          { scoreRole: 'texture' },
        );
      });
      if (bassTrack.enabled) {
        const bassRoot = chord.root - 12;
        const bassPulses = project.brief?.energy === 'low' ? 1 : 2;
        for (let pulse = 0; pulse < bassPulses; pulse += 1) {
          const bassMidi = pulse === 0 ? bassRoot : chord.notes[2] - 12;
          addTrackNote(
            bassTrack,
            bassMidi,
            humanizedTime(time + pulse * (barSeconds / bassPulses), random, style.groove * 0.5, time),
            barSeconds / bassPulses * 0.68,
            0.54 + (sectionFamily(sectionName) === 'chorus' ? 0.08 : 0),
            { scoreRole: 'bass', articulation: 'rounded' },
          );
        }
      }
      if (drumTrack.enabled) {
        const hatSteps = beatsPerBar * 2;
        for (let step = 0; step < hatSteps; step += 1) {
          const drumTime = humanizedTime(time + step * (beatUnit / 2), random, style.groove * 0.35, time);
          addTrackNote(drumTrack, 42, drumTime, 0.08, step % 2 ? 0.3 : 0.42, { percussion: 'closed-hat' });
        }
        const kickBeats = project.brief?.energy === 'high' ? [0, 2, 2.75] : project.brief?.energy === 'low' ? [0] : [0, 2];
        kickBeats
          .filter((beat) => beat < beatsPerBar)
          .forEach((beat) => addTrackNote(drumTrack, 36, time + beat * beatUnit, 0.12, beat === 0 ? 0.7 : 0.58, { percussion: 'kick' }));
        const snareBeats = beatsPerBar >= 4 ? [1, 3] : [Math.floor(beatsPerBar / 2)];
        snareBeats.forEach((beat) => addTrackNote(
          drumTrack,
          38,
          humanizedTime(time + beat * beatUnit, random, style.groove * 0.25, time),
          0.12,
          0.62,
          { percussion: 'snare' },
        ));
      }
    }

    if (lines.length) {
      const lineDuration = sectionDuration / lines.length;
      lines.forEach((line, lineIndex) => {
        const lineStart = sectionStart + lineIndex * lineDuration;
        const units = lyricUnits(line);
        const family = sectionFamily(sectionName);
        const leadIn = Math.min(beatUnit * 0.25, lineDuration * 0.06);
        const restTail = Math.max(beatUnit * 0.65, lineDuration * 0.18);
        const phraseDuration = Math.max(beatUnit, lineDuration - leadIn - restTail);
        const rhythm = RHYTHM_WEIGHTS[style.id] || RHYTHM_WEIGHTS['catchy-acoustic'];
        const weights = units.map((unit, index) => (
          rhythm[index % rhythm.length] * (unit.continuation ? 0.72 : 1)
        ));
        const weightTotal = weights.reduce((total, weight) => total + weight, 0) || 1;
        const scaleDuration = phraseDuration / weightTotal;
        lyricCues.push({
          id: `${sectionIndex}-${lineIndex}`,
          section: sectionName,
          text: line,
          start: Number(lineStart.toFixed(4)),
          end: Number((lineStart + lineDuration).toFixed(4)),
        });
        let phraseCursor = lineStart + leadIn;
        units.forEach((unit, unitIndex) => {
          const localBar = clamp(Math.floor((phraseCursor - sectionStart) / barSeconds), 0, bars - 1);
          const degree = progression[(localBar + sectionIndex) % progression.length];
          const activeChord = chordForDegree(key, degree, 3);
          activeChord.degree = degree;
          const pitch = phraseMelodyPitch({
            key,
            scaleMidis,
            chord: activeChord,
            family,
            index: unitIndex,
            count: units.length,
            lineIndex,
            previous: previousMelody,
          });
          const start = humanizedTime(phraseCursor, random, style.groove * 0.32, lineStart);
          const noteSpace = weights[unitIndex] * scaleDuration;
          const available = Math.max(0.05, lineStart + lineDuration - start - 0.04);
          const duration = Math.min(available, noteSpace * (unitIndex === units.length - 1 ? 1.18 : 0.82));
          const actualDuration = Math.min(available, Math.max(beatUnit * 0.22, duration));
          const strongBeat = Math.abs(((start - sectionStart) / beatUnit) % 1) < 0.15;
          const melodicLift = family === 'chorus' ? 0.08 : 0;
          addTrackNote(melodyTrack, pitch, start, actualDuration, (strongBeat ? 0.76 : 0.66) + melodicLift, {
            hand: 'right',
            scoreRole: 'vocal-melody',
            lyric: unit.text,
            word: unit.word,
            articulation: unit.continuation ? 'legato' : 'sung',
          });
          guideSyllables.push({
            time: Number(start.toFixed(4)),
            duration: Number(actualDuration.toFixed(4)),
            note: midiToNote(pitch),
            midi: pitch,
            text: unit.text,
            word: unit.word,
            section: sectionName,
          });
          previousMelody = pitch;
          phraseCursor += noteSpace;
        });
      });
    }
    cursor += sectionDuration;
  });

  const pianoNotes = polishPianoPerformance([
    ...pianoTrack.events,
    ...melodyTrack.events.map((event) => ({ ...event, velocity: Math.min(0.96, event.velocity + 0.1) })),
  ]);
  const pedals = chords.flatMap((chord) => [
    { time: Number((chord.time + 0.035).toFixed(4)), down: true },
    { time: Number(Math.max(chord.time + 0.08, chord.time + chord.duration - 0.055).toFixed(4)), down: false },
  ]);

  return {
    title: project.title || titleFromIdea(project.brief?.idea),
    composer: 'Created with Polymath Musician',
    sourceType: 'polymath-create-music',
    readyToPlayFormat: 'polymath-arrangement-v2',
    bpm,
    key: project.brief?.key || 'C major',
    timeSignature: project.brief?.timeSignature || '4/4',
    genre: project.brief?.genre || 'Pop',
    mood: project.brief?.mood || 'Hopeful',
    stylePreset: style.id,
    styleLabel: style.label,
    singerVoice: voice.id,
    singerVoiceLabel: voice.label,
    duration: Number(cursor.toFixed(4)),
    notes: pianoNotes,
    pedals,
    tracks: tracks.filter((track) => track.enabled !== false),
    chords,
    sections,
    lyrics: project.lyrics,
    lyricCues,
    guideSyllables,
    vocalCoach: project.vocalCoach,
    provenance: {
      generatedAt: new Date().toISOString(),
      engine: 'polymath-musical-arranger-v002',
      userLed: true,
      referencePolicy: 'high-level musical traits only',
      pedalPolicy: 'repedal-each-harmony-change',
    },
  };
}

export function lyricsAsText(project) {
  return Object.entries(project?.lyrics || {})
    .map(([section, lines]) => `[${section.replace(/-/g, ' ')}]\n${(lines || []).join('\n')}`)
    .join('\n\n');
}

export function lyricsFromText(value) {
  const sections = {};
  let current = 'verse-1';
  String(value || '').split(/\r?\n/).forEach((rawLine) => {
    const line = rawLine.trim();
    const heading = line.match(/^\[([^\]]+)\]$/);
    if (heading) {
      current = heading[1].trim().toLowerCase().replace(/[^a-z0-9]+/g, '-');
      if (!sections[current]) sections[current] = [];
      return;
    }
    if (!line) return;
    if (!sections[current]) sections[current] = [];
    sections[current].push(line.slice(0, 120));
  });
  return Object.keys(sections).length ? sections : defaultLyrics();
}

export { DEFAULT_STRUCTURE };
