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
      vocalRange: 'C3-G4',
      reference: '',
      referenceNotes: '',
      instruments: ['piano', 'upright-bass', 'drums'],
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

function melodyPitch(scaleMidis, index, previous, random, energy) {
  const contour = Math.sin(index * 0.82) * 1.6 + (random() - 0.5) * (energy === 'high' ? 2.2 : 1.2);
  const target = clamp(Math.round((scaleMidis.length - 1) / 2 + contour), 0, scaleMidis.length - 1);
  const pitch = scaleMidis[target];
  if (!Number.isFinite(previous)) return pitch;
  if (Math.abs(pitch - previous) <= 7) return pitch;
  return scaleMidis.reduce((best, candidate) => (
    Math.abs(candidate - previous) < Math.abs(best - previous) ? candidate : best
  ), pitch);
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
  const harmonyInstruments = Object.keys(harmonyLabels).filter((instrument) => instruments.has(instrument));
  if (!harmonyInstruments.length) harmonyInstruments.push('piano');
  const harmonyTracks = harmonyInstruments.map((instrument) => makeTrack(
    `harmony-${instrument}`,
    harmonyLabels[instrument],
    instrument,
    'harmony',
    harmonyColours[instrument],
  ));
  const bassTrack = makeTrack('bass', 'Bass', 'upright-bass', 'bass', '#5ce1d7');
  const drumTrack = makeTrack('drums', 'Drums', 'drums', 'rhythm', '#ff6c9f');
  const melodyTrack = makeTrack('guide-melody', 'Vocal guide melody', 'synth', 'melody', '#ffd85c');
  const textureTracks = ['violin', 'flute']
    .filter((instrument) => instruments.has(instrument))
    .map((instrument) => makeTrack(
      `texture-${instrument}`,
      instrument === 'violin' ? 'Strings' : 'Flute',
      instrument,
      'texture',
      instrument === 'violin' ? '#b88cff' : '#6cbcff',
    ));
  const tracks = [...harmonyTracks, bassTrack, drumTrack, ...textureTracks, melodyTrack];
  bassTrack.enabled = instruments.has('upright-bass');
  drumTrack.enabled = instruments.has('drums');
  melodyTrack.enabled = true;
  const seed = hashString(JSON.stringify({
    idea: project.brief?.idea,
    title: project.title,
    lyrics: project.lyrics,
    bpm,
    key: project.brief?.key,
  }));
  const random = seededRandom(seed);
  const sections = [];
  const chords = [];
  const lyricCues = [];
  const guideSyllables = [];
  const scaleMidis = vocalScale(key, project.brief?.vocalRange || project.vocalCoach?.comfortableRange);
  let cursor = 0;
  let melodyIndex = 0;
  let previousMelody = null;

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
      const time = sectionStart + bar * barSeconds;
      chords.push({ time: Number(time.toFixed(4)), duration: Number(barSeconds.toFixed(4)), degree, label: chord.label });
      harmonyTracks.forEach((harmonyTrack, trackIndex) => {
        const plucked = harmonyTrack.instrument.includes('guitar');
        chord.notes.forEach((midi, voiceIndex) => {
          const playedMidi = midi + (voiceIndex === 2 && sectionName.includes('chorus') ? 12 : 0);
          const guitarStringIndex = voiceIndex + 1;
          const guitarOpenMidi = [40, 45, 50, 55, 59, 64][guitarStringIndex];
          addTrackNote(
            harmonyTrack,
            playedMidi,
            time + (voiceIndex * (plucked ? 0.026 : 0.012)) + trackIndex * 0.006,
            barSeconds * (plucked ? 0.72 : 0.9),
            (sectionName.includes('chorus') ? 0.58 : 0.46) + voiceIndex * 0.02,
            {
              hand: midi < 60 ? 'left' : 'right',
              scoreRole: 'harmony',
              ...(plucked ? {
                stringIndex: guitarStringIndex,
                fret: clamp(playedMidi - guitarOpenMidi, 0, 24),
              } : {}),
            },
          );
        });
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
          addTrackNote(bassTrack, bassRoot, time + pulse * (barSeconds / bassPulses), barSeconds / bassPulses * 0.72, 0.62, { scoreRole: 'bass' });
        }
      }
      if (drumTrack.enabled) {
        const subdivisions = project.brief?.energy === 'high' ? beatsPerBar * 2 : beatsPerBar;
        for (let step = 0; step < subdivisions; step += 1) {
          const drumTime = time + step * (barSeconds / subdivisions);
          addTrackNote(drumTrack, 42, drumTime, 0.08, step % 2 ? 0.3 : 0.42, { percussion: 'closed-hat' });
        }
        addTrackNote(drumTrack, 36, time, 0.12, 0.72, { percussion: 'kick' });
        if (beatsPerBar >= 4) addTrackNote(drumTrack, 38, time + barSeconds / 2, 0.12, 0.68, { percussion: 'snare' });
      }
    }

    if (lines.length) {
      const lineDuration = sectionDuration / lines.length;
      lines.forEach((line, lineIndex) => {
        const lineStart = sectionStart + lineIndex * lineDuration;
        const units = lyricUnits(line);
        const unitDuration = lineDuration / Math.max(1, units.length);
        lyricCues.push({
          id: `${sectionIndex}-${lineIndex}`,
          section: sectionName,
          text: line,
          start: Number(lineStart.toFixed(4)),
          end: Number((lineStart + lineDuration).toFixed(4)),
        });
        units.forEach((unit, unitIndex) => {
          const pitch = melodyPitch(scaleMidis, melodyIndex, previousMelody, random, project.brief?.energy);
          const start = lineStart + unitIndex * unitDuration;
          const duration = unitDuration * (unitIndex === units.length - 1 ? 1.45 : 0.86);
          const strongBeat = Math.abs(((start - sectionStart) / beatUnit) % 1) < 0.15;
          addTrackNote(melodyTrack, pitch, start, duration, strongBeat ? 0.82 : 0.72, {
            hand: 'right', scoreRole: 'vocal-melody', lyric: unit.text, word: unit.word,
          });
          guideSyllables.push({
            time: Number(start.toFixed(4)),
            duration: Number(duration.toFixed(4)),
            note: midiToNote(pitch),
            midi: pitch,
            text: unit.text,
            word: unit.word,
            section: sectionName,
          });
          previousMelody = pitch;
          melodyIndex += 1;
        });
      });
    }
    cursor += sectionDuration;
  });

  const pianoNotes = [
    ...harmonyTracks[0].events,
    ...melodyTrack.events.map((event) => ({ ...event, velocity: Math.min(1, event.velocity + 0.06) })),
    ...(bassTrack.enabled ? bassTrack.events : []),
  ].sort((left, right) => left.time - right.time || left.midi - right.midi);
  const pedals = sections.flatMap((section) => [
    { time: section.start, down: true },
    { time: Math.max(section.start, section.end - 0.08), down: false },
  ]);

  return {
    title: project.title || titleFromIdea(project.brief?.idea),
    composer: 'Created with Polymath Musician',
    sourceType: 'polymath-create-music',
    readyToPlayFormat: 'polymath-arrangement-v1',
    bpm,
    key: project.brief?.key || 'C major',
    timeSignature: project.brief?.timeSignature || '4/4',
    genre: project.brief?.genre || 'Pop',
    mood: project.brief?.mood || 'Hopeful',
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
      engine: 'polymath-deterministic-arranger-v001',
      userLed: true,
      referencePolicy: 'high-level musical traits only',
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
