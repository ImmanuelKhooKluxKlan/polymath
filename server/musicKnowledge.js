'use strict';

const MUSIC_KNOWLEDGE_VERSION = 'polymath-music-fundamentals-v1';

const ENTRIES = Object.freeze([
  {
    id: 'pitch-midi',
    keywords: ['pitch', 'midi note', 'middle c', 'a4', 'concert pitch', 'frequency', '440 hz'],
    fact: 'In the common scientific-pitch/MIDI convention, A4 is MIDI 69 and is often tuned to 440 Hz; C4 (middle C) is MIDI 60. Octave naming can differ between manufacturers, so state the convention.',
  },
  {
    id: 'intervals',
    keywords: ['interval', 'semitone', 'half step', 'whole step', 'major third', 'minor third', 'perfect fifth', 'octave'],
    fact: 'In 12-tone equal temperament: minor 2nd=1 semitone, major 2nd=2, minor 3rd=3, major 3rd=4, perfect 4th=5, tritone=6, perfect 5th=7, minor 6th=8, major 6th=9, minor 7th=10, major 7th=11, octave=12.',
  },
  {
    id: 'scales-modes',
    keywords: ['scale', 'major scale', 'minor scale', 'mode', 'ionian', 'dorian', 'mixolydian'],
    fact: 'Step patterns in semitones: major/Ionian 2-2-1-2-2-2-1; natural minor/Aeolian 2-1-2-2-1-2-2; Dorian 2-1-2-2-2-1-2; Mixolydian 2-2-1-2-2-1-2. Fingering depends on instrument and key.',
  },
  {
    id: 'chord-construction',
    keywords: ['chord', 'triad', 'seventh chord', 'major chord', 'minor chord', 'diminished', 'augmented', 'voicing'],
    fact: 'Root-position triad semitone stacks are major 4+3, minor 3+4, diminished 3+3, and augmented 4+4. Voicing changes spacing or order without changing chord identity; inversion places a chord member other than the root in the bass.',
  },
  {
    id: 'meter-rhythm-tempo',
    keywords: ['rhythm', 'meter', 'time signature', 'tempo', 'bpm', 'beat', 'syncopation', 'polyrhythm'],
    fact: 'Tempo is beat speed; meter is the recurring grouping of beats. In a time signature, the top number gives the notated beat count per bar and the bottom number identifies the note value used as the written beat unit; compound meter commonly groups subdivisions in threes.',
  },
  {
    id: 'piano-layout-action',
    keywords: ['piano', 'keyboard', '88 key', 'hammer', 'piano range', 'velocity', 'touch'],
    fact: 'A standard modern 88-key piano spans A0 to C8. Pressing a key launches a felt-covered hammer; key velocity strongly affects level and timbre, while the mechanism lets the hammer escape before striking so it does not remain against the string.',
  },
  {
    id: 'piano-pedals',
    keywords: ['pedal', 'damper pedal', 'sustain pedal', 'una corda', 'soft pedal', 'sostenuto', 'half pedal', 'muddy'],
    fact: 'The right damper pedal lifts dampers and permits sympathetic resonance; the left una-corda/soft pedal changes action and timbre; the middle sostenuto pedal on equipped grands sustains only notes held when it is engaged. Pedal timing follows harmony and acoustics, not one fixed rule.',
  },
  {
    id: 'guitar-standard-tuning',
    keywords: ['guitar', 'guitar tuning', 'fret', 'string', 'capo', 'e a d g b e'],
    fact: 'Six-string standard guitar tuning from lowest sounding string is E2-A2-D3-G3-B3-E4. Adjacent frets are one equal-tempered semitone. Guitar notation is commonly written one octave above sounding pitch; alternate tunings must be confirmed.',
  },
  {
    id: 'ukulele-tuning',
    keywords: ['ukulele', 'uke', 'reentrant', 'high g', 'low g'],
    fact: 'Common soprano/concert/tenor high-G ukulele tuning is re-entrant G4-C4-E4-A4; low-G tuning replaces G4 with G3. Baritone ukulele commonly uses D3-G3-B3-E4, but the actual tuning should be confirmed.',
  },
  {
    id: 'bowed-string-tunings',
    keywords: ['violin', 'viola', 'cello', 'double bass', 'upright bass', 'bowed string', 'bowing'],
    fact: 'Open-string tunings low to high: violin G3-D4-A4-E5; viola C3-G3-D4-A4; cello C2-G2-D3-A3; standard four-string double bass E1-A1-D2-G2. Double-bass notation normally sounds one octave below written pitch.',
  },
  {
    id: 'transposing-instruments',
    keywords: ['transposing instrument', 'clarinet', 'trumpet', 'saxophone', 'concert pitch', 'written pitch', 'horn'],
    fact: 'For a B-flat instrument, written C sounds concert B-flat; for an E-flat instrument, written C sounds concert E-flat (with register depending on instrument). Always identify the exact instrument and transposition before converting written and concert pitch.',
  },
  {
    id: 'woodwind-production',
    keywords: ['flute', 'clarinet', 'saxophone', 'oboe', 'bassoon', 'woodwind', 'reed', 'embouchure'],
    fact: 'Flute tone begins from an air jet at an edge; clarinet uses a single reed; oboe and bassoon use double reeds; saxophone uses a single reed. Breath support, embouchure, voicing, reed setup, and instrument condition all affect response.',
  },
  {
    id: 'brass-production',
    keywords: ['trumpet', 'trombone', 'french horn', 'tuba', 'brass', 'buzz', 'lip slur'],
    fact: 'Brass players excite an air column with vibrating lips. Valves or slide change effective tube length, while embouchure and air select notes from the harmonic series. More pressure alone is not a safe substitute for coordinated air and aperture.',
  },
  {
    id: 'percussion',
    keywords: ['drum', 'drums', 'percussion', 'snare', 'cymbal', 'rudiment', 'stick'],
    fact: 'Percussion technique balances rebound, stick height, timing, and tone. Drum notation and MIDI mappings vary by kit and publisher, so confirm the legend or map rather than assuming every note number.',
  },
  {
    id: 'voice',
    keywords: ['sing', 'singer', 'singing', 'vocal', 'voice type', 'soprano', 'alto', 'tenor', 'baritone', 'breath support'],
    fact: 'Voice categories describe tessitura, timbre, passaggi, and comfortable function—not only highest and lowest notes. Pain, persistent hoarseness, or forced phonation is not a normal training goal and warrants qualified clinical or pedagogical assessment.',
  },
  {
    id: 'acoustics',
    keywords: ['acoustic', 'acoustics', 'harmonic', 'overtone', 'timbre', 'decibel', 'loudness', 'resonance'],
    fact: 'Frequency strongly relates to perceived pitch, waveform spectrum and envelopes shape timbre, and sound-pressure level is measured logarithmically in decibels. Perceived loudness also depends on frequency, duration, context, room, and listener.',
  },
  {
    id: 'digital-audio',
    keywords: ['sample rate', 'bit depth', 'audio interface', 'latency', 'buffer', 'clipping', 'recording', 'mixing'],
    fact: 'Sample rate sets the sampling frequency; bit depth affects quantization resolution and digital dynamic range. Buffer size trades latency against processing stability. Digital clipping occurs when signal level exceeds the representable ceiling and is not repaired by lowering playback volume later.',
  },
  {
    id: 'functional-harmony',
    keywords: ['functional harmony', 'tonic', 'dominant', 'predominant', 'cadence', 'roman numeral', 'voice leading'],
    fact: 'In common-practice functional harmony, tonic supplies stability, predominant commonly moves toward dominant, and dominant creates directed tension toward tonic. Real music may prolong, substitute, tonicize, modalize, or avoid this pattern, so analyze sounding context and voice leading.',
  },
  {
    id: 'practice-design',
    keywords: ['practice', 'practise', 'learn faster', 'mistake', 'slow practice', 'metronome', 'memorize'],
    fact: 'Efficient practice isolates a small measurable problem, reduces difficulty or tempo, repeats with attention, varies the retrieval context, and retests after a short gap. Repeating an entire passage without identifying the error is less diagnostic.',
  },
  {
    id: 'ensemble-tuning',
    keywords: ['ensemble', 'band', 'orchestra', 'intonation', 'tune together', 'blend', 'balance'],
    fact: 'Ensemble intonation is contextual: players align pitch, overtone interaction, articulation, tone, balance, and harmonic function. A tuner is a reference, not a substitute for listening to beats and the ensemble chord.',
  },
]);

function normalizedWords(value) {
  return new Set(String(value || '').toLowerCase().match(/[a-z0-9#]+/g) || []);
}

function entryScore(entry, query, words) {
  return entry.keywords.reduce((score, keyword) => {
    const normalized = keyword.toLowerCase();
    if (query.includes(normalized)) return score + (normalized.includes(' ') ? 8 : 4);
    const keywordWords = normalizedWords(normalized);
    const overlap = [...keywordWords].filter((word) => words.has(word)).length;
    return score + overlap;
  }, 0);
}

function retrieveMusicKnowledge(query, maximum = 5) {
  const normalized = String(query || '').trim().toLowerCase();
  if (!normalized) return { version: MUSIC_KNOWLEDGE_VERSION, entries: [] };
  const words = normalizedWords(normalized);
  const entries = ENTRIES
    .map((entry) => ({ entry, score: entryScore(entry, normalized, words) }))
    .filter((candidate) => candidate.score >= 2)
    .sort((left, right) => right.score - left.score || left.entry.id.localeCompare(right.entry.id))
    .slice(0, Math.max(0, Math.min(8, Number(maximum) || 5)))
    .map(({ entry }) => ({ id: entry.id, fact: entry.fact }));
  return { version: MUSIC_KNOWLEDGE_VERSION, entries };
}

module.exports = {
  MUSIC_KNOWLEDGE_VERSION,
  retrieveMusicKnowledge,
};
