export const INSTRUMENTS = [
  {
    id: 'piano',
    label: 'Piano',
    shortLabel: 'Piano',
    icon: 'piano',
    route: 'studio',
    description: 'Grand-piano playback with pedal-aware falling notes and illuminated keys.',
  },
  {
    id: 'guitar',
    label: 'Acoustic Guitar',
    shortLabel: 'Guitar',
    icon: 'guitar',
    route: 'guitar',
    description: 'Acoustic guitar chords, tablature, strings, and fretboard playback.',
  },
  {
    id: 'fiddle',
    label: 'Five-string Fiddle',
    shortLabel: 'Fiddle',
    icon: 'fiddle',
    route: 'ensemble',
    description: 'See the exact string and finger position glow while each bowed note plays.',
    manualNotes: ['C3', 'G3', 'D4', 'A4', 'E5', 'F#4', 'B4', 'C5'],
  },
  {
    id: 'banjo',
    label: 'Five-string Banjo',
    shortLabel: 'Banjo',
    icon: 'banjo',
    route: 'ensemble',
    description: 'Follow illuminated strings and fret positions for bright bluegrass picking.',
    manualNotes: ['D3', 'G3', 'B3', 'D4', 'G4', 'A4', 'B4', 'D5'],
  },
  {
    id: 'mandolin',
    label: 'Mandolin',
    shortLabel: 'Mandolin',
    icon: 'mandolin',
    route: 'ensemble',
    description: 'Paired-string courses show exactly where to fret, pluck, and tremolo.',
    manualNotes: ['G3', 'D4', 'A4', 'E5', 'B4', 'C5', 'D5', 'G5'],
  },
  {
    id: 'dobro',
    label: 'Dobro / Resonator Guitar',
    shortLabel: 'Dobro',
    icon: 'dobro',
    route: 'ensemble',
    description: 'A horizontal resonator view shows the string and steel-bar position to copy.',
    manualNotes: ['G2', 'B2', 'D3', 'G3', 'B3', 'D4', 'E4', 'G4'],
  },
  {
    id: 'upright-bass',
    label: 'Upright Double Bass',
    shortLabel: 'Upright Bass',
    icon: 'upright-bass',
    route: 'ensemble',
    description: 'Fingerboard targets guide left-hand placement while the plucking hand keeps time.',
    manualNotes: ['E1', 'A1', 'D2', 'G2', 'A2', 'C3', 'D3', 'G3'],
  },
  {
    id: 'ukulele',
    label: 'Ukulele',
    shortLabel: 'Ukulele',
    icon: 'ukulele',
    route: 'ensemble',
    description: 'Four-string visual guidance for chords, melody, strumming, and pop accompaniment.',
    manualNotes: ['C4', 'E4', 'G4', 'A4', 'C5', 'D5', 'E5', 'G5'],
  },
  {
    id: 'electric-guitar',
    label: 'Electric Guitar',
    shortLabel: 'Electric',
    icon: 'electric-guitar',
    route: 'ensemble',
    description: 'Illuminated fretboard positions for riffs, lead lines, power chords, and pop parts.',
    manualNotes: ['E2', 'A2', 'D3', 'G3', 'B3', 'E4', 'G4', 'B4'],
  },
  {
    id: 'drums',
    label: 'Drum Set',
    shortLabel: 'Drums',
    icon: 'drums',
    route: 'ensemble',
    description: 'The exact kick, snare, tom, hi-hat, ride, or crash lights up on every hit.',
    manualNotes: ['C2', 'D2', 'F#2', 'G2', 'A2', 'C3', 'C#3', 'D#3'],
  },
  {
    id: 'synth',
    label: 'Synth Keyboard',
    shortLabel: 'Synth',
    icon: 'synth',
    route: 'ensemble',
    description: 'A compact illuminated keyboard for electronic-pop melodies, chords, and bass lines.',
    manualNotes: ['C3', 'D3', 'E3', 'G3', 'A3', 'C4', 'E4', 'G4'],
  },
  {
    id: 'violin', label: 'Violin', shortLabel: 'Violin', icon: 'violin', route: 'ensemble',
    description: 'Four-string violin fingerboard guidance with illuminated bowing positions.',
    manualNotes: ['G3', 'A3', 'B3', 'D4', 'E4', 'F#4', 'A4', 'E5'],
  },
  {
    id: 'cello', label: 'Cello', shortLabel: 'Cello', icon: 'cello', route: 'ensemble',
    description: 'Cello string and finger-position targets for bowed orchestral parts.',
    manualNotes: ['C2', 'G2', 'D3', 'A3', 'C3', 'E3', 'G3', 'D4'],
  },
  {
    id: 'flute', label: 'Flute', shortLabel: 'Flute', icon: 'flute', route: 'ensemble',
    description: 'Follow illuminated flute keys and fingering combinations for each pitch.',
    manualNotes: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
  },
  {
    id: 'saxophone', label: 'Saxophone', shortLabel: 'Sax', icon: 'saxophone', route: 'ensemble',
    description: 'Saxophone keywork highlights the fingering needed for melody and band parts.',
    manualNotes: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'Bb4', 'C5'],
  },
  {
    id: 'trumpet', label: 'Trumpet', shortLabel: 'Trumpet', icon: 'trumpet', route: 'ensemble',
    description: 'The correct three-valve combination lights up for every trumpet note.',
    manualNotes: ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'Bb4', 'C5'],
  },
  {
    id: 'clarinet', label: 'Clarinet', shortLabel: 'Clarinet', icon: 'clarinet', route: 'ensemble',
    description: 'Clarinet tone-hole and key targets show the required fingering.',
    manualNotes: ['D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4'],
  },
];

export const INSTRUMENT_BY_ID = Object.fromEntries(
  INSTRUMENTS.map((instrument) => [instrument.id, instrument]),
);

export const ENSEMBLE_INSTRUMENTS = INSTRUMENTS.filter(
  (instrument) => instrument.route === 'ensemble',
);

export function instrumentLabel(instrumentId) {
  return INSTRUMENT_BY_ID[instrumentId]?.label || 'Instrument';
}
