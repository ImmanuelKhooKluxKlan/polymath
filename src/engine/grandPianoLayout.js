import { buildPianoRange, midiToNote, NOTE_NAMES, parseNote } from './noteMath.js';

export const GRAND_START_NOTE = 'A0';
export const GRAND_END_NOTE = 'C8';
export const PIANELLA_DEFAULT_START_NOTE = 'A1';
export const PIANELLA_DEFAULT_END_NOTE = 'C7';
export const TWO_STOREY_SPLIT_NOTE = 'C4';

// Real piano dimensions: black keys are roughly 0.58 of a white key.
export const BLACK_KEY_WIDTH_RATIO = 0.583;

export const GRAND_START_MIDI = parseNote(GRAND_START_NOTE).midi;
export const GRAND_END_MIDI = parseNote(GRAND_END_NOTE).midi;
export const PIANELLA_DEFAULT_START_MIDI = parseNote(PIANELLA_DEFAULT_START_NOTE).midi;
export const PIANELLA_DEFAULT_END_MIDI = parseNote(PIANELLA_DEFAULT_END_NOTE).midi;
export const TWO_STOREY_SPLIT_MIDI = parseNote(TWO_STOREY_SPLIT_NOTE).midi;
export const PIANELLA_SINGLE_STOREY_MAX_SPAN = PIANELLA_DEFAULT_END_MIDI - PIANELLA_DEFAULT_START_MIDI + 1;

export const grandPianoKeys = buildPianoRange(GRAND_START_NOTE, GRAND_END_NOTE);
export const grandWhiteCount = grandPianoKeys.filter((key) => !key.isBlack).length;

function positiveMod(value, mod) {
  return ((value % mod) + mod) % mod;
}

function isWhiteMidi(midi) {
  return !NOTE_NAMES[positiveMod(midi, 12)].includes('#');
}

function countWhiteKeysBefore(midi, startMidi) {
  let count = 0;
  for (let candidate = startMidi; candidate < midi; candidate += 1) {
    if (isWhiteMidi(candidate)) count += 1;
  }
  return count;
}

function rowForMidi(rows, midi) {
  return rows.find((row) => midi >= row.startMidi && midi <= row.endMidi) || null;
}

function buildRow(id, label, startMidi, endMidi) {
  const keys = buildPianoRange(midiToNote(startMidi), midiToNote(endMidi));
  const whiteKeys = keys.filter((key) => !key.isBlack);
  const whiteCount = whiteKeys.length;

  function getPosition(noteOrMidi) {
    const midi = typeof noteOrMidi === 'number' ? noteOrMidi : parseNote(noteOrMidi).midi;
    if (midi < startMidi || midi > endMidi) return null;

    const note = midiToNote(midi);
    const pitch = NOTE_NAMES[positiveMod(midi, 12)];
    const isBlack = pitch.includes('#');
    const widthInWhiteKeys = isBlack ? BLACK_KEY_WIDTH_RATIO : 1;
    const whiteIndexFromStart = countWhiteKeysBefore(midi, startMidi);

    if (!isBlack) {
      return {
        rowId: id,
        note,
        midi,
        pitch,
        isBlack,
        whiteIndex: whiteIndexFromStart,
        leftEdgeWhiteUnits: whiteIndexFromStart,
        centerWhiteUnits: whiteIndexFromStart + 0.5,
        widthInWhiteKeys,
        leftPercent: (whiteIndexFromStart / whiteCount) * 100,
        centerPercent: ((whiteIndexFromStart + 0.5) / whiteCount) * 100,
        widthPercent: (widthInWhiteKeys / whiteCount) * 100,
      };
    }

    // For a black key, the visual center sits on the boundary after the previous white key.
    const boundaryAfterPreviousWhite = whiteIndexFromStart;
    const leftEdgeWhiteUnits = boundaryAfterPreviousWhite - (BLACK_KEY_WIDTH_RATIO / 2);

    return {
      rowId: id,
      note,
      midi,
      pitch,
      isBlack,
      whiteIndex: boundaryAfterPreviousWhite - 1,
      leftEdgeWhiteUnits,
      centerWhiteUnits: boundaryAfterPreviousWhite,
      widthInWhiteKeys,
      leftPercent: (leftEdgeWhiteUnits / whiteCount) * 100,
      centerPercent: (boundaryAfterPreviousWhite / whiteCount) * 100,
      widthPercent: (widthInWhiteKeys / whiteCount) * 100,
    };
  }

  const keysWithPositions = keys
    .map((key) => ({ ...key, position: getPosition(key.midi) }))
    .filter((key) => key.position);

  return {
    id,
    label,
    startMidi,
    endMidi,
    startNote: midiToNote(startMidi),
    endNote: midiToNote(endMidi),
    whiteCount,
    chromaticCount: endMidi - startMidi + 1,
    keys: keysWithPositions,
    whiteKeys: keysWithPositions.filter((key) => !key.isBlack),
    blackKeys: keysWithPositions.filter((key) => key.isBlack),
    getPosition,
  };
}

function normalizeMidiRange(songOrNotes) {
  const notes = Array.isArray(songOrNotes) ? songOrNotes : (songOrNotes?.notes || []);
  const midis = [];

  for (const note of notes) {
    try {
      const midi = typeof note.midi === 'number' ? note.midi : parseNote(note.note).midi;
      if (midi >= GRAND_START_MIDI && midi <= GRAND_END_MIDI) midis.push(midi);
    } catch {
      // Ignore unreadable notes.
    }
  }

  if (!midis.length) {
    return {
      minMidi: PIANELLA_DEFAULT_START_MIDI,
      maxMidi: PIANELLA_DEFAULT_END_MIDI,
      hasNotes: false,
    };
  }

  return {
    minMidi: Math.min(...midis),
    maxMidi: Math.max(...midis),
    hasNotes: true,
  };
}

export function getSongMidiRange(songOrNotes) {
  const range = normalizeMidiRange(songOrNotes);
  return {
    ...range,
    minNote: midiToNote(range.minMidi),
    maxNote: midiToNote(range.maxMidi),
    span: range.maxMidi - range.minMidi + 1,
  };
}

export function shouldUseTwoStoreys(songOrNotes) {
  const { minMidi, maxMidi, span } = getSongMidiRange(songOrNotes);

  return (
    minMidi < PIANELLA_DEFAULT_START_MIDI
    || maxMidi > PIANELLA_DEFAULT_END_MIDI
    || span > PIANELLA_SINGLE_STOREY_MAX_SPAN
  );
}

export function buildAdaptivePianoLayout(songOrNotes = null) {
  const songRange = getSongMidiRange(songOrNotes);
  const twoStoreys = shouldUseTwoStoreys(songOrNotes);

  if (!twoStoreys) {
    const row = buildRow(
      'main',
      'Polymath Musician row',
      PIANELLA_DEFAULT_START_MIDI,
      PIANELLA_DEFAULT_END_MIDI,
    );

    return {
      mode: 'pianella-single',
      isTwoStorey: false,
      songRange,
      rows: [row],
      rangeLabel: `${row.startNote}-${row.endNote}`,
      getPosition(noteOrMidi) {
        return row.getPosition(noteOrMidi);
      },
    };
  }

  const lower = buildRow('lower', 'Lower storey', GRAND_START_MIDI, TWO_STOREY_SPLIT_MIDI - 1);
  const upper = buildRow('upper', 'Upper storey', TWO_STOREY_SPLIT_MIDI, GRAND_END_MIDI);
  const rows = [upper, lower];

  return {
    mode: 'two-storey-grand',
    isTwoStorey: true,
    songRange,
    rows,
    rangeLabel: `${GRAND_START_NOTE}-${GRAND_END_NOTE}`,
    getPosition(noteOrMidi) {
      const midi = typeof noteOrMidi === 'number' ? noteOrMidi : parseNote(noteOrMidi).midi;
      const row = rowForMidi(rows, midi);
      return row?.getPosition(midi) || null;
    },
  };
}

export function buildLearningHandLayout(songOrNotes = null, hand = 'both') {
  if (hand === 'both') return buildAdaptivePianoLayout(songOrNotes);

  const isLeft = hand === 'left';
  const row = isLeft
    ? buildRow('lower', 'Left hand · lower storey', GRAND_START_MIDI, TWO_STOREY_SPLIT_MIDI - 1)
    : buildRow('upper', 'Right hand · upper storey', TWO_STOREY_SPLIT_MIDI, GRAND_END_MIDI);

  return {
    mode: isLeft ? 'learn-left-single-storey' : 'learn-right-single-storey',
    isTwoStorey: false,
    learningHand: hand,
    songRange: getSongMidiRange(songOrNotes),
    rows: [row],
    rangeLabel: `${row.startNote}-${row.endNote}`,
    getPosition(noteOrMidi) {
      return row.getPosition(noteOrMidi);
    },
  };
}

export function describeLayout(layout = buildAdaptivePianoLayout()) {
  if (layout.isTwoStorey) {
    return `Two-storey grand piano ${GRAND_START_NOTE}-${GRAND_END_NOTE} because song range is ${layout.songRange.minNote}-${layout.songRange.maxNote}`;
  }

  return `Polymath Musician single row ${PIANELLA_DEFAULT_START_NOTE}-${PIANELLA_DEFAULT_END_NOTE}`;
}

// Backwards-compatible exports for older imports. New components should use buildAdaptivePianoLayout().
const defaultLayout = buildAdaptivePianoLayout();
export const GRAND_VISIBLE_START_NOTE = PIANELLA_DEFAULT_START_NOTE;
export const GRAND_VISIBLE_END_NOTE = PIANELLA_DEFAULT_END_NOTE;
export const VISIBLE_START_MIDI = PIANELLA_DEFAULT_START_MIDI;
export const VISIBLE_END_MIDI = PIANELLA_DEFAULT_END_MIDI;
export const visiblePianoKeys = defaultLayout.rows[0].keys;
export const visibleWhiteKeys = defaultLayout.rows[0].whiteKeys;
export const visibleBlackKeys = defaultLayout.rows[0].blackKeys;
export const visibleWhiteCount = defaultLayout.rows[0].whiteCount;
export const visibleKeysWithPositions = defaultLayout.rows[0].keys;
export const visibleWhiteKeysWithPositions = defaultLayout.rows[0].whiteKeys;
export const visibleBlackKeysWithPositions = defaultLayout.rows[0].blackKeys;
export function getGrandKeyPosition(noteOrMidi) {
  return defaultLayout.getPosition(noteOrMidi);
}
export function isInsideVisibleRange(noteOrMidi) {
  return Boolean(defaultLayout.getPosition(noteOrMidi));
}
export function describeVisibleRange() {
  return describeLayout(defaultLayout);
}
