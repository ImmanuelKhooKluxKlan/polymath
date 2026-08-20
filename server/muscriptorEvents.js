const MIN_NOTE_SECONDS = 0.04;
const DEFAULT_NOTE_SECONDS = 0.4;

function round(value, places = 4) {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function midiToNote(midi) {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const value = Math.round(Number(midi));
  return `${names[((value % 12) + 12) % 12]}${Math.floor(value / 12) - 1}`;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function normalizeBeatGrid(value) {
  if (!value || typeof value !== 'object') return null;
  const bpm = finiteNumber(value.bpm);
  const beatsPerBar = finiteNumber(value.beats_per_bar ?? value.beatsPerBar);
  const firstDownbeat = finiteNumber(value.first_downbeat ?? value.firstDownbeat);
  const onsetDelay = finiteNumber(value.onset_delay ?? value.onsetDelay);
  if (!(bpm >= 20 && bpm <= 400) || !(beatsPerBar >= 1 && beatsPerBar <= 16)) return null;
  return {
    bpm: round(bpm, 3),
    beatsPerBar: Math.round(beatsPerBar),
    firstDownbeatSeconds: firstDownbeat === null ? null : round(firstDownbeat),
    onsetDelaySeconds: onsetDelay === null ? 0 : round(onsetDelay),
  };
}

function createMuscriptorEventCollector({ model = 'large', source = 'runpod', onProgress } = {}) {
  const starts = new Map();
  const completedNotes = [];
  let progress = { completed: 0, total: 0 };
  let beatGrid = null;
  let malformedEvents = 0;
  let unmatchedEndEvents = 0;
  let replacedStartEvents = 0;
  let finalResult = null;

  function buildNote(start, endTime, velocity = 0.78) {
    const midi = finiteNumber(start?.pitch);
    const startTime = finiteNumber(start?.start_time);
    if (midi === null || midi < 0 || midi > 127 || startTime === null || startTime < 0) return null;
    const roundedMidi = Math.round(midi);
    const parsedEnd = finiteNumber(endTime);
    const safeEnd = parsedEnd === null
      ? startTime + DEFAULT_NOTE_SECONDS
      : Math.max(startTime + MIN_NOTE_SECONDS, parsedEnd);
    const instrument = String(start.instrument || 'acoustic_piano').trim() || 'acoustic_piano';
    return {
      midi: roundedMidi,
      note: midiToNote(roundedMidi),
      time: round(startTime),
      duration: round(safeEnd - startTime),
      velocity,
      hand: roundedMidi < 60 ? 'left' : 'right',
      instrument,
      source: `muscriptor-${model}-${source}`,
    };
  }

  function accept(message) {
    if (finalResult || !message || typeof message !== 'object') {
      malformedEvents += 1;
      return;
    }
    if (message.type === 'start') {
      const index = finiteNumber(message.index);
      if (!Number.isInteger(index) || !buildNote(message, finiteNumber(message.start_time) + MIN_NOTE_SECONDS)) {
        malformedEvents += 1;
        return;
      }
      if (starts.has(index)) replacedStartEvents += 1;
      starts.set(index, message);
      return;
    }
    if (message.type === 'end') {
      const index = finiteNumber(message.start_event_index);
      const start = starts.get(index);
      if (!start) {
        unmatchedEndEvents += 1;
        return;
      }
      const note = buildNote(start, message.end_time);
      if (note) completedNotes.push(note);
      else malformedEvents += 1;
      starts.delete(index);
      return;
    }
    if (message.type === 'progress') {
      const completed = finiteNumber(message.completed);
      const total = finiteNumber(message.total);
      if (completed === null || total === null || total <= 0) return;
      progress = {
        completed: Math.max(0, Math.round(completed)),
        total: Math.max(1, Math.round(total)),
      };
      onProgress?.({ ...progress });
      return;
    }
    if (message.type === 'transcription_complete') {
      beatGrid = normalizeBeatGrid(message.beat_grid ?? message.beatGrid);
    }
  }

  function finish() {
    if (finalResult) return finalResult;
    const danglingStartEvents = starts.size;
    starts.forEach((start) => {
      const note = buildNote(start, null, 0.7);
      if (note) completedNotes.push(note);
      else malformedEvents += 1;
    });
    const onsetDelay = beatGrid?.onsetDelaySeconds || 0;
    const notes = completedNotes
      .map((note) => ({
        ...note,
        time: round(Math.max(0, note.time - onsetDelay)),
      }))
      .sort((a, b) => a.time - b.time || a.midi - b.midi || a.instrument.localeCompare(b.instrument));
    finalResult = {
      notes,
      progress: { ...progress },
      beatGrid,
      diagnostics: {
        malformedEvents,
        unmatchedEndEvents,
        replacedStartEvents,
        danglingStartEvents,
        onsetDelayAppliedSeconds: onsetDelay,
      },
    };
    return finalResult;
  }

  return { accept, finish };
}

module.exports = {
  createMuscriptorEventCollector,
  normalizeBeatGrid,
};
