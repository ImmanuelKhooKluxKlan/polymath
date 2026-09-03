import { useMemo, useState } from 'react';
import { pianoAudio } from '../engine/audioEngine.js';

const PIANO_MIN_MIDI = 21;
const PIANO_MAX_MIDI = 108;
const MIDDLE_C_MIDI = 60;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function midiToNote(midi) {
  const value = Math.round(Number(midi));
  return `${NOTE_NAMES[((value % 12) + 12) % 12]}${Math.floor(value / 12) - 1}`;
}

function seconds(value) {
  const safe = Math.max(0, Number(value) || 0);
  return safe < 1 ? `${Math.round(safe * 1000)} ms` : `${safe.toFixed(2)} s`;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function normalizeNotes(raw) {
  return (Array.isArray(raw?.notes) ? raw.notes : [])
    .map((note, index) => ({
      id: `${note.instrument || 'unknown'}-${note.midi}-${note.time}-${index}`,
      midi: Math.round(Number(note.midi)),
      time: Math.max(0, Number(note.time) || 0),
      duration: Math.max(0.01, Number(note.duration) || 0.1),
      velocity: Math.max(0, Math.min(1, Number(note.velocity) || 0.76)),
      instrument: String(note.instrument || 'unknown').toLowerCase(),
    }))
    .filter((note) => Number.isFinite(note.midi) && Number.isFinite(note.time))
    .sort((a, b) => a.time - b.time || a.midi - b.midi);
}

function analyzePiano(raw) {
  const allNotes = normalizeNotes(raw);
  const playable = allNotes.filter((note) => note.midi >= PIANO_MIN_MIDI && note.midi <= PIANO_MAX_MIDI);
  const outsideRange = allNotes.filter((note) => note.midi < PIANO_MIN_MIDI || note.midi > PIANO_MAX_MIDI);
  const byMidi = new Map();
  playable.forEach((note) => {
    const notes = byMidi.get(note.midi) || [];
    notes.push(note);
    byMidi.set(note.midi, notes);
  });

  let rapidRepeats = 0;
  let samePitchOverlaps = 0;
  byMidi.forEach((notes) => {
    notes.sort((a, b) => a.time - b.time || b.duration - a.duration);
    for (let index = 1; index < notes.length; index += 1) {
      const previous = notes[index - 1];
      const current = notes[index];
      if (current.time - previous.time <= 0.075) rapidRepeats += 1;
      if (current.time < previous.time + previous.duration) samePitchOverlaps += 1;
    }
  });

  const keys = Array.from({ length: 88 }, (_, index) => {
    const midi = PIANO_MIN_MIDI + index;
    const notes = byMidi.get(midi) || [];
    return {
      midi,
      note: midiToNote(midi),
      black: NOTE_NAMES[midi % 12].includes('#'),
      count: notes.length,
      averageDuration: notes.length
        ? notes.reduce((sum, note) => sum + note.duration, 0) / notes.length
        : 0,
      averageVelocity: notes.length
        ? notes.reduce((sum, note) => sum + note.velocity, 0) / notes.length
        : 0,
      firstTime: notes[0]?.time ?? null,
      lastTime: notes.at(-1)?.time ?? null,
    };
  });
  const maximumKeyCount = Math.max(1, ...keys.map((key) => key.count));
  const pianoLabelNotes = playable.filter((note) => /piano|keyboard/.test(note.instrument)).length;

  return {
    allNotes,
    playable,
    outsideRange,
    keys,
    maximumKeyCount,
    pianoLabelNotes,
    uniqueKeys: byMidi.size,
    leftHandNotes: playable.filter((note) => note.midi < MIDDLE_C_MIDI).length,
    rightHandNotes: playable.filter((note) => note.midi >= MIDDLE_C_MIDI).length,
    shortNotes: playable.filter((note) => note.duration < 0.1).length,
    longNotes: playable.filter((note) => note.duration >= 2).length,
    medianDuration: median(playable.map((note) => note.duration)),
    rapidRepeats,
    samePitchOverlaps,
  };
}

function Detail({ label, value, warning = false }) {
  return (
    <article className={warning ? 'piano-test-detail warning' : 'piano-test-detail'}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

export default function PianoDetailsTester({ raw, maximumPolyphony = 0 }) {
  const analysis = useMemo(() => analyzePiano(raw), [raw]);
  const [selectedMidi, setSelectedMidi] = useState(MIDDLE_C_MIDI);
  const [audioStatus, setAudioStatus] = useState('Choose any key to audition Polymath Piano');
  const selectedKey = analysis.keys.find((key) => key.midi === selectedMidi) || analysis.keys[39];

  async function auditionKey(key) {
    setSelectedMidi(key.midi);
    setAudioStatus(`Preparing ${key.note}…`);
    const preview = {
      note: key.note,
      midi: key.midi,
      time: 0,
      duration: Math.max(0.65, key.averageDuration || 0.9),
      velocity: key.averageVelocity || 0.68,
    };
    try {
      pianoAudio.ensure();
      await pianoAudio.preloadSongNotes({ notes: [preview] });
      pianoAudio.playAt(
        preview.note,
        preview.velocity,
        preview.duration,
        pianoAudio.getCurrentTime() + 0.01,
        { source: 'manual', retriggerSameNote: true },
      );
      setAudioStatus(`${key.note} · ${key.count} detected occurrence${key.count === 1 ? '' : 's'}`);
    } catch (error) {
      setAudioStatus(error.message || `Could not audition ${key.note}`);
    }
  }

  return (
    <section className='model-lab-panel piano-details-tester'>
      <div className='model-lab-panel-heading'>
        <div>
          <p className='eyebrow'>Piano specialist v001</p>
          <h2>88-key piano details tester</h2>
          <p>Judge every detected note as a possible piano-training target, regardless of its original instrument label.</p>
        </div>
        <div className='piano-test-readiness'>
          <span>Playable range</span>
          <strong>{analysis.playable.length}/{analysis.allNotes.length}</strong>
        </div>
      </div>

      <div className='piano-test-detail-grid'>
        <Detail label='Playable notes' value={analysis.playable.length} />
        <Detail label='Unique keys used' value={`${analysis.uniqueKeys}/88`} />
        <Detail label='Left-hand notes' value={analysis.leftHandNotes} />
        <Detail label='Right-hand notes' value={analysis.rightHandNotes} />
        <Detail label='Maximum polyphony' value={maximumPolyphony || '—'} warning={maximumPolyphony > 10} />
        <Detail label='Median sustain' value={seconds(analysis.medianDuration)} />
        <Detail label='Under 100 ms' value={analysis.shortNotes} warning={analysis.shortNotes > 0} />
        <Detail label='Two seconds or longer' value={analysis.longNotes} />
        <Detail label='Rapid repeats ≤75 ms' value={analysis.rapidRepeats} warning={analysis.rapidRepeats > 0} />
        <Detail label='Same-pitch overlaps' value={analysis.samePitchOverlaps} warning={analysis.samePitchOverlaps > 0} />
        <Detail label='Outside A0–C8' value={analysis.outsideRange.length} warning={analysis.outsideRange.length > 0} />
        <Detail label='Model labelled piano' value={analysis.pianoLabelNotes} />
      </div>

      <div className='piano-key-inspector'>
        <div>
          <p className='eyebrow'>Key-coordinate coverage</p>
          <h3>Click any key to hear it</h3>
          <small>{audioStatus}</small>
        </div>
        <div className='piano-key-map' role='list' aria-label='All 88 piano keys and detected note counts'>
          {analysis.keys.map((key) => (
            <button
              key={key.midi}
              type='button'
              role='listitem'
              className={`${key.black ? 'black-key' : 'white-key'}${selectedMidi === key.midi ? ' selected' : ''}${key.count ? ' used' : ''}`}
              style={{ '--key-heat': key.count ? 0.18 + (key.count / analysis.maximumKeyCount) * 0.82 : 0 }}
              onClick={() => auditionKey(key)}
              title={`${key.note}: ${key.count} detected notes`}
            >
              <strong>{key.note}</strong>
              <span>{key.count}</span>
            </button>
          ))}
        </div>
      </div>

      <div className='piano-selected-key'>
        <div><span>Selected key</span><strong>{selectedKey.note}</strong><small>MIDI {selectedKey.midi}</small></div>
        <div><span>Occurrences</span><strong>{selectedKey.count}</strong></div>
        <div><span>Average sustain</span><strong>{selectedKey.count ? seconds(selectedKey.averageDuration) : 'No data'}</strong></div>
        <div><span>Average velocity*</span><strong>{selectedKey.count ? selectedKey.averageVelocity.toFixed(3) : 'No data'}</strong></div>
        <div><span>First → last</span><strong>{selectedKey.count ? `${seconds(selectedKey.firstTime)} → ${seconds(selectedKey.lastTime)}` : 'No detections'}</strong></div>
      </div>

      <p className='piano-test-caveat'>* Polymath currently returns synthetic velocity. Use it to test playback plumbing, not to judge the performer’s true softness. Alignment and verified MIDI will provide the supervised piano target.</p>
    </section>
  );
}

export { analyzePiano, midiToNote };
