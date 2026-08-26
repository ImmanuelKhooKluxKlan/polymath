const test = require('node:test');
const assert = require('node:assert/strict');
const { analyzeTranscription, midiToNote } = require('./modelLabAnalysis');
const { isLoopback } = require('./modelLab');

test('Model Lab reports instruments, pitch, MIDI, polyphony, and rapid repeats from raw notes', () => {
  const result = analyzeTranscription({
    transcriptionProvider: 'Tester v001',
    notes: [
      { instrument: 'acoustic_piano', midi: 60, time: 0, duration: 0.5, velocity: 0.78 },
      { instrument: 'acoustic_piano', midi: 60, time: 0.05, duration: 0.4, velocity: 0.78 },
      { instrument: 'acoustic_guitar', midi: 52, time: 0, duration: 1.1, velocity: 0.78 },
      { instrument: 'voice', midi: 67, time: 0.02, duration: 0.7, velocity: 0.78 },
    ],
  });

  assert.equal(result.headline.validNotes, 4);
  assert.equal(result.headline.detectedInstrumentGroups, 3);
  assert.equal(result.headline.pitchRange, 'E3–G4');
  assert.equal(result.pitch.minimumMidi, 52);
  assert.equal(result.pitch.maximumMidi, 67);
  assert.equal(result.midi.noteOnEvents, 4);
  assert.equal(result.midi.noteOffEvents, 4);
  assert.equal(result.timing.rapidRepeats75ms, 1);
  assert.equal(result.timing.samePitchOverlaps, 1);
  assert.equal(result.timing.maximumPolyphony, 4);
  assert.equal(result.instruments[0].instrument, 'acoustic_piano');
  assert.equal(result.model.cleanupApplied, false);
});

test('Model Lab rejects malformed notes without corrupting valid statistics', () => {
  const result = analyzeTranscription({
    notes: [
      { instrument: 'piano', midi: 21, time: 0, duration: 1 },
      { instrument: 'piano', midi: 130, time: 0, duration: 1 },
      { instrument: 'piano', midi: 22, time: -1, duration: 1 },
      { instrument: 'piano', midi: 23, time: 0, duration: 0 },
    ],
  });

  assert.equal(result.headline.validNotes, 1);
  assert.equal(result.headline.rejectedMalformedNotes, 3);
  assert.equal(result.headline.pitchRange, 'A0–A0');
});

test('MIDI note naming and localhost boundary are explicit', () => {
  assert.equal(midiToNote(21), 'A0');
  assert.equal(midiToNote(60), 'C4');
  assert.equal(midiToNote(108), 'C8');
  assert.equal(isLoopback({ ip: '::1', socket: {} }), true);
  assert.equal(isLoopback({ ip: '::ffff:127.0.0.1', socket: {} }), true);
  assert.equal(isLoopback({ ip: '203.0.113.8', socket: {} }), false);
});
