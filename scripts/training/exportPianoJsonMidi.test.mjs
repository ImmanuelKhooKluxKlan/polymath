import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import ToneMidi from '@tonejs/midi';

import { exportPianoJsonMidi, normalizeListeningNotes } from './exportPianoJsonMidi.mjs';

const { Midi } = ToneMidi;

test('normalizes only valid notes and follows the browser audio-duration contract', () => {
  const notes = normalizeListeningNotes({
    notes: [
      { midi: 64, time: 1, duration: 2, audioDuration: 0.4, velocity: 0.8 },
      { midi: 60, time: 0, duration: 0.5 },
      { midi: 200, time: 0, duration: 1 },
    ],
  });
  assert.deepEqual(notes.map((note) => note.midi), [60, 64]);
  assert.equal(notes[1].duration, 0.4);
  assert.equal(notes[0].velocity, 0.75);
});

test('exports an arranged JSON file as a readable piano MIDI and audit summary', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-listening-export-'));
  const input = path.join(root, 'input.json');
  const output = path.join(root, 'out');
  await fs.writeFile(input, JSON.stringify({
    bpm: 96,
    notes: [
      { midi: 60, time: 0.25, duration: 1, audioDuration: 0.35, velocity: 0.6 },
      { midi: 67, time: 1, duration: 0.5, velocity: 0.9 },
    ],
  }));

  const summary = await exportPianoJsonMidi({
    input,
    outputDirectory: output,
    label: 'candidate v012',
  });
  const bytes = await fs.readFile(summary.midiFile);
  const midi = new Midi(bytes);
  assert.equal(summary.label, 'candidate-v012');
  assert.equal(summary.noteCount, 2);
  assert.equal(summary.playbackContract.sustainPedalSynthesized, false);
  assert.equal(midi.tracks[0].instrument.number, 0);
  assert.equal(midi.tracks[0].notes.length, 2);
  assert.ok(Math.abs(midi.tracks[0].notes[0].duration - 0.35) < 0.01);
  assert.equal(Math.round(midi.header.tempos[0].bpm), 96);
});
