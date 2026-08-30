#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import ToneMidi from '@tonejs/midi';

const { Midi } = ToneMidi;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    values[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function noteName(midi) {
  return `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function included(track, melodyPattern) {
  const name = `${track.name || ''} ${track.instrument?.name || ''} ${track.instrument?.family || ''}`.toLowerCase();
  if (/drum|percussion|kit/.test(name) || track.channel === 9) return false;
  if (/piano/.test(name)) return true;
  return melodyPattern ? melodyPattern.test(name) : false;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.output) {
    throw new Error('Usage: node scripts/training/buildPianoReductionTarget.mjs --input arrangement.mid --output piano-target.json [--melody "flute|voice"]');
  }
  const source = path.resolve(args.input);
  const destination = path.resolve(args.output);
  const melodyPattern = args.melody ? new RegExp(args.melody, 'i') : null;
  const midi = new Midi(await fs.readFile(source));
  const selectedTracks = midi.tracks
    .map((track, trackIndex) => ({ track, trackIndex }))
    .filter(({ track }) => included(track, melodyPattern));
  const notes = selectedTracks.flatMap(({ track, trackIndex }) => track.notes.map((note) => ({
    midi: note.midi,
    note: noteName(note.midi),
    time: Number(note.time.toFixed(6)),
    duration: Number(Math.max(0.01, note.duration).toFixed(6)),
    velocity: Number(Math.max(0.05, Math.min(1, note.velocity || 0.75)).toFixed(4)),
    instrument: 'acoustic_piano',
    sourceTrack: trackIndex,
    sourceInstrument: track.instrument?.name || 'unknown',
    role: /piano/i.test(`${track.instrument?.name || ''} ${track.instrument?.family || ''}`)
      ? 'piano-accompaniment'
      : 'melody-revoiced-on-piano',
  }))).sort((a, b) => a.time - b.time || a.midi - b.midi);
  if (!notes.length) throw new Error('No piano or requested melody tracks were found.');

  const payload = {
    schema: 'polymath-piano-reduction-target-v1',
    title: path.basename(source, path.extname(source)),
    sourceMidi: source,
    sourceMidiBytes: (await fs.stat(source)).size,
    bpm: midi.header.tempos[0]?.bpm || 120,
    instrument: 'piano',
    instrumentGroups: ['acoustic_piano'],
    reduction: {
      method: 'preserve-authored-piano-plus-selected-melody-v1',
      selectedTrackCount: selectedTracks.length,
      selectedTracks: selectedTracks.map(({ track, trackIndex }) => ({
        trackIndex,
        name: track.name || '',
        instrument: track.instrument?.name || '',
        family: track.instrument?.family || '',
        notes: track.notes.length,
      })),
      melodyPattern: args.melody || '',
      warning: 'Research target only. This relabels selected authored melody notes as acoustic piano; it does not create guitar supervision.',
    },
    notes,
  };
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, `${JSON.stringify(payload, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ output: destination, notes: notes.length, tracks: selectedTracks.length }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Piano target build failed: ${error.message}\n`);
  process.exitCode = 1;
});
