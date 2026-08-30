#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import ToneMidi from '@tonejs/midi';

const { Midi } = ToneMidi;
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function name(midi) {
  return `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function deduplicate(notes) {
  const output = [];
  for (const note of notes.sort((a, b) => a.time - b.time || a.midi - b.midi)) {
    const duplicate = output.findLast((candidate) => (
      candidate.midi === note.midi && Math.abs(candidate.time - note.time) <= 0.03
    ));
    if (duplicate) {
      duplicate.duration = Math.max(duplicate.duration, note.time + note.duration - duplicate.time);
    } else {
      output.push(note);
    }
  }
  return output;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.evaluation || !args.out) {
    throw new Error('Usage: --evaluation evaluation.json --model candidate --out listening-exports');
  }
  const evaluation = JSON.parse(await fs.readFile(path.resolve(args.evaluation), 'utf8'));
  const model = args.model || 'candidate';
  const clips = evaluation?.[model]?.decodedClips;
  if (!Array.isArray(clips)) throw new Error(`${model}.decodedClips is missing from the full evaluation.`);
  const grouped = new Map();
  for (const clip of clips) {
    const notes = grouped.get(clip.songId) || [];
    for (const note of clip.notes || []) {
      notes.push({
        midi: Math.round(Number(note.midi)),
        time: Number(clip.sourceStart || 0) + Number(note.time || 0),
        duration: Math.max(0.01, Number(note.duration || 0.01)),
        velocity: 0.75,
        instrument: 'acoustic_piano',
      });
    }
    grouped.set(clip.songId, notes);
  }

  const destination = path.resolve(args.out);
  await fs.mkdir(destination, { recursive: true });
  const summary = [];
  for (const [songId, rawNotes] of grouped) {
    const notes = deduplicate(rawNotes).map((note) => ({ ...note, note: name(note.midi) }));
    const midi = new Midi();
    midi.header.setTempo(120);
    const track = midi.addTrack();
    track.name = `${songId} ${model}`;
    track.instrument.number = 0;
    for (const note of notes) {
      track.addNote({ midi: note.midi, time: note.time, duration: note.duration, velocity: note.velocity });
    }
    const base = `${songId}-${model}`;
    await fs.writeFile(path.join(destination, `${base}.mid`), Buffer.from(midi.toArray()));
    await fs.writeFile(path.join(destination, `${base}.json`), `${JSON.stringify({
      schema: 'polymath-evaluation-listening-export-v1',
      title: `${songId} (${model})`,
      instrument: 'piano',
      instrumentGroups: ['acoustic_piano'],
      sourceEvaluation: path.resolve(args.evaluation),
      model,
      notes,
    }, null, 2)}\n`);
    summary.push({ songId, model, notes: notes.length, duration: Math.max(0, ...notes.map((note) => note.time + note.duration)) });
  }
  await fs.writeFile(path.join(destination, `${model}-export-summary.json`), `${JSON.stringify(summary, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Evaluation export failed: ${error.message}\n`);
  process.exitCode = 1;
});
