#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import ToneMidi from '@tonejs/midi';

const { Midi } = ToneMidi;

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (!flag.startsWith('--')) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`${flag} requires a value`);
    }
    values[flag.slice(2)] = value;
    index += 1;
  }
  return values;
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function normalizeListeningNotes(payload) {
  if (!Array.isArray(payload?.notes)) {
    throw new Error('Input JSON must contain a notes array');
  }
  return payload.notes
    .map((event, sourceIndex) => {
      const midi = Math.round(finiteNumber(event?.midi, Number.NaN));
      const time = finiteNumber(event?.time, Number.NaN);
      const duration = finiteNumber(event?.audioDuration ?? event?.duration, Number.NaN);
      if (!Number.isFinite(midi) || midi < 0 || midi > 127) return null;
      if (!Number.isFinite(time) || time < 0) return null;
      if (!Number.isFinite(duration) || duration <= 0) return null;
      return {
        midi,
        time,
        duration: clamp(duration, 0.01, 60),
        velocity: clamp(finiteNumber(event?.velocity, 0.75), 0.01, 1),
        sourceIndex,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.time - right.time || left.midi - right.midi);
}

export async function exportPianoJsonMidi({ input, outputDirectory, label }) {
  const inputPath = path.resolve(input);
  const destination = path.resolve(outputDirectory);
  const payload = JSON.parse(await fs.readFile(inputPath, 'utf8'));
  const notes = normalizeListeningNotes(payload);
  if (!notes.length) throw new Error('Input contains no valid playable notes');

  const safeLabel = String(label || path.parse(inputPath).name)
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'piano-listening-export';
  const midi = new Midi();
  midi.header.setTempo(clamp(finiteNumber(payload?.bpm, 120), 20, 300));
  const track = midi.addTrack();
  track.name = safeLabel;
  track.instrument.number = 0;
  for (const note of notes) {
    track.addNote({
      midi: note.midi,
      time: note.time,
      duration: note.duration,
      velocity: note.velocity,
    });
  }

  await fs.mkdir(destination, { recursive: true });
  const midiPath = path.join(destination, `${safeLabel}.mid`);
  const summaryPath = path.join(destination, `${safeLabel}.summary.json`);
  const durationSeconds = Math.max(...notes.map((note) => note.time + note.duration));
  const summary = {
    schema: 'polymath-piano-listening-export-v1',
    label: safeLabel,
    sourceJson: inputPath,
    midiFile: midiPath,
    noteCount: notes.length,
    durationSeconds: Number(durationSeconds.toFixed(4)),
    bpm: midi.header.tempos[0]?.bpm || 120,
    playbackContract: {
      duration: 'audioDuration ?? duration',
      velocity: 'velocity ?? 0.75',
      leadingSilencePreserved: true,
      sustainPedalSynthesized: false,
    },
  };
  await fs.writeFile(midiPath, Buffer.from(midi.toArray()));
  await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.out) {
    throw new Error('Usage: --input arranged.json --out listening-folder [--label candidate-v012]');
  }
  const summary = await exportPianoJsonMidi({
    input: args.input,
    outputDirectory: args.out,
    label: args.label,
  });
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

const isMain = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main().catch((error) => {
    process.stderr.write(`Piano listening export failed: ${error.message}\n`);
    process.exitCode = 1;
  });
}
