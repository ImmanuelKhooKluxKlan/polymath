#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { postProcessMuscriptorResult } = require('../../server/muscriptorPostprocess.js');

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

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.output) {
    throw new Error('Usage: node scripts/training/replayPianoRoute.mjs --input raw.json --output cleaned.json [--mode full|instrumental] [--audio prepared.wav]');
  }
  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  const audioPath = args.audio ? path.resolve(args.audio) : null;
  if (audioPath) await fs.access(audioPath);
  const mode = args.mode === 'instrumental' ? 'instrumental' : 'full';
  const raw = JSON.parse(await fs.readFile(inputPath, 'utf8'));
  const cleaned = postProcessMuscriptorResult(raw, {
    instrument: 'piano',
    playbackMode: mode,
    ...(audioPath ? { preparedPath: audioPath } : {}),
  });
  cleaned.playbackMode = mode;
  cleaned.vocalMelodyIncluded = mode === 'full';
  cleaned.selectedInstrument = 'piano';
  cleaned.arrangementProfile = 'pre-arranger-piano-cleanup-v3';
  cleaned.replayProvenance = {
    schema: 'polymath-piano-cleanup-replay-v1',
    sourceAudio: audioPath,
    sourceAudioSha256: audioPath
      ? createHash('sha256').update(await fs.readFile(audioPath)).digest('hex')
      : null,
    sourceDynamicsApplied: cleaned.transcriptionCleanup?.sourceDynamicsApplied === true,
    onsetClockTreatment: 'preserved-from-input; no tempo or onset-delay detection in this replay',
    focusedPassTreatment: 'preserved-from-input; no additional transcription pass in this replay',
    interpretation: 'Replays piano cleanup only. Input must already contain any worker tempo-clock correction and focused-pass fusion to represent those production stages.',
  };
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(cleaned, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    inputNotes: Array.isArray(raw.notes) ? raw.notes.length : 0,
    outputNotes: Array.isArray(cleaned.notes) ? cleaned.notes.length : 0,
    removedDuplicates: cleaned.transcriptionCleanup?.removedDuplicateNotes || 0,
    shortenedOverlaps: cleaned.transcriptionCleanup?.shortenedSameKeyOverlaps || 0,
    sourceDynamicsApplied: cleaned.replayProvenance.sourceDynamicsApplied,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`Piano route replay failed: ${error.message}\n`);
  process.exitCode = 1;
});
