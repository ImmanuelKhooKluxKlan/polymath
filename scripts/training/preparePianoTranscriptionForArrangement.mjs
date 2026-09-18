#!/usr/bin/env node

/**
 * Reproduce the live server's audio-dynamics cleanup for offline experiments.
 *
 * Raw RunPod snapshots contain pitch/onset/offset but a placeholder velocity.
 * Training directly from those snapshots silently tests a different pipeline
 * from production. This utility joins the original prepared WAV back to the
 * raw JSON and runs the exact server postprocessor before Python arrangement.
 */

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const { applySourceDynamics, postProcessMuscriptorResult, readWavRmsEnvelope } = require(
  path.resolve(scriptDirectory, '../../server/muscriptorPostprocess.js'),
);

function argumentsByName(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith('--')) continue;
    if (item === '--dynamics-only') {
      values.set('dynamics-only', 'true');
      continue;
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`${item} requires a value.`);
    values.set(item.slice(2), value);
    index += 1;
  }
  return values;
}

const args = argumentsByName(process.argv.slice(2));
const inputPath = path.resolve(args.get('input') || '');
const audioPath = path.resolve(args.get('audio') || '');
const outputPath = path.resolve(args.get('output') || '');
if (!args.get('input') || !args.get('audio') || !args.get('output')) {
  throw new Error('Usage: --input RAW.json --audio PREPARED.wav --output CLEAN.json');
}
for (const [label, target] of [['input', inputPath], ['audio', audioPath]]) {
  if (!fs.existsSync(target)) throw new Error(`${label} file does not exist: ${target}`);
}

const payload = JSON.parse(fs.readFileSync(inputPath, 'utf8').replace(/^\uFEFF/, ''));
const dynamicsOnly = args.get('dynamics-only') === 'true';
const result = dynamicsOnly
  ? structuredClone(payload)
  : postProcessMuscriptorResult(payload, {
      instrument: 'piano',
      playbackMode: args.get('mode') || 'full',
      preparedPath: audioPath,
    });
const dynamicsApplied = dynamicsOnly
  ? applySourceDynamics(result.notes, readWavRmsEnvelope(audioPath))
  : result?.transcriptionCleanup?.sourceDynamicsApplied;
if (!dynamicsApplied) {
  throw new Error(
    'The WAV did not yield a usable RMS envelope; refusing to write a flat-velocity training input.',
  );
}
result.trainingPreparation = {
  schema: 'polymath-production-parity-piano-input-v1',
  rawInput: inputPath,
  sourceAudio: audioPath,
  sourceDynamicsApplied: true,
  sourceCoordinatesPreserved: dynamicsOnly,
  note: 'Offline research input produced by the same postprocessor used by the live server.',
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
const temporaryPath = `${outputPath}.tmp`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  output: outputPath,
  notes: result.notes.length,
  sourceDynamicsApplied: true,
  sourceCoordinatesPreserved: dynamicsOnly,
  velocityMinimum: Math.min(...result.notes.map((note) => Number(note.velocity))),
  velocityMaximum: Math.max(...result.notes.map((note) => Number(note.velocity))),
}, null, 2)}\n`);
