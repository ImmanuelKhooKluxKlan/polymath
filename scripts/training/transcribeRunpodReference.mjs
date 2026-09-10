#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createRunpodServerlessClient } = require('../../server/runpodServerless.js');

function argumentsFrom(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  if (!args.input || !args.output) {
    throw new Error('Usage: --input prepared.wav --output raw.json [--checkpoint phase1-v002] [--instrument piano]');
  }
  const repoRoot = path.resolve(import.meta.dirname, '..', '..');
  process.loadEnvFile(path.join(repoRoot, 'server', '.env'));
  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  const checkpoint = args.checkpoint || 'phase1-v002';
  const client = createRunpodServerlessClient({
    endpointId: process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || process.env.RUNPOD_ENDPOINT_ID,
    apiKey: process.env.RUNPOD_API_KEY,
    volumeId: process.env.RUNPOD_NETWORK_VOLUME_ID,
    region: process.env.RUNPOD_S3_REGION,
    s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    replicas: process.env.RUNPOD_S3_REPLICAS,
    inferenceVersion: checkpoint,
    timeoutMs: 45 * 60 * 1000,
    pollIntervalMs: 2_000,
  });
  if (!client.configured) throw new Error(`RunPod configuration is missing: ${client.missing.join(', ')}`);
  let previousState = '';
  const result = await client.transcribe({
    job: {
      id: `research-${Date.now()}`,
      title: args.title || path.basename(inputPath, path.extname(inputPath)),
      instrument: args.instrument || 'piano',
    },
    preparedPath: inputPath,
    constraints: [],
    checkpointVersion: checkpoint,
    onProgress({ state, progress }) {
      const message = `${state}:${String(progress || '')}`;
      if (message === previousState) return;
      previousState = message;
      process.stdout.write(`[RunPod] ${state}${progress ? ` - ${progress}` : ''}\n`);
    },
  });
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.partial`;
  await fs.writeFile(temporaryPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  await fs.rename(temporaryPath, outputPath);
  process.stdout.write(`${JSON.stringify({ output: outputPath, notes: result.notes.length, instrumentGroups: result.instrumentGroups })}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod reference transcription failed: ${error.message}\n`);
  process.exitCode = 1;
});
