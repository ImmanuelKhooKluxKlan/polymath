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
  const remoteInput = String(args['remote-input'] || '').trim();
  if ((!args.input && !remoteInput) || (args.input && remoteInput) || !args.output) {
    throw new Error('Usage: (--input prepared.wav | --remote-input /runpod-volume/jobs/prepared.wav) --output raw.json [--checkpoint phase1-v002] [--instrument piano] [--constraints acoustic_piano,electric_piano] [--focused-constraints voice] [--endpoint-id id] [--volume-id id] [--replicas none] [--timeout-minutes 45]');
  }
  if (remoteInput) {
    const segments = remoteInput.split('/');
    if (!remoteInput.startsWith('/runpod-volume/jobs/')
      || !remoteInput.toLowerCase().endsWith('.wav')
      || segments.includes('..')) {
      throw new Error('--remote-input must be a WAV inside /runpod-volume/jobs');
    }
  }
  const repoRoot = path.resolve(import.meta.dirname, '..', '..');
  process.loadEnvFile(path.join(repoRoot, 'server', '.env'));
  const inputPath = args.input ? path.resolve(args.input) : null;
  const outputPath = path.resolve(args.output);
  const checkpoint = args.checkpoint || 'phase1-v002';
  const constraints = String(args.constraints || '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
  const focusedConstraints = String(args['focused-constraints'] || '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
  const timeoutMinutes = Number(args['timeout-minutes'] || 45);
  if (!Number.isFinite(timeoutMinutes) || timeoutMinutes < 1 || timeoutMinutes > 720) {
    throw new Error('--timeout-minutes must be between 1 and 720');
  }
  const replicas = String(args.replicas || '').trim().toLowerCase() === 'none'
    ? ''
    : process.env.RUNPOD_S3_REPLICAS;
  const client = createRunpodServerlessClient({
    endpointId: args['endpoint-id'] || process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || process.env.RUNPOD_ENDPOINT_ID,
    apiKey: process.env.RUNPOD_API_KEY,
    volumeId: args['volume-id'] || process.env.RUNPOD_NETWORK_VOLUME_ID,
    region: process.env.RUNPOD_S3_REGION,
    s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    replicas,
    inferenceVersion: checkpoint,
    timeoutMs: timeoutMinutes * 60 * 1000,
    pollIntervalMs: 2_000,
  });
  if (!client.configured) throw new Error(`RunPod configuration is missing: ${client.missing.join(', ')}`);
  let previousState = '';
  const reportProgress = ({ state, progress }) => {
    const message = `${state}:${String(progress || '')}`;
    if (message === previousState) return;
    previousState = message;
    process.stdout.write(`[RunPod] ${state}${progress ? ` - ${progress}` : ''}\n`);
  };
  let result;
  if (remoteInput) {
    const submitted = await client.submitAction({
      audio_path: remoteInput,
      delete_audio: false,
      title: args.title || path.posix.basename(remoteInput, path.posix.extname(remoteInput)),
      instrument: args.instrument || 'piano',
      instruments: constraints,
      ...(focusedConstraints.length ? { focused_instruments: focusedConstraints } : {}),
      checkpoint_version: checkpoint,
    }, {
      executionTimeout: timeoutMinutes * 60 * 1000,
      ttl: Math.min(7 * 24 * 60 * 60 * 1000, timeoutMinutes * 120 * 1000),
    });
    const remoteJobId = String(submitted.id || '').trim();
    if (!remoteJobId) throw new Error('RunPod Serverless did not return a job ID.');
    process.stdout.write(`JOB_ID=${remoteJobId}\n`);
    const deadline = Date.now() + timeoutMinutes * 60 * 1000;
    while (Date.now() < deadline) {
      const status = await client.getJobStatus(remoteJobId);
      const state = String(status.status || '').trim().toUpperCase();
      reportProgress({ state, progress: status.progress });
      if (state === 'COMPLETED') {
        if (!status.output || !Array.isArray(status.output.notes)) {
          throw new Error(status.output?.error || 'RunPod completed without a Polymath note result.');
        }
        result = status.output;
        break;
      }
      if (['CANCELLED', 'FAILED', 'TIMED_OUT'].includes(state)) {
        throw new Error(status.error || `RunPod Serverless job ${state.toLowerCase()}.`);
      }
      await new Promise((resolve) => setTimeout(resolve, 2_000));
    }
    if (!result) {
      await client.cancelJob(remoteJobId).catch(() => {});
      throw new Error(`RunPod Serverless exceeded the ${timeoutMinutes}-minute processing limit.`);
    }
  } else {
    result = await client.transcribe({
      job: {
        id: `research-${Date.now()}`,
        title: args.title || path.basename(inputPath, path.extname(inputPath)),
        instrument: args.instrument || 'piano',
      },
      preparedPath: inputPath,
      constraints,
      focusedConstraints,
      checkpointVersion: checkpoint,
      onProgress: reportProgress,
    });
  }
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.partial`;
  await fs.writeFile(temporaryPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  await fs.rename(temporaryPath, outputPath);
  process.stdout.write(`${JSON.stringify({
    output: outputPath,
    checkpoint,
    input: remoteInput || inputPath,
    constraints,
    focusedConstraints,
    endpointId: args['endpoint-id'] || process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || process.env.RUNPOD_ENDPOINT_ID,
    volumeId: args['volume-id'] || process.env.RUNPOD_NETWORK_VOLUME_ID,
    notes: result.notes.length,
    instrumentGroups: result.instrumentGroups,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod reference transcription failed: ${error.message}\n`);
  process.exitCode = 1;
});
