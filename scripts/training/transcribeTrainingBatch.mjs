#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dotenv = require('../../server/node_modules/dotenv');
const { createRunpodServerlessClient } = require('../../server/runpodServerless');

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : '';
}

function required(value, label) {
  const cleaned = String(value || '').trim();
  if (!cleaned) throw new Error(`${label} is required`);
  return cleaned;
}

async function main() {
  dotenv.config({ path: path.resolve('server/.env'), quiet: true });
  const specPath = path.resolve(required(argument('spec'), '--spec'));
  const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
  if (!Array.isArray(spec.tasks) || !spec.tasks.length) throw new Error('The batch spec contains no tasks');
  const defaultCheckpoint = String(spec.checkpoint || 'original').trim();
  const endpointId = String(argument('endpoint-id') || spec.endpointId || process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || '').trim();
  const volumeId = String(argument('volume-id') || spec.volumeId || process.env.RUNPOD_NETWORK_VOLUME_ID || '').trim();
  const timeoutMinutes = Number(argument('timeout-minutes') || spec.timeoutMinutes || 60);
  if (!Number.isFinite(timeoutMinutes) || timeoutMinutes < 1 || timeoutMinutes > 720) {
    throw new Error('--timeout-minutes must be between 1 and 720');
  }
  const replicas = String(argument('replicas') || spec.replicas || '').trim().toLowerCase() === 'none'
    ? ''
    : process.env.RUNPOD_S3_REPLICAS;

  const client = createRunpodServerlessClient({
    endpointId,
    apiKey: process.env.RUNPOD_API_KEY,
    volumeId,
    region: process.env.RUNPOD_S3_REGION,
    s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    replicas,
    timeoutMs: timeoutMinutes * 60 * 1000,
    pollIntervalMs: 2_000,
  });
  if (!client.configured) throw new Error(`RunPod is missing: ${client.missing.join(', ')}`);

  for (const [index, task] of spec.tasks.entries()) {
    const audio = path.resolve(required(task.audio, `tasks[${index}].audio`));
    const output = path.resolve(required(task.output, `tasks[${index}].output`));
    const id = required(task.id, `tasks[${index}].id`).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 80);
    const checkpoint = String(task.checkpoint || defaultCheckpoint).trim();
    if (!/^(?:original|phase\d+-v\d+)$/i.test(checkpoint)) {
      throw new Error(`tasks[${index}].checkpoint is invalid`);
    }
    const constraints = Array.isArray(task.constraints)
      ? task.constraints.map((value) => String(value).trim()).filter(Boolean)
      : Array.isArray(spec.constraints)
        ? spec.constraints.map((value) => String(value).trim()).filter(Boolean)
        : [];
    const focusedConstraints = Array.isArray(task.focusedConstraints)
      ? task.focusedConstraints.map((value) => String(value).trim()).filter(Boolean)
      : Array.isArray(spec.focusedConstraints)
        ? spec.focusedConstraints.map((value) => String(value).trim()).filter(Boolean)
        : [];
    const instrument = String(task.instrument || spec.instrument || 'piano').trim();
    await fs.access(audio);
    await fs.mkdir(path.dirname(output), { recursive: true });
    process.stdout.write(`[${index + 1}/${spec.tasks.length}] ${task.title || id}\n`);
    let lastProgress = '';
    const result = await client.transcribe({
      job: {
        id: `training-${id}-${Date.now()}`,
        title: String(task.title || id),
        instrument,
      },
      preparedPath: audio,
      constraints,
      focusedConstraints,
      checkpointVersion: checkpoint,
      onProgress(status) {
        const text = `${status.state || ''} ${status.progress || ''}`.trim();
        if (text && text !== lastProgress) {
          process.stdout.write(`  ${text}\n`);
          lastProgress = text;
        }
      },
    });
    const record = {
      ...result,
      trainingProvenance: {
        generatedAt: new Date().toISOString(),
        endpointId,
        volumeId,
        checkpoint,
        constraints,
        focusedConstraints,
        instrument,
        sourceAudio: audio,
        phase: spec.phase || '',
      },
    };
    await fs.writeFile(output, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
    process.stdout.write(`  saved ${output} (${Array.isArray(result.notes) ? result.notes.length : 0} notes)\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`Training transcription batch failed: ${error.message}\n`);
  process.exitCode = 1;
});
