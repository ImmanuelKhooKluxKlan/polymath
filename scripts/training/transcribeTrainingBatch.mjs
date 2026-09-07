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

  const client = createRunpodServerlessClient({
    endpointId: process.env.RUNPOD_SERVERLESS_ENDPOINT_ID,
    apiKey: process.env.RUNPOD_API_KEY,
    volumeId: process.env.RUNPOD_NETWORK_VOLUME_ID,
    region: process.env.RUNPOD_S3_REGION,
    s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    replicas: process.env.RUNPOD_S3_REPLICAS,
    timeoutMs: Number(process.env.MUSCRIPTOR_TIMEOUT_MS) || 60 * 60 * 1000,
    pollIntervalMs: 2_000,
  });
  if (!client.configured) throw new Error(`RunPod is missing: ${client.missing.join(', ')}`);

  for (const [index, task] of spec.tasks.entries()) {
    const audio = path.resolve(required(task.audio, `tasks[${index}].audio`));
    const output = path.resolve(required(task.output, `tasks[${index}].output`));
    const id = required(task.id, `tasks[${index}].id`).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 80);
    await fs.access(audio);
    await fs.mkdir(path.dirname(output), { recursive: true });
    process.stdout.write(`[${index + 1}/${spec.tasks.length}] ${task.title || id}\n`);
    let lastProgress = '';
    const result = await client.transcribe({
      job: {
        id: `training-${id}-${Date.now()}`,
        title: String(task.title || id),
        instrument: 'band',
      },
      preparedPath: audio,
      constraints: [],
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
        endpointId: process.env.RUNPOD_SERVERLESS_ENDPOINT_ID,
        checkpoint: 'endpoint-default-muscriptor-large',
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
