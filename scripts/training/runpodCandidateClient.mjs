#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

function parseArguments(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    result[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  return result;
}

async function loadEnvironment(filename) {
  const content = await fs.readFile(filename, 'utf8');
  for (const line of content.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2];
  }
}

function required(name) {
  const value = String(process.env[name] || '').trim();
  if (!value) throw new Error(`${name} is missing`);
  return value;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.dataset || !args.version) {
    throw new Error('Usage: --dataset phase-1-v001 --version phase1-v001 [--base-version original] [--result result.json]');
  }
  await loadEnvironment(path.resolve('server/.env'));
  const endpoint = required('RUNPOD_SERVERLESS_ENDPOINT_ID');
  const apiKey = required('RUNPOD_API_KEY');
  const baseUrl = `https://api.runpod.ai/v2/${endpoint}`;
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
  const request = async (pathname, options = {}) => {
    const response = await fetch(`${baseUrl}${pathname}`, { ...options, headers });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(`RunPod HTTP ${response.status}: ${payload.error || payload.message || text}`);
    }
    return payload;
  };

  const submitted = await request('/run', {
    method: 'POST',
    body: JSON.stringify({
      input: {
        action: 'train_piano_candidate',
        dataset_id: args.dataset,
        version: args.version,
        base_version: args['base-version'] || 'original',
        epochs: Number(args.epochs || 1),
        train_last_layers: Number(args.layers || 1),
        learning_rate: Number(args['learning-rate'] || 0.000002),
        timing_token_weight: Number(args['timing-token-weight'] || 1.15),
        note_off_token_weight: Number(args['note-off-token-weight'] || 1.25),
        eos_token_weight: Number(args['eos-token-weight'] || 1.20),
        rights_acknowledgement: 'I_HAVE_TRAINING_RIGHTS',
      },
      policy: {
        executionTimeout: 60 * 60 * 1000,
        ttl: 2 * 60 * 60 * 1000,
      },
    }),
  });
  if (!submitted.id) throw new Error('RunPod did not return a training job id');
  process.stdout.write(`JOB_ID=${submitted.id}\n`);

  const deadline = Date.now() + 60 * 60 * 1000;
  let previous = '';
  while (Date.now() < deadline) {
    const status = await request(`/status/${encodeURIComponent(submitted.id)}`);
    const message = `${status.status || 'UNKNOWN'} ${status.output?.progress || ''}`.trim();
    if (message !== previous) {
      process.stdout.write(`${message}\n`);
      previous = message;
    }
    if (status.status === 'COMPLETED') {
      const record = {
        jobId: submitted.id,
        completedAt: new Date().toISOString(),
        ...status.output,
      };
      if (args.result) {
        const destination = path.resolve(args.result);
        await fs.mkdir(path.dirname(destination), { recursive: true });
        await fs.writeFile(destination, `${JSON.stringify(record, null, 2)}\n`);
      }
      process.stdout.write(`${JSON.stringify(record, null, 2)}\n`);
      return;
    }
    if (['CANCELLED', 'FAILED', 'TIMED_OUT'].includes(status.status)) {
      throw new Error(status.output?.error || JSON.stringify(status.output) || status.status);
    }
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  throw new Error('RunPod training exceeded the one-hour client deadline');
}

main().catch((error) => {
  process.stderr.write(`RunPod candidate training failed: ${error.message}\n`);
  process.exitCode = 1;
});
