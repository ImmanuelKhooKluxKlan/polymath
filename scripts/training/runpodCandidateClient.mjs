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
    throw new Error('Usage: --dataset phase-1-v001 --version phase1-v001 [--base-version original] [--conditioning-mode instrument|unconditioned] [--endpoint-id id] [--timeout-minutes 60] [--result result.json]');
  }
  if (!/^[a-z0-9][a-z0-9-]{2,50}$/.test(args.dataset)) {
    throw new Error('--dataset must be 3-51 lowercase letters, digits, or hyphens, starting with a letter or digit');
  }
  if (!/^phase\d+-v\d{3,}$/.test(args.version)) {
    throw new Error('--version must look like phase1-v001');
  }
  const baseVersion = args['base-version'] || 'original';
  if (baseVersion !== 'original' && !/^phase\d+-v\d{3,}$/.test(baseVersion)) {
    throw new Error('--base-version must be original or look like phase1-v001');
  }
  await loadEnvironment(path.resolve('server/.env'));
  const endpoint = String(args['endpoint-id'] || '').trim() || required('RUNPOD_SERVERLESS_ENDPOINT_ID');
  const apiKey = required('RUNPOD_API_KEY');
  const timeoutMinutes = Number(args['timeout-minutes'] || 60);
  if (!Number.isFinite(timeoutMinutes) || timeoutMinutes < 1 || timeoutMinutes > 720) {
    throw new Error('--timeout-minutes must be between 1 and 720');
  }
  const timeoutMs = timeoutMinutes * 60 * 1000;
  const resultDestination = args.result ? path.resolve(args.result) : '';
  const writeResult = async (record) => {
    if (!resultDestination) return;
    await fs.mkdir(path.dirname(resultDestination), { recursive: true });
    await fs.writeFile(resultDestination, `${JSON.stringify(record, null, 2)}\n`);
  };
  const requestTimeoutSeconds = Number(args['request-timeout-seconds'] || 30);
  if (!Number.isFinite(requestTimeoutSeconds) || requestTimeoutSeconds < 5 || requestTimeoutSeconds > 120) {
    throw new Error('--request-timeout-seconds must be between 5 and 120');
  }
  const requestTimeoutMs = requestTimeoutSeconds * 1000;
  const baseUrl = `https://api.runpod.ai/v2/${endpoint}`;
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
  const request = async (pathname, options = {}) => {
    const method = String(options.method || 'GET').toUpperCase();
    // GET polling is idempotent and may be retried after a transport timeout.
    // Never retry POST /run automatically: the server may have accepted a job
    // even when its response was lost, which would create duplicate GPU work.
    const attempts = method === 'GET' ? 4 : 1;
    let lastError;
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
      timeout.unref?.();
      try {
        const response = await fetch(`${baseUrl}${pathname}`, {
          ...options,
          headers,
          signal: controller.signal,
        });
        const text = await response.text();
        const payload = text ? JSON.parse(text) : {};
        if (!response.ok) {
          const error = new Error(`RunPod HTTP ${response.status}: ${payload.error || payload.message || text}`);
          error.retryable = response.status === 429 || response.status >= 500;
          throw error;
        }
        return payload;
      } catch (error) {
        lastError = error;
        const retryable = method === 'GET'
          && (error?.name === 'TimeoutError' || error?.name === 'AbortError' || error?.retryable);
        if (!retryable || attempt === attempts) throw error;
        await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
      } finally {
        clearTimeout(timeout);
      }
    }
    throw lastError;
  };

  const submitted = await request('/run', {
    method: 'POST',
    body: JSON.stringify({
      input: {
        action: 'train_piano_candidate',
        dataset_id: args.dataset,
        version: args.version,
        base_version: baseVersion,
        epochs: Number(args.epochs || 1),
        train_last_layers: Number(args.layers || 1),
        learning_rate: Number(args['learning-rate'] || 0.000002),
        timing_token_weight: Number(args['timing-token-weight'] || 1.15),
        note_off_token_weight: Number(args['note-off-token-weight'] || 1.25),
        eos_token_weight: Number(args['eos-token-weight'] || 1.20),
        conditioning_mode: args['conditioning-mode'] || 'instrument',
        rights_acknowledgement: 'I_HAVE_TRAINING_RIGHTS',
      },
      policy: {
        executionTimeout: timeoutMs,
        ttl: Math.min(7 * 24 * 60 * 60 * 1000, Math.max(timeoutMs * 2, 2 * 60 * 60 * 1000)),
      },
    }),
  });
  if (!submitted.id) throw new Error('RunPod did not return a training job id');
  process.stdout.write(`JOB_ID=${submitted.id}\n`);
  await writeResult({
    jobId: submitted.id,
    submittedAt: new Date().toISOString(),
    status: submitted.status || 'IN_QUEUE',
    endpointId: endpoint,
    action: 'train_piano_candidate',
    datasetId: args.dataset,
    version: args.version,
    baseVersion,
  });

  const deadline = Date.now() + timeoutMs;
  let previous = '';
  while (Date.now() < deadline) {
    const status = await request(`/status/${encodeURIComponent(submitted.id)}`);
    const message = `${status.status || 'UNKNOWN'} ${status.progress || status.output?.progress || ''}`.trim();
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
      await writeResult(record);
      process.stdout.write(`${JSON.stringify(record, null, 2)}\n`);
      return;
    }
    if (['CANCELLED', 'FAILED', 'TIMED_OUT'].includes(status.status)) {
      let failure = status.output?.error || status.error || status.status;
      if (typeof failure === 'string') {
        try {
          const parsed = JSON.parse(failure);
          failure = parsed.error_message || parsed.error_type || failure;
        } catch {
          // RunPod also returns ordinary non-JSON failure strings.
        }
      }
      throw new Error(typeof failure === 'string' ? failure : JSON.stringify(failure));
    }
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  throw new Error(`RunPod training exceeded the ${timeoutMinutes}-minute client deadline`);
}

main().catch((error) => {
  process.stderr.write(`RunPod candidate training failed: ${error.message}\n`);
  process.exitCode = 1;
});
