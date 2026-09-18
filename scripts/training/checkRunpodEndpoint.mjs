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

async function main() {
  const args = parseArguments(process.argv.slice(2));
  await loadEnvironment(path.resolve(args.env || 'server/.env'));
  const endpoint = String(args['endpoint-id'] || process.env.RUNPOD_SERVERLESS_ENDPOINT_ID || '').trim();
  const apiKey = String(process.env.RUNPOD_API_KEY || '').trim();
  if (!endpoint) throw new Error('RunPod endpoint ID is missing');
  if (!apiKey) throw new Error('RUNPOD_API_KEY is missing');
  const response = await fetch(`https://api.runpod.ai/v2/${encodeURIComponent(endpoint)}/health`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(30_000),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`RunPod health HTTP ${response.status}: ${body}`);
  const payload = body ? JSON.parse(body) : {};
  process.stdout.write(`${JSON.stringify({ endpoint, checkedAt: new Date().toISOString(), ...payload }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod health check failed: ${error.message}\n`);
  process.exitCode = 1;
});
