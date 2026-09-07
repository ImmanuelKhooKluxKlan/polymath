#!/usr/bin/env node

import crypto from 'node:crypto';
import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const requireFromServer = createRequire(fileURLToPath(new URL('../../server/package.json', import.meta.url)));
const dotenv = requireFromServer('dotenv');
const { GetObjectCommand, ListObjectsV2Command, S3Client } = requireFromServer('@aws-sdk/client-s3');

function argumentsFrom(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    result[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return result;
}

function safeVersion(value) {
  const version = String(value || '').trim();
  if (!/^[a-zA-Z0-9._-]{1,40}$/.test(version)) {
    throw new Error('The version may contain only letters, numbers, dot, underscore, and hyphen.');
  }
  return version;
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  if (!args.env || !args.version) {
    throw new Error('Usage: node download_evidence.mjs --env /private/server.env --chatboss-env /private/chatboss.env --version v002 [--output evidence/v002]');
  }
  dotenv.config({ path: path.resolve(args.env), quiet: true });
  if (args['chatboss-env']) {
    dotenv.config({ path: path.resolve(args['chatboss-env']), quiet: true, override: true });
  }

  const version = safeVersion(args.version);
  const volumeId = String(process.env.RUNPOD_CHAT_BOSS_NETWORK_VOLUME_ID || '').trim();
  const region = String(process.env.RUNPOD_CHAT_BOSS_S3_REGION || '').trim();
  const endpoint = String(process.env.RUNPOD_CHAT_BOSS_S3_ENDPOINT || '').trim();
  const accessKeyId = String(process.env.RUNPOD_S3_ACCESS_KEY_ID || '').trim();
  const secretAccessKey = String(process.env.RUNPOD_S3_SECRET_ACCESS_KEY || '').trim();
  if (!volumeId || !region || !endpoint || !accessKeyId || !secretAccessKey) {
    throw new Error('ChatBoss volume or RunPod S3 settings are incomplete.');
  }

  const output = path.resolve(args.output || path.join('ml', 'chatboss', 'evidence', version));
  await fsp.mkdir(output, { recursive: true });
  const client = new S3Client({
    region,
    endpoint,
    forcePathStyle: true,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    credentials: { accessKeyId, secretAccessKey },
  });
  const objects = [
    [`chatboss/candidates/${version}/training_manifest.json`, 'training_manifest.json'],
    [`chatboss/evals/${version}/base.json`, 'base.json'],
    [`chatboss/evals/${version}/candidate.json`, 'candidate.json'],
    [`chatboss/evals/${version}/decision.json`, 'decision.json'],
    [`chatboss/runs/${version}/status.json`, 'status.json'],
  ];
  const files = [];
  async function downloadJson(key, filename, required = true) {
    let response;
    try {
      response = await client.send(new GetObjectCommand({ Bucket: volumeId, Key: key }));
    } catch (error) {
      if (!required && ['NoSuchKey', 'NotFound'].includes(error?.name)) return false;
      throw error;
    }
    const bytes = Buffer.from(await response.Body.transformToByteArray());
    JSON.parse(bytes.toString('utf8'));
    await fsp.writeFile(path.join(output, filename), bytes);
    files.push({
      filename,
      bytes: bytes.length,
      sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
      sourceKey: key,
    });
    return true;
  }
  for (const [key, filename] of objects) {
    await downloadJson(key, filename);
  }
  await downloadJson(
    `chatboss/runs/${version}/token-boundary-preflight.json`,
    'token_boundary_preflight.json',
    false,
  );
  let checkpointObjects = [];
  try {
    const checkpoints = await client.send(new ListObjectsV2Command({
      Bucket: volumeId,
      Prefix: `chatboss/candidates/${version}/checkpoint-`,
    }));
    checkpointObjects = checkpoints.Contents || [];
  } catch {
    // RunPod's S3-compatible volume API can reject directory-style listing in
    // some data centres. Core evidence remains downloadable by exact key.
  }
  const trainerStates = checkpointObjects
    .map((item) => String(item.Key || ''))
    .filter((key) => /\/checkpoint-\d+\/trainer_state\.json$/.test(key))
    .sort((left, right) => {
      const leftStep = Number(left.match(/checkpoint-(\d+)/)?.[1] || 0);
      const rightStep = Number(right.match(/checkpoint-(\d+)/)?.[1] || 0);
      return rightStep - leftStep;
    });
  if (trainerStates[0]) {
    await downloadJson(trainerStates[0], 'trainer_state.json', false);
  }
  const manifest = {
    schema: 'polymath-chatboss-evidence-export-v1',
    version,
    exportedAt: new Date().toISOString(),
    sourceVolumeId: volumeId,
    files,
  };
  await fsp.writeFile(path.join(output, 'evidence_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ version, output, files: files.length }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`ChatBoss evidence download failed: ${error.message}\n`);
  process.exitCode = 1;
});
