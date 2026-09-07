#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const requireFromServer = createRequire(fileURLToPath(new URL('../../server/package.json', import.meta.url)));
const dotenv = requireFromServer('dotenv');
const { HeadObjectCommand, PutObjectCommand, S3Client } = requireFromServer('@aws-sdk/client-s3');
const ALLOWED_FILES = new Set([
  'requirements.txt',
  'train_lora.py',
  'evaluate_adapter.py',
  'compare_evaluations.py',
  'run_candidate.py',
  'verify_chat_boundaries.py',
]);

function argumentsFrom(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    result[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return result;
}

function readTargets() {
  const primary = {
    volumeId: String(process.env.RUNPOD_CHAT_BOSS_NETWORK_VOLUME_ID || '').trim(),
    region: String(process.env.RUNPOD_CHAT_BOSS_S3_REGION || '').trim(),
    s3Endpoint: String(process.env.RUNPOD_CHAT_BOSS_S3_ENDPOINT || '').trim(),
  };
  const replicaText = String(process.env.RUNPOD_CHAT_BOSS_S3_REPLICAS || '').trim();
  const replicas = replicaText ? JSON.parse(replicaText) : [];
  if (!Array.isArray(replicas)) throw new Error('RUNPOD_CHAT_BOSS_S3_REPLICAS must be a JSON array.');
  const targets = [primary, ...replicas].map((target) => ({
    volumeId: String(target.volumeId || '').trim(),
    region: String(target.region || '').trim(),
    s3Endpoint: String(target.s3Endpoint || '').trim(),
  }));
  if (targets.some((target) => !target.volumeId || !target.region || !target.s3Endpoint)) {
    throw new Error('Every ChatBoss training target needs volumeId, region, and s3Endpoint.');
  }
  return [...new Map(targets.map((target) => [target.volumeId, target])).values()];
}

async function digest(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function verifiedLength(client, bucket, key) {
  try {
    const remote = await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
    return Number(remote.ContentLength);
  } catch (error) {
    const response = error?.$response;
    const rawLength = Number(response?.headers?.['content-length']);
    if (Number(response?.statusCode) === 200 && Number.isFinite(rawLength)) return rawLength;
    throw error;
  }
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  if (!args.env || !args.root) {
    throw new Error('Usage: node upload_training_bundle.mjs --env server.env --chatboss-env chatboss.env --root ml/chatboss --version v003 --holdout data/behavior_holdout_v002.jsonl');
  }
  dotenv.config({ path: path.resolve(args.env), quiet: true });
  if (args['chatboss-env']) dotenv.config({ path: path.resolve(args['chatboss-env']), quiet: true, override: true });
  const accessKeyId = String(process.env.RUNPOD_S3_ACCESS_KEY_ID || '').trim();
  const secretAccessKey = String(process.env.RUNPOD_S3_SECRET_ACCESS_KEY || '').trim();
  if (!accessKeyId || !secretAccessKey) throw new Error('RunPod S3 credentials are incomplete.');
  const root = path.resolve(args.root);
  const version = String(args.version || 'v002').replace(/[^a-zA-Z0-9._-]/g, '');
  const holdout = path.resolve(root, args.holdout || path.join('data', 'behavior_holdout_v001.jsonl'));
  const sourceFiles = [
    ...[...ALLOWED_FILES].map((name) => path.join(root, name)),
    holdout,
  ];
  for (const filename of sourceFiles) {
    if (!await fsp.stat(filename).then((entry) => entry.isFile()).catch(() => false)) {
      throw new Error(`Required training bundle file is missing: ${filename}`);
    }
  }
  const uploads = [];
  for (const target of readTargets()) {
    const client = new S3Client({
      region: target.region,
      endpoint: target.s3Endpoint,
      forcePathStyle: true,
      requestChecksumCalculation: 'WHEN_REQUIRED',
      responseChecksumValidation: 'WHEN_REQUIRED',
      credentials: { accessKeyId, secretAccessKey },
    });
    const files = [];
    for (const filename of sourceFiles) {
      const stat = await fsp.stat(filename);
      const key = `chatboss/training/${version}/${path.basename(filename)}`;
      await client.send(new PutObjectCommand({
        Bucket: target.volumeId,
        Key: key,
        Body: fs.createReadStream(filename),
        ContentLength: stat.size,
        ContentType: filename.endsWith('.jsonl') ? 'application/x-ndjson' : 'text/plain; charset=utf-8',
      }), { abortSignal: AbortSignal.timeout(45_000) });
      const remoteLength = await verifiedLength(client, target.volumeId, key);
      if (remoteLength !== stat.size) throw new Error(`Remote size check failed for ${target.volumeId}/${key}`);
      files.push({ key, bytes: stat.size, sha256: await digest(filename) });
    }
    uploads.push({ volumeId: target.volumeId, region: target.region, files });
  }
  const manifest = {
    schema: 'polymath-chatboss-training-bundle-v1',
    version,
    uploads,
  };
  await fsp.writeFile(path.join(root, `training-bundle-${version}-manifest.json`), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    version,
    targetCount: uploads.length,
    fileCount: uploads.reduce((sum, target) => sum + target.files.length, 0),
    totalBytes: uploads.reduce((sum, target) => sum + target.files.reduce((n, file) => n + file.bytes, 0), 0),
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`ChatBoss training bundle upload failed: ${error.message}\n`);
  process.exitCode = 1;
});
