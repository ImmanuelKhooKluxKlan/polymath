#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const serverPackage = fileURLToPath(new URL('../../server/package.json', import.meta.url));
const requireFromServer = createRequire(serverPackage);
const dotenv = requireFromServer('dotenv');
const { GetObjectCommand, ListObjectsV2Command, S3Client } = requireFromServer('@aws-sdk/client-s3');

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function gib(bytes) {
  return Number((bytes / 1024 ** 3).toFixed(3));
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  dotenv.config({ path: path.resolve(args.env || 'server/.env') });
  const volumeId = String(args['volume-id'] || process.env.RUNPOD_NETWORK_VOLUME_ID || '').trim();
  const prefix = String(args.prefix || '').replace(/^\/+/, '');
  const depth = Math.max(1, Math.min(8, Number(args.depth || 3)));
  if (!volumeId) throw new Error('RunPod volume ID is missing');
  const client = new S3Client({
    region: process.env.RUNPOD_S3_REGION,
    endpoint: process.env.RUNPOD_S3_ENDPOINT,
    forcePathStyle: true,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    maxAttempts: 10,
    credentials: {
      accessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
      secretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    },
  });
  const checkpoints = String(args.checkpoints || '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
  if (checkpoints.length) {
    const includeMetadata = String(args['include-metadata'] || '').toLowerCase() === 'true';
    const rows = [];
    for (const version of checkpoints) {
      if (version !== 'original' && !/^phase\d+-v\d{3,}$/.test(version)) {
        throw new Error(`Invalid checkpoint version: ${version}`);
      }
      const root = version === 'original'
        ? 'models/original'
        : `models/muscriptor-tester/${version}`;
      const files = [];
      for (const name of ['model.safetensors', 'config.json', 'training-metadata.json']) {
        const key = `${root}/${name}`;
        try {
          const response = await client.send(new GetObjectCommand({
            Bucket: volumeId,
            Key: key,
            Range: 'bytes=0-0',
          }));
          response.Body?.destroy?.();
          const size = response.ContentRange
            ? Number(response.ContentRange.split('/').at(-1))
            : Number(response.ContentLength || 0);
          files.push({ key, bytes: size, gib: gib(size), exists: true });
        } catch (error) {
          const status = Number(error?.$metadata?.httpStatusCode || 0);
          if (status !== 404 && !['NoSuchKey', 'NotFound'].includes(error?.name)) throw error;
          files.push({ key, bytes: 0, gib: 0, exists: false });
        }
      }
      const totalBytes = files.reduce((sum, file) => sum + file.bytes, 0);
      let metadata = null;
      if (includeMetadata) {
        const metadataFile = files.find((file) => file.exists && file.key.endsWith('/training-metadata.json'));
        if (metadataFile) {
          const response = await client.send(new GetObjectCommand({
            Bucket: volumeId,
            Key: metadataFile.key,
          }));
          const body = await response.Body?.transformToString?.();
          metadata = body ? JSON.parse(body) : null;
        }
      }
      rows.push({ version, bytes: totalBytes, gib: gib(totalBytes), files, metadata });
    }
    process.stdout.write(`${JSON.stringify({ volumeId, checkpoints: rows }, null, 2)}\n`);
    return;
  }
  const groups = new Map();
  let continuationToken;
  let objects = 0;
  let bytes = 0;
  do {
    const page = await client.send(new ListObjectsV2Command({
      Bucket: volumeId,
      Prefix: prefix || undefined,
      ContinuationToken: continuationToken,
      MaxKeys: 1000,
    }));
    for (const item of page.Contents || []) {
      const key = String(item.Key || '');
      const size = Number(item.Size || 0);
      const group = key.split('/').slice(0, depth).join('/');
      const current = groups.get(group) || { prefix: group, objects: 0, bytes: 0 };
      current.objects += 1;
      current.bytes += size;
      groups.set(group, current);
      objects += 1;
      bytes += size;
    }
    continuationToken = page.IsTruncated ? page.NextContinuationToken : undefined;
  } while (continuationToken);
  const rows = [...groups.values()]
    .sort((left, right) => right.bytes - left.bytes)
    .map((row) => ({ ...row, gib: gib(row.bytes) }));
  process.stdout.write(`${JSON.stringify({
    volumeId,
    queriedPrefix: prefix,
    objects,
    bytes,
    gib: gib(bytes),
    groups: rows,
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod volume audit failed: ${error.message}\n`);
  process.exitCode = 1;
});
