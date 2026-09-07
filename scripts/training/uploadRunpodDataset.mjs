#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const serverPackage = fileURLToPath(new URL('../../server/package.json', import.meta.url));
const requireFromServer = createRequire(serverPackage);
const dotenv = requireFromServer('dotenv');
const {
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
} = requireFromServer('@aws-sdk/client-s3');

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const value = argv[index + 1];
    values[token.slice(2)] = !value || value.startsWith('--') ? true : value;
    if (value && !value.startsWith('--')) index += 1;
  }
  return values;
}

function parseReplicas(value) {
  try {
    const parsed = JSON.parse(String(value || '[]'));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function targetForVolume(volumeId) {
  const targets = [
    {
      volumeId: process.env.RUNPOD_NETWORK_VOLUME_ID,
      region: process.env.RUNPOD_S3_REGION,
      s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    },
    ...parseReplicas(process.env.RUNPOD_S3_REPLICAS),
  ];
  return targets.find((target) => String(target.volumeId) === String(volumeId));
}

function createClient(target) {
  const client = new S3Client({
    region: target.region,
    endpoint: target.s3Endpoint,
    forcePathStyle: true,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    maxAttempts: 10,
    credentials: {
      accessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID,
      secretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY,
    },
  });
  client.middlewareStack.addRelativeTo(
    (next) => async (request) => {
      const output = await next(request);
      const headers = output.response?.headers || {};
      for (const name of ['date', 'last-modified']) {
        if (typeof headers[name] === 'string') {
          headers[name] = headers[name].replace(/ UTC$/, ' GMT');
        }
      }
      return output;
    },
    {
      relation: 'after',
      toMiddleware: 'deserializerMiddleware',
      name: 'normalizeRunpodDatasetDates',
    },
  );
  return client;
}

async function collectFiles(root) {
  const result = [];
  async function visit(directory) {
    for (const entry of await fsp.readdir(directory, { withFileTypes: true })) {
      const filename = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(filename);
      else if (entry.isFile()) result.push(filename);
    }
  }
  await visit(root);
  return result.sort();
}

async function sha256(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.root || !args['volume-id'] || !args.prefix) {
    throw new Error('Usage: node uploadRunpodDataset.mjs --root DIR --volume-id ID --prefix training/name [--env server/.env]');
  }
  dotenv.config({ path: path.resolve(args.env || 'server/.env'), quiet: true });
  const root = path.resolve(args.root);
  const stat = await fsp.stat(root).catch(() => null);
  if (!stat?.isDirectory()) throw new Error(`Dataset root does not exist: ${root}`);
  const target = targetForVolume(args['volume-id']);
  if (!target?.region || !target?.s3Endpoint) {
    throw new Error(`No configured RunPod S3 target matches volume ${args['volume-id']}`);
  }
  const client = createClient(target);
  const prefix = String(args.prefix).replace(/^\/+|\/+$/g, '');
  const files = await collectFiles(root);
  const manifest = [];
  for (const [index, filename] of files.entries()) {
    const relative = path.relative(root, filename).split(path.sep).join('/');
    const key = `${prefix}/${relative}`;
    const size = (await fsp.stat(filename)).size;
    process.stdout.write(`[${index + 1}/${files.length}] ${relative}\n`);
    await client.send(new PutObjectCommand({
      Bucket: target.volumeId,
      Key: key,
      Body: fs.createReadStream(filename),
      ContentType: path.extname(filename).toLowerCase() === '.wav'
        ? 'audio/wav'
        : 'application/octet-stream',
    }));
    const verification = await client.send(new GetObjectCommand({
      Bucket: target.volumeId,
      Key: key,
      Range: 'bytes=0-0',
    }));
    verification.Body?.destroy?.();
    const remoteSize = verification.ContentRange
      ? Number(verification.ContentRange.split('/').at(-1))
      : Number.NaN;
    if (Number.isFinite(remoteSize) && remoteSize !== size) {
      throw new Error(`${key}: remote size ${remoteSize} does not equal local size ${size}`);
    }
    manifest.push({
      key,
      bytes: size,
      sha256: await sha256(filename),
      remoteSizeVerified: remoteSize === size,
    });
  }
  const result = {
    schema: 'polymath-runpod-dataset-upload-v1',
    generatedAt: new Date().toISOString(),
    localRoot: root,
    volumeId: target.volumeId,
    region: target.region,
    prefix,
    files: manifest,
    totalBytes: manifest.reduce((sum, file) => sum + file.bytes, 0),
  };
  const manifestPath = path.resolve(args.manifest || path.join(root, 'runpod-upload-manifest.json'));
  await fsp.writeFile(manifestPath, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    volumeId: result.volumeId,
    region: result.region,
    files: result.files.length,
    totalBytes: result.totalBytes,
    manifest: manifestPath,
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod dataset upload failed: ${error.message}\n`);
  process.exitCode = 1;
});
