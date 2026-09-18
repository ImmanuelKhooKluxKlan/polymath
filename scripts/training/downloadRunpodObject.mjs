#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pipeline } from 'node:stream/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const serverPackage = fileURLToPath(new URL('../../server/package.json', import.meta.url));
const requireFromServer = createRequire(serverPackage);
const dotenv = requireFromServer('dotenv');
const { GetObjectCommand, S3Client } = requireFromServer('@aws-sdk/client-s3');

function argumentsFrom(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    values[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function replicasFrom(value) {
  try {
    const parsed = JSON.parse(String(value || '[]'));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function targetFor(volumeId) {
  return [
    {
      volumeId: process.env.RUNPOD_NETWORK_VOLUME_ID,
      region: process.env.RUNPOD_S3_REGION,
      s3Endpoint: process.env.RUNPOD_S3_ENDPOINT,
    },
    ...replicasFrom(process.env.RUNPOD_S3_REPLICAS),
  ].find((target) => String(target.volumeId) === String(volumeId));
}

function clientFor(target) {
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
        if (typeof headers[name] === 'string') headers[name] = headers[name].replace(/ UTC$/, ' GMT');
      }
      return output;
    },
    {
      relation: 'after',
      toMiddleware: 'deserializerMiddleware',
      name: 'normalizeRunpodDownloadDates',
    },
  );
  return client;
}

async function sha256(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  if (!args.key || !args.output || !args['volume-id']) {
    throw new Error('Usage: --key jobs/path/file.wav --output local.wav --volume-id ID [--env server/.env]');
  }
  const key = String(args.key).replace(/^\/+/, '');
  if (!key || key.split('/').includes('..')) throw new Error('--key must be a safe relative object path');
  dotenv.config({ path: path.resolve(args.env || 'server/.env'), quiet: true });
  const target = targetFor(args['volume-id']);
  if (!target?.region || !target?.s3Endpoint) {
    throw new Error(`No configured RunPod S3 target matches volume ${args['volume-id']}`);
  }
  const output = path.resolve(args.output);
  const temporary = `${output}.${process.pid}.partial`;
  await fsp.mkdir(path.dirname(output), { recursive: true });
  try {
    const response = await clientFor(target).send(new GetObjectCommand({
      Bucket: target.volumeId,
      Key: key,
    }));
    if (!response.Body) throw new Error('RunPod returned an object without a body');
    await pipeline(response.Body, fs.createWriteStream(temporary, { flags: 'wx' }));
    const bytes = (await fsp.stat(temporary)).size;
    if (Number.isFinite(Number(response.ContentLength)) && bytes !== Number(response.ContentLength)) {
      throw new Error(`downloaded ${bytes} bytes but RunPod reported ${response.ContentLength}`);
    }
    await fsp.rename(temporary, output);
    process.stdout.write(`${JSON.stringify({
      volumeId: target.volumeId,
      key,
      output,
      bytes,
      sha256: await sha256(output),
    }, null, 2)}\n`);
  } finally {
    await fsp.rm(temporary, { force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`RunPod object download failed: ${error.message}\n`);
  process.exitCode = 1;
});
