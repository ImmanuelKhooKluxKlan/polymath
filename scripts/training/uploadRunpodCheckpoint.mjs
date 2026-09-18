#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const serverPackage = fileURLToPath(new URL('../../server/package.json', import.meta.url));
const requireFromServer = createRequire(serverPackage);
const dotenv = requireFromServer('dotenv');
const {
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  GetObjectCommand,
  ListPartsCommand,
  PutObjectCommand,
  S3Client,
  UploadPartCommand,
} = requireFromServer('@aws-sdk/client-s3');

const REQUIRED_FILES = ['model.safetensors', 'config.json'];
const OPTIONAL_FILES = ['training-metadata.json'];
const PART_BYTES = 64 * 1024 * 1024;

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

function validateCheckpoint(value) {
  const checkpoint = String(value || '').trim();
  if (!/^phase\d+-v\d{3,}$/.test(checkpoint)) {
    throw new Error('--checkpoint must look like phase46-v017');
  }
  return checkpoint;
}

function createClient(env) {
  const client = new S3Client({
    region: env.RUNPOD_S3_REGION,
    endpoint: env.RUNPOD_S3_ENDPOINT,
    forcePathStyle: true,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    maxAttempts: 10,
    credentials: {
      accessKeyId: env.RUNPOD_S3_ACCESS_KEY_ID,
      secretAccessKey: env.RUNPOD_S3_SECRET_ACCESS_KEY,
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
      name: 'normalizeRunpodCheckpointDates',
    },
  );
  return client;
}

function isMissing(error) {
  return Number(error?.$metadata?.httpStatusCode || 0) === 404
    || ['NoSuchKey', 'NotFound'].includes(error?.name);
}

async function sha256File(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function inspectRemote(client, bucket, key) {
  try {
    const response = await client.send(new GetObjectCommand({
      Bucket: bucket,
      Key: key,
      Range: 'bytes=0-0',
    }));
    response.Body?.destroy?.();
    return {
      bytes: response.ContentRange
        ? Number(response.ContentRange.split('/').at(-1))
        : Number(response.ContentLength || 0),
      sha256: String(response.Metadata?.sha256 || ''),
    };
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
}

async function sha256RemoteObject(client, bucket, key) {
  const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  if (!response.Body) throw new Error(`RunPod returned no body for ${key}`);
  const hash = createHash('sha256');
  for await (const chunk of response.Body) hash.update(chunk);
  return hash.digest('hex');
}

function matchingUploadState(state, { bucket, key, bytes, sha256 }) {
  return Boolean(
    state
    && state.bucket === bucket
    && state.key === key
    && state.bytes === bytes
    && state.sha256 === sha256
    && state.uploadId,
  );
}

async function writeState(filename, state) {
  const temporary = `${filename}.tmp`;
  await writeFile(temporary, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
  await rm(filename, { force: true });
  await import('node:fs/promises').then(({ rename }) => rename(temporary, filename));
}

async function readState(filename) {
  try {
    return JSON.parse(await readFile(filename, 'utf8'));
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

async function listParts(client, bucket, key, uploadId) {
  const parts = new Map();
  let marker;
  do {
    const response = await client.send(new ListPartsCommand({
      Bucket: bucket,
      Key: key,
      UploadId: uploadId,
      PartNumberMarker: marker,
    }));
    for (const part of response.Parts || []) {
      parts.set(Number(part.PartNumber), {
        ETag: part.ETag,
        PartNumber: Number(part.PartNumber),
        Size: Number(part.Size),
      });
    }
    marker = response.IsTruncated ? response.NextPartNumberMarker : undefined;
  } while (marker);
  return parts;
}

async function uploadMultipart({
  client,
  bucket,
  key,
  filename,
  bytes,
  sha256,
  stateFilename,
}) {
  let state = await readState(stateFilename);
  if (state && (
    state.bucket !== bucket
    || state.key !== key
    || state.bytes !== bytes
    || state.sha256 !== sha256
  )) {
    throw new Error(`Upload state belongs to a different artifact: ${stateFilename}`);
  }
  if (!state) {
    const created = await client.send(new CreateMultipartUploadCommand({
      Bucket: bucket,
      Key: key,
      ContentType: 'application/octet-stream',
      Metadata: { sha256 },
    }));
    state = {
      schema: 'polymath-runpod-checkpoint-upload-v1',
      bucket,
      key,
      bytes,
      sha256,
      uploadId: created.UploadId,
      createdAt: new Date().toISOString(),
    };
    await writeState(stateFilename, state);
  }
  if (!state.uploadId) throw new Error('RunPod did not return a multipart upload ID');

  const uploaded = await listParts(client, bucket, key, state.uploadId);
  const partCount = Math.ceil(bytes / PART_BYTES);
  const completedParts = [];
  let lastReported = -1;
  for (let index = 0; index < partCount; index += 1) {
    const partNumber = index + 1;
    const start = index * PART_BYTES;
    const length = Math.min(PART_BYTES, bytes - start);
    const existing = uploaded.get(partNumber);
    if (existing?.ETag && existing.Size === length) {
      completedParts.push({ PartNumber: partNumber, ETag: existing.ETag });
    } else {
      let response;
      for (let attempt = 1; attempt <= 5; attempt += 1) {
        const body = createReadStream(filename, { start, end: start + length - 1 });
        try {
          response = await client.send(new UploadPartCommand({
            Bucket: bucket,
            Key: key,
            UploadId: state.uploadId,
            PartNumber: partNumber,
            Body: body,
            ContentLength: length,
          }));
          break;
        } catch (error) {
          body.destroy();
          if (attempt === 5) throw error;
          await new Promise((resolve) => setTimeout(resolve, 1_000 * 2 ** (attempt - 1)));
        }
      }
      completedParts.push({ PartNumber: partNumber, ETag: response.ETag });
    }
    const percent = Math.floor((partNumber / partCount) * 100);
    if (percent >= lastReported + 5 || partNumber === partCount) {
      process.stdout.write(`${key}: ${percent}% (${partNumber}/${partCount})\n`);
      lastReported = percent;
    }
  }

  await client.send(new CompleteMultipartUploadCommand({
    Bucket: bucket,
    Key: key,
    UploadId: state.uploadId,
    MultipartUpload: { Parts: completedParts },
  }));
}

async function uploadFile({
  client,
  bucket,
  key,
  filename,
  stateFilename,
  verifyExistingByDownload = false,
}) {
  const bytes = (await stat(filename)).size;
  const sha256 = await sha256File(filename);
  const existing = await inspectRemote(client, bucket, key);
  if (existing) {
    if (existing.bytes !== bytes) {
      throw new Error(`Refusing to overwrite existing object ${key}; size differs`);
    }
    if (existing.sha256 === sha256) {
      await rm(stateFilename, { force: true });
      process.stdout.write(`Verified existing ${key}\n`);
      return { key, bytes, sha256, reused: true, verification: 'size-and-sha256-metadata' };
    }
    if (bytes <= 16 * 1024 * 1024) {
      const remoteSha256 = await sha256RemoteObject(client, bucket, key);
      if (remoteSha256 === sha256) {
        await rm(stateFilename, { force: true });
        process.stdout.write(`Verified existing ${key} by content hash\n`);
        return { key, bytes, sha256, reused: true, verification: 'downloaded-content-sha256' };
      }
    }
    if (verifyExistingByDownload) {
      process.stdout.write(`Hashing existing remote content for ${key}\n`);
      const remoteSha256 = await sha256RemoteObject(client, bucket, key);
      if (remoteSha256 === sha256) {
        await rm(stateFilename, { force: true });
        process.stdout.write(`Verified existing ${key} by full content hash\n`);
        return { key, bytes, sha256, reused: true, verification: 'downloaded-content-sha256' };
      }
      throw new Error(`Existing object ${key} failed full SHA-256 verification`);
    }
    const state = await readState(stateFilename);
    if (matchingUploadState(state, { bucket, key, bytes, sha256 })) {
      await rm(stateFilename, { force: true });
      process.stdout.write(`Verified completed multipart ${key} by receipt and exact size\n`);
      return {
        key,
        bytes,
        sha256,
        reused: true,
        verification: 'completed-multipart-receipt-exact-size',
      };
    }
    throw new Error(`Refusing to trust existing object ${key}; no matching SHA-256 or multipart receipt`);
  }

  if (bytes > 500 * 1024 * 1024) {
    await uploadMultipart({ client, bucket, key, filename, bytes, sha256, stateFilename });
  } else {
    await client.send(new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: createReadStream(filename),
      ContentLength: bytes,
      ContentType: key.endsWith('.json') ? 'application/json' : 'application/octet-stream',
      Metadata: { sha256 },
    }));
  }

  const verified = await inspectRemote(client, bucket, key);
  if (verified?.bytes !== bytes) {
    throw new Error(`Remote verification failed for ${key}`);
  }
  let verification = 'size-and-sha256-metadata';
  if (verified.sha256 !== sha256) {
    if (bytes <= 16 * 1024 * 1024) {
      const remoteSha256 = await sha256RemoteObject(client, bucket, key);
      if (remoteSha256 !== sha256) throw new Error(`Remote SHA-256 verification failed for ${key}`);
      verification = 'downloaded-content-sha256';
    } else {
      const state = await readState(stateFilename);
      if (!matchingUploadState(state, { bucket, key, bytes, sha256 })) {
        throw new Error(`Remote multipart receipt verification failed for ${key}`);
      }
      verification = 'completed-multipart-receipt-exact-size';
    }
  }
  await rm(stateFilename, { force: true });
  process.stdout.write(`Verified upload ${key} (${verification})\n`);
  return { key, bytes, sha256, reused: false, verification };
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  dotenv.config({ path: path.resolve(args.env || 'server/.env'), quiet: true });
  const checkpoint = validateCheckpoint(args.checkpoint);
  const source = path.resolve(String(args.source || ''));
  if (!args.source || !(await stat(source).catch(() => null))?.isDirectory()) {
    throw new Error('--source must be an existing checkpoint directory');
  }
  const bucket = String(args['volume-id'] || process.env.RUNPOD_NETWORK_VOLUME_ID || '').trim();
  if (!bucket) throw new Error('RunPod volume ID is missing');
  for (const name of [
    'RUNPOD_S3_REGION',
    'RUNPOD_S3_ENDPOINT',
    'RUNPOD_S3_ACCESS_KEY_ID',
    'RUNPOD_S3_SECRET_ACCESS_KEY',
  ]) {
    if (!String(process.env[name] || '').trim()) throw new Error(`${name} is missing`);
  }

  const files = [];
  for (const name of REQUIRED_FILES) {
    const filename = path.join(source, name);
    if (!(await stat(filename).catch(() => null))?.isFile()) {
      throw new Error(`Checkpoint is missing required file ${name}`);
    }
    files.push({ name, filename });
  }
  for (const name of OPTIONAL_FILES) {
    const filename = path.join(source, name);
    if ((await stat(filename).catch(() => null))?.isFile()) files.push({ name, filename });
  }

  const client = createClient(process.env);
  const prefix = `models/muscriptor-tester/${checkpoint}`;
  const uploaded = [];
  for (const file of files) {
    uploaded.push(await uploadFile({
      client,
      bucket,
      key: `${prefix}/${file.name}`,
      filename: file.filename,
      stateFilename: path.join(source, `.runpod-upload-${checkpoint}-${file.name}.json`),
      verifyExistingByDownload: Boolean(args['verify-existing-by-download']),
    }));
  }
  process.stdout.write(`${JSON.stringify({ checkpoint, bucket, prefix, files: uploaded }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Checkpoint upload failed: ${error.message}\n`);
  process.exitCode = 1;
});
