#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const serverPackage = fileURLToPath(new URL('../../server/package.json', import.meta.url));
const requireFromServer = createRequire(serverPackage);
const dotenv = requireFromServer('dotenv');
const { DeleteObjectCommand, GetObjectCommand, S3Client } = requireFromServer('@aws-sdk/client-s3');

const CHECKPOINT_FILES = ['model.safetensors', 'config.json', 'training-metadata.json'];
const PROTECTED_VERSIONS = new Set(['original', 'phase8-v005', 'phase46-v007']);
const DELETE_CONFIRMATION = 'DELETE_VERIFIED_REMOTE_CHECKPOINT';

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function validateVersion(value) {
  const version = String(value || '').trim();
  if (!/^phase\d+-v\d{3,}$/.test(version)) {
    throw new Error('Checkpoint must look like phase46-v008; original is never archivable here');
  }
  if (PROTECTED_VERSIONS.has(version)) {
    throw new Error(`Checkpoint ${version} is protected and cannot be removed by this tool`);
  }
  return version;
}

function createClient(env) {
  return new S3Client({
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
}

function notFound(error) {
  return Number(error?.$metadata?.httpStatusCode || 0) === 404
    || ['NoSuchKey', 'NotFound'].includes(error?.name);
}

function formatGiB(bytes) {
  return (bytes / 1024 ** 3).toFixed(3);
}

async function sha256File(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function inspectRemoteObject(client, bucket, key) {
  try {
    const response = await client.send(new GetObjectCommand({
      Bucket: bucket,
      Key: key,
      Range: 'bytes=0-0',
    }));
    response.Body?.destroy?.();
    const bytes = response.ContentRange
      ? Number(response.ContentRange.split('/').at(-1))
      : Number(response.ContentLength || 0);
    return { exists: true, bytes, etag: response.ETag || null };
  } catch (error) {
    if (notFound(error)) return { exists: false, bytes: 0, etag: null };
    throw error;
  }
}

async function downloadVerifiedObject(client, bucket, key, destination, expectedBytes) {
  const existing = await stat(destination).catch(() => null);
  if (existing?.isFile() && existing.size === expectedBytes) {
    const sha256 = await sha256File(destination);
    console.log(`Verified existing archive ${path.basename(destination)} (${formatGiB(existing.size)} GiB)`);
    return { bytes: existing.size, sha256, reused: true };
  }
  if (existing) {
    throw new Error(`Archive destination already exists with the wrong size: ${destination}`);
  }

  const temporary = `${destination}.partial`;
  await rm(temporary, { force: true });
  const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  if (!response.Body) throw new Error(`RunPod returned no body for ${key}`);
  const declaredBytes = Number(response.ContentLength || expectedBytes || 0);
  let bytes = 0;
  let nextProgress = 512 * 1024 ** 2;
  const hash = createHash('sha256');
  const meter = new Transform({
    transform(chunk, encoding, callback) {
      bytes += chunk.length;
      hash.update(chunk);
      if (bytes >= nextProgress) {
        console.log(`Archived ${path.basename(destination)}: ${formatGiB(bytes)} / ${formatGiB(expectedBytes)} GiB`);
        nextProgress += 512 * 1024 ** 2;
      }
      callback(null, chunk);
    },
  });
  try {
    await pipeline(response.Body, meter, createWriteStream(temporary, { flags: 'wx' }));
    if (expectedBytes && bytes !== expectedBytes) {
      throw new Error(`Size mismatch for ${key}: expected ${expectedBytes}, received ${bytes}`);
    }
    if (declaredBytes && bytes !== declaredBytes) {
      throw new Error(`Content-Length mismatch for ${key}: expected ${declaredBytes}, received ${bytes}`);
    }
    await rename(temporary, destination);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
  return { bytes, sha256: hash.digest('hex'), reused: false };
}

async function writeManifestAtomic(destinationRoot, manifest) {
  const filename = path.join(destinationRoot, 'archive-manifest.json');
  const temporary = `${filename}.tmp`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  await rm(filename, { force: true });
  await rename(temporary, filename);
  return filename;
}

async function verifyManifestFiles(destinationRoot, files) {
  for (const file of files) {
    const filename = path.join(destinationRoot, file.name);
    const local = await stat(filename);
    if (!local.isFile() || local.size !== file.bytes) {
      throw new Error(`Local verification failed for ${filename}`);
    }
    const sha256 = await sha256File(filename);
    if (sha256 !== file.sha256) throw new Error(`SHA-256 verification failed for ${filename}`);
  }
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  dotenv.config({ path: path.resolve(args.env || 'server/.env') });
  const volumeId = String(args['volume-id'] || process.env.RUNPOD_NETWORK_VOLUME_ID || '').trim();
  if (!volumeId) throw new Error('RunPod volume ID is missing');
  const version = validateVersion(args.checkpoint);
  const destinationRoot = path.resolve(String(args.destination || '').trim());
  if (!args.destination) throw new Error('--destination is required and should point to a dedicated D: archive folder');
  if (process.platform === 'win32' && path.parse(destinationRoot).root.toUpperCase() !== 'D:\\') {
    throw new Error(`Refusing to store a large checkpoint outside D: (${destinationRoot})`);
  }
  const deleteAfterVerify = args.confirm === DELETE_CONFIRMATION;
  if (args.confirm && !deleteAfterVerify) {
    throw new Error(`Invalid confirmation. Exact value is ${DELETE_CONFIRMATION}`);
  }

  const client = createClient(process.env);
  const root = `models/muscriptor-tester/${version}`;
  const remote = [];
  for (const name of CHECKPOINT_FILES) {
    const key = `${root}/${name}`;
    const inspection = await inspectRemoteObject(client, volumeId, key);
    if (inspection.exists) remote.push({ name, key, ...inspection });
  }
  const weights = remote.find((file) => file.name === 'model.safetensors');
  const config = remote.find((file) => file.name === 'config.json');
  if (!weights || !config) throw new Error(`Checkpoint ${version} is incomplete or absent; nothing will be deleted`);

  console.log(`Archiving ${version}: ${formatGiB(remote.reduce((sum, file) => sum + file.bytes, 0))} GiB from exact prefix ${root}`);
  await mkdir(destinationRoot, { recursive: true });
  const archived = [];
  for (const file of remote) {
    const local = await downloadVerifiedObject(
      client,
      volumeId,
      file.key,
      path.join(destinationRoot, file.name),
      file.bytes,
    );
    archived.push({
      name: file.name,
      remoteKey: file.key,
      bytes: local.bytes,
      sha256: local.sha256,
      etag: file.etag,
      reusedLocalFile: local.reused,
    });
  }
  await verifyManifestFiles(destinationRoot, archived);
  const manifest = {
    schema: 'polymath-runpod-checkpoint-archive-v1',
    checkpoint: version,
    volumeId,
    remotePrefix: root,
    archivedAt: new Date().toISOString(),
    verifiedBeforeRemoteDelete: true,
    remoteDeleted: false,
    files: archived,
  };
  const manifestFilename = await writeManifestAtomic(destinationRoot, manifest);
  console.log(`Verified archive manifest: ${manifestFilename}`);

  if (!deleteAfterVerify) {
    console.log(`Remote checkpoint retained. To remove it, rerun with --confirm ${DELETE_CONFIRMATION}`);
    return;
  }

  for (const file of remote) {
    await client.send(new DeleteObjectCommand({ Bucket: volumeId, Key: file.key }));
  }
  for (const file of remote) {
    const inspection = await inspectRemoteObject(client, volumeId, file.key);
    if (inspection.exists) throw new Error(`Remote deletion could not be verified for ${file.key}`);
  }
  manifest.remoteDeleted = true;
  manifest.remoteDeletedAt = new Date().toISOString();
  await writeManifestAtomic(destinationRoot, manifest);
  console.log(`Removed verified remote checkpoint ${version}; recovery copy remains at ${destinationRoot}`);
}

main().catch((error) => {
  process.stderr.write(`Checkpoint archive failed: ${error.message}\n`);
  process.exitCode = 1;
});
