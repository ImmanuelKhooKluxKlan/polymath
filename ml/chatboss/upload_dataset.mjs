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

function argumentsFrom(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    result[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return result;
}

async function sha256(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

async function main() {
  const args = argumentsFrom(process.argv.slice(2));
  if (!args.env || !args.root) {
    throw new Error('Usage: node upload_dataset.mjs --env /private/server.env --chatboss-env /private/chatboss.env --root prepared/v001 [--version v001]');
  }
  dotenv.config({ path: path.resolve(args.env), quiet: true });
  if (args['chatboss-env']) {
    dotenv.config({ path: path.resolve(args['chatboss-env']), quiet: true, override: true });
  }
  const root = path.resolve(args.root);
  const volumeId = String(process.env.RUNPOD_CHAT_BOSS_NETWORK_VOLUME_ID || '').trim();
  const region = String(process.env.RUNPOD_CHAT_BOSS_S3_REGION || process.env.RUNPOD_S3_REGION || '').trim();
  const endpoint = String(process.env.RUNPOD_CHAT_BOSS_S3_ENDPOINT || process.env.RUNPOD_S3_ENDPOINT || '').trim();
  const accessKeyId = String(process.env.RUNPOD_S3_ACCESS_KEY_ID || '').trim();
  const secretAccessKey = String(process.env.RUNPOD_S3_SECRET_ACCESS_KEY || '').trim();
  if (!volumeId || !region || !endpoint || !accessKeyId || !secretAccessKey) {
    throw new Error('ChatBoss volume or RunPod S3 settings are incomplete.');
  }
  const version = String(args.version || 'v001').replace(/[^a-zA-Z0-9._-]/g, '');
  let configuredReplicas = [];
  const replicaText = String(process.env.RUNPOD_CHAT_BOSS_S3_REPLICAS || '').trim();
  if (replicaText) {
    const parsed = JSON.parse(replicaText);
    if (!Array.isArray(parsed)) throw new Error('RUNPOD_CHAT_BOSS_S3_REPLICAS must be a JSON array.');
    configuredReplicas = parsed;
  }
  const targets = [
    { volumeId, region, s3Endpoint: endpoint },
    ...configuredReplicas.map((replica) => ({
      volumeId: String(replica?.volumeId || '').trim(),
      region: String(replica?.region || '').trim(),
      s3Endpoint: String(replica?.s3Endpoint || '').trim(),
    })),
  ];
  if (targets.some((target) => !target.volumeId || !target.region || !target.s3Endpoint)) {
    throw new Error('Every ChatBoss S3 replica needs volumeId, region, and s3Endpoint.');
  }
  const uniqueTargets = [...new Map(targets.map((target) => [target.volumeId, target])).values()];
  const filenames = (await fsp.readdir(root, { withFileTypes: true }))
    .filter((entry) => (
      entry.isFile()
      && entry.name !== 'volume-upload-manifest.json'
      && ['.json', '.jsonl'].includes(path.extname(entry.name).toLowerCase())
    ))
    .map((entry) => path.join(root, entry.name));
  if (!filenames.length) throw new Error('No JSON or JSONL dataset files were found.');
  const uploads = [];
  for (const target of uniqueTargets) {
    const client = new S3Client({
      region: target.region,
      endpoint: target.s3Endpoint,
      forcePathStyle: true,
      requestChecksumCalculation: 'WHEN_REQUIRED',
      responseChecksumValidation: 'WHEN_REQUIRED',
      credentials: { accessKeyId, secretAccessKey },
    });
    const files = [];
    for (const filename of filenames) {
      const stat = await fsp.stat(filename);
      const key = `chatboss/datasets/${version}/${path.basename(filename)}`;
      await client.send(new PutObjectCommand({
        Bucket: target.volumeId,
        Key: key,
        Body: fs.createReadStream(filename),
        ContentLength: stat.size,
        ContentType: path.extname(filename).toLowerCase() === '.jsonl' ? 'application/x-ndjson' : 'application/json',
      }), { abortSignal: AbortSignal.timeout(45_000) });
      let remoteLength;
      try {
        const remote = await client.send(
          new HeadObjectCommand({ Bucket: target.volumeId, Key: key }),
          { abortSignal: AbortSignal.timeout(30_000) },
        );
        remoteLength = Number(remote.ContentLength);
      } catch (error) {
        // RunPod's S3 layer has returned an otherwise-successful HEAD response with
        // a non-RFC `Last-Modified: ... UTC` value. Smithy rejects that date before
        // returning HeadObject, but leaves the successful raw response available.
        const response = error?.$response;
        const rawLength = Number(response?.headers?.['content-length']);
        if (Number(response?.statusCode) === 200 && Number.isFinite(rawLength)) {
          remoteLength = rawLength;
        } else {
          throw error;
        }
      }
      if (remoteLength !== stat.size) throw new Error(`Remote size check failed for ${target.volumeId}/${key}`);
      files.push({ key, bytes: stat.size, sha256: await sha256(filename) });
    }
    uploads.push({ volumeId: target.volumeId, region: target.region, files });
  }
  const manifest = { schema: 'polymath-chatboss-volume-upload-v2', version, uploads };
  await fsp.writeFile(path.join(root, 'volume-upload-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    version,
    targetCount: uploads.length,
    fileCount: uploads.reduce((sum, upload) => sum + upload.files.length, 0),
    totalBytes: uploads.reduce((sum, upload) => sum + upload.files.reduce((fileSum, file) => fileSum + file.bytes, 0), 0),
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`ChatBoss dataset upload failed: ${error.message}\n`);
  process.exitCode = 1;
});
