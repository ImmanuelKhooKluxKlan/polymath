#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';

const repoRoot = path.resolve(import.meta.dirname, '..', '..');
const require = createRequire(path.join(repoRoot, 'server', 'runpodServerless.js'));
const { HeadObjectCommand, S3Client } = require('@aws-sdk/client-s3');
const { parseReplicas } = require(path.join(repoRoot, 'server', 'runpodServerless.js'));

process.loadEnvFile(path.join(repoRoot, 'server', '.env'));

function clean(value) {
  return String(value || '').trim();
}

function targets() {
  const output = [{
    volumeId: clean(process.env.RUNPOD_NETWORK_VOLUME_ID),
    region: clean(process.env.RUNPOD_S3_REGION),
    s3Endpoint: clean(process.env.RUNPOD_S3_ENDPOINT).replace(/\/+$/, ''),
  }];
  for (const replica of parseReplicas(process.env.RUNPOD_S3_REPLICAS)) {
    const item = {
      volumeId: clean(replica?.volumeId),
      region: clean(replica?.region),
      s3Endpoint: clean(replica?.s3Endpoint).replace(/\/+$/, ''),
    };
    if (item.volumeId && !output.some((entry) => entry.volumeId === item.volumeId)) {
      output.push(item);
    }
  }
  return output;
}

async function head(client, volumeId, key) {
  try {
    const result = await client.send(new HeadObjectCommand({ Bucket: volumeId, Key: key }));
    return { exists: true, bytes: Number(result.ContentLength || 0) };
  } catch (error) {
    const status = Number(error?.$metadata?.httpStatusCode || 0);
    if (status === 404 || error?.name === 'NotFound' || error?.Code === 'NoSuchKey') {
      return { exists: false, bytes: 0 };
    }
    return { exists: false, bytes: 0, error: String(error?.message || error) };
  }
}

async function main() {
  const accessKeyId = clean(process.env.RUNPOD_S3_ACCESS_KEY_ID);
  const secretAccessKey = clean(process.env.RUNPOD_S3_SECRET_ACCESS_KEY);
  if (!accessKeyId || !secretAccessKey) throw new Error('RunPod S3 credentials are missing.');
  const reports = [];
  for (const target of targets()) {
    const client = new S3Client({
      region: target.region,
      endpoint: target.s3Endpoint,
      forcePathStyle: true,
      requestChecksumCalculation: 'WHEN_REQUIRED',
      responseChecksumValidation: 'WHEN_REQUIRED',
      credentials: { accessKeyId, secretAccessKey },
      maxAttempts: 5,
    });
    const requestedVersions = ['phase46-v007', 'phase46-v017'];
    const originalWeights = await head(client, target.volumeId, 'models/original/model.safetensors');
    const originalConfig = await head(client, target.volumeId, 'models/original/config.json');
    const versions = [];
    for (const version of requestedVersions) {
      const weights = await head(client, target.volumeId, `models/muscriptor-tester/${version}/model.safetensors`);
      const config = await head(client, target.volumeId, `models/muscriptor-tester/${version}/config.json`);
      versions.push({
        version,
        complete: weights.exists && config.exists,
        files: { 'model.safetensors': weights, 'config.json': config },
      });
    }
    reports.push({
      volumeId: target.volumeId,
      region: target.region,
      s3Endpoint: target.s3Endpoint,
      original: {
        complete: originalWeights.exists && originalConfig.exists,
        files: { 'model.safetensors': originalWeights, 'config.json': originalConfig },
      },
      tester: versions,
    });
  }
  process.stdout.write(`${JSON.stringify({ volumes: reports }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`RunPod checkpoint audit failed: ${error.message}\n`);
  process.exitCode = 1;
});
