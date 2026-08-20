const fs = require('node:fs');
const {
  DeleteObjectCommand,
  PutObjectCommand,
  S3Client,
} = require('@aws-sdk/client-s3');

const TERMINAL_FAILURES = new Set(['CANCELLED', 'FAILED', 'TIMED_OUT']);

function clean(value) {
  return String(value || '').trim();
}

function createRunpodServerlessClient(configuration = {}, dependencies = {}) {
  const endpointId = clean(configuration.endpointId);
  const apiKey = clean(configuration.apiKey);
  const volumeId = clean(configuration.volumeId);
  const region = clean(configuration.region);
  const s3Endpoint = clean(configuration.s3Endpoint).replace(/\/+$/, '');
  const s3AccessKeyId = clean(configuration.s3AccessKeyId);
  const s3SecretAccessKey = clean(configuration.s3SecretAccessKey);
  const timeoutMs = Math.max(60_000, Number(configuration.timeoutMs) || 60 * 60 * 1000);
  const pollIntervalMs = Math.max(250, Number(configuration.pollIntervalMs) || 2_000);
  const fetchImpl = dependencies.fetchImpl || fetch;

  const missing = [];
  if (!endpointId) missing.push('RUNPOD_SERVERLESS_ENDPOINT_ID');
  if (!apiKey) missing.push('RUNPOD_API_KEY');
  if (!volumeId) missing.push('RUNPOD_NETWORK_VOLUME_ID');
  if (!region) missing.push('RUNPOD_S3_REGION');
  if (!s3Endpoint) missing.push('RUNPOD_S3_ENDPOINT');
  if (!s3AccessKeyId) missing.push('RUNPOD_S3_ACCESS_KEY_ID');
  if (!s3SecretAccessKey) missing.push('RUNPOD_S3_SECRET_ACCESS_KEY');

  const s3 = dependencies.s3 || (missing.length ? null : new S3Client({
    region,
    endpoint: s3Endpoint,
    forcePathStyle: true,
    credentials: {
      accessKeyId: s3AccessKeyId,
      secretAccessKey: s3SecretAccessKey,
    },
    maxAttempts: 10,
  }));

  async function runpodRequest(pathname, options = {}) {
    const response = await fetchImpl(`https://api.runpod.ai/v2/${endpointId}${pathname}`, {
      ...options,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    let payload = {};
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(`RunPod Serverless returned an invalid response (HTTP ${response.status}).`);
    }
    if (!response.ok) {
      const detail = payload.error || payload.message || text;
      throw new Error(`RunPod Serverless returned HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
    }
    return payload;
  }

  async function removeAudio(key) {
    if (!s3) return;
    try {
      await s3.send(new DeleteObjectCommand({ Bucket: volumeId, Key: key }));
    } catch {
      // The worker normally deletes its own input; this is a crash-recovery cleanup.
    }
  }

  async function transcribe({ job, preparedPath, constraints = [], onProgress = () => {} }) {
    if (missing.length) {
      throw new Error(`RunPod Serverless is missing: ${missing.join(', ')}`);
    }
    const key = `jobs/${job.id}.wav`;
    await s3.send(new PutObjectCommand({
      Bucket: volumeId,
      Key: key,
      Body: fs.createReadStream(preparedPath),
      ContentType: 'audio/wav',
    }));

    let remoteJobId = '';
    const deadline = Date.now() + timeoutMs;
    try {
      const submitted = await runpodRequest('/run', {
        method: 'POST',
        body: JSON.stringify({
          input: {
            audio_path: `/runpod-volume/${key}`,
            delete_audio: true,
            title: job.title,
            instrument: job.instrument,
            instruments: constraints,
          },
          policy: {
            executionTimeout: timeoutMs,
            ttl: Math.min(7 * 24 * 60 * 60 * 1000, Math.max(timeoutMs * 2, 2 * 60 * 60 * 1000)),
          },
        }),
      });
      remoteJobId = clean(submitted.id);
      if (!remoteJobId) throw new Error('RunPod Serverless did not return a job ID.');

      while (Date.now() < deadline) {
        const status = await runpodRequest(`/status/${encodeURIComponent(remoteJobId)}`);
        const state = clean(status.status).toUpperCase();
        onProgress({ state, progress: status.progress, delayTime: status.delayTime });
        if (state === 'COMPLETED') {
          if (!status.output || !Array.isArray(status.output.notes)) {
            throw new Error(status.output?.error || 'RunPod completed without a MuScriptor note result.');
          }
          return status.output;
        }
        if (TERMINAL_FAILURES.has(state)) {
          throw new Error(status.error || `RunPod Serverless job ${state.toLowerCase()}.`);
        }
        await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
      }
      throw new Error(`RunPod Serverless exceeded the ${Math.round(timeoutMs / 60000)}-minute processing limit.`);
    } finally {
      if (remoteJobId && Date.now() >= deadline) {
        runpodRequest(`/cancel/${encodeURIComponent(remoteJobId)}`, { method: 'POST' }).catch(() => {});
      }
      await removeAudio(key);
    }
  }

  return {
    configured: missing.length === 0,
    missing,
    transcribe,
  };
}

module.exports = { createRunpodServerlessClient };
