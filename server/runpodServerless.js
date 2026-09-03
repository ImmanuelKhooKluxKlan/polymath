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

function parseReplicas(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(String(value));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function createRunpodS3Client(configuration) {
  const client = new S3Client({
    ...configuration,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
  });
  client.middlewareStack.addRelativeTo(
    (next) => async (args) => {
      const output = await next(args);
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
      name: 'normalizeRunpodDates',
    },
  );
  return client;
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
  const inferenceVersion = clean(configuration.inferenceVersion || 'original').toLowerCase();
  const fetchImpl = dependencies.fetchImpl || fetch;

  const missing = [];
  if (!endpointId) missing.push('RUNPOD_SERVERLESS_ENDPOINT_ID');
  if (!apiKey) missing.push('RUNPOD_API_KEY');
  if (!volumeId) missing.push('RUNPOD_NETWORK_VOLUME_ID');
  if (!region) missing.push('RUNPOD_S3_REGION');
  if (!s3Endpoint) missing.push('RUNPOD_S3_ENDPOINT');
  if (!s3AccessKeyId) missing.push('RUNPOD_S3_ACCESS_KEY_ID');
  if (!s3SecretAccessKey) missing.push('RUNPOD_S3_SECRET_ACCESS_KEY');

  const storageTargets = [{ volumeId, region, s3Endpoint }];
  for (const [index, replica] of parseReplicas(configuration.replicas).entries()) {
    const target = {
      volumeId: clean(replica?.volumeId),
      region: clean(replica?.region),
      s3Endpoint: clean(replica?.s3Endpoint).replace(/\/+$/, ''),
    };
    if (!target.volumeId) missing.push(`RUNPOD_S3_REPLICAS[${index}].volumeId`);
    if (!target.region) missing.push(`RUNPOD_S3_REPLICAS[${index}].region`);
    if (!target.s3Endpoint) missing.push(`RUNPOD_S3_REPLICAS[${index}].s3Endpoint`);
    if (target.volumeId && !storageTargets.some((candidate) => candidate.volumeId === target.volumeId)) {
      storageTargets.push(target);
    }
  }

  const storageClients = missing.length ? [] : storageTargets.map((target, index) => ({
    ...target,
    s3: dependencies.s3Clients?.[index] || dependencies.s3 || createRunpodS3Client({
      region: target.region,
      endpoint: target.s3Endpoint,
      forcePathStyle: true,
      credentials: {
        accessKeyId: s3AccessKeyId,
        secretAccessKey: s3SecretAccessKey,
      },
      maxAttempts: 10,
    }),
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
    await Promise.all(storageClients.map(async ({ s3, volumeId: targetVolumeId }) => {
      try {
        await s3.send(new DeleteObjectCommand({ Bucket: targetVolumeId, Key: key }));
      } catch {
        // The worker normally deletes its own input; this is a crash-recovery cleanup.
      }
    }));
  }

  function assertConfigured() {
    if (missing.length) {
      throw new Error(`RunPod Serverless is missing: ${missing.join(', ')}`);
    }
  }

  async function submitAction(input, policy = {}) {
    assertConfigured();
    if (!input || typeof input !== 'object' || Array.isArray(input)) {
      throw new Error('RunPod action input must be an object.');
    }
    return runpodRequest('/run', {
      method: 'POST',
      body: JSON.stringify({
        input,
        policy: {
          executionTimeout: Math.max(60_000, Number(policy.executionTimeout) || timeoutMs),
          ttl: Math.max(60_000, Number(policy.ttl) || Math.min(7 * 24 * 60 * 60 * 1000, timeoutMs * 2)),
        },
      }),
    });
  }

  async function getJobStatus(jobId) {
    assertConfigured();
    const id = clean(jobId);
    if (!id) throw new Error('RunPod job ID is required.');
    return runpodRequest(`/status/${encodeURIComponent(id)}`);
  }

  async function cancelJob(jobId) {
    assertConfigured();
    const id = clean(jobId);
    if (!id) throw new Error('RunPod job ID is required.');
    return runpodRequest(`/cancel/${encodeURIComponent(id)}`, { method: 'POST' });
  }

  async function transcribe({ job, preparedPath, constraints = [], onProgress = () => {} }) {
    assertConfigured();
    const key = `jobs/${job.id}.wav`;
    const uploads = await Promise.allSettled(storageClients.map(({ s3, volumeId: targetVolumeId }) => s3.send(new PutObjectCommand({
        Bucket: targetVolumeId,
        Key: key,
        Body: fs.createReadStream(preparedPath),
        ContentType: 'audio/wav',
      }))));
    const failedUpload = uploads.find((result) => result.status === 'rejected');
    if (failedUpload) {
      await removeAudio(key);
      const error = failedUpload.reason;
      throw new Error(`RunPod job replication failed before submission: ${error?.message || error}`);
    }

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
            checkpoint_version: inferenceVersion,
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
            throw new Error(status.output?.error || 'RunPod completed without a Polymath note result.');
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
    cancelJob,
    getJobStatus,
    missing,
    storageTargetCount: storageTargets.length,
    inferenceVersion,
    submitAction,
    transcribe,
  };
}

module.exports = { createRunpodServerlessClient, parseReplicas };
