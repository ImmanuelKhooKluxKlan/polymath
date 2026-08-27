const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { createRunpodServerlessClient } = require('./runpodServerless');

test('uploads audio, polls a permanent endpoint, and cleans the volume object', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'runpod-serverless-test-'));
  const audioPath = path.join(directory, 'prepared.wav');
  fs.writeFileSync(audioPath, 'wav-fixture');
  const s3Commands = [];
  const requests = [];
  const replies = [
    { id: 'remote-job-1', status: 'IN_QUEUE' },
    { id: 'remote-job-1', status: 'IN_PROGRESS', progress: 'Transcribing 1 of 2 audio sections' },
    { id: 'remote-job-1', status: 'COMPLETED', output: { notes: [{ midi: 60 }] } },
  ];
  const client = createRunpodServerlessClient({
    endpointId: 'endpoint-1',
    apiKey: 'secret',
    volumeId: 'volume-1',
    region: 'US-KS-2',
    s3Endpoint: 'https://s3api-us-ks-2.runpod.io',
    s3AccessKeyId: 'user-test',
    s3SecretAccessKey: 'rps-test',
    timeoutMs: 60_000,
    pollIntervalMs: 1,
  }, {
    s3: { async send(command) { s3Commands.push(command.constructor.name); } },
    async fetchImpl(url, options) {
      requests.push({ url, options });
      const reply = replies.shift();
      return { ok: true, status: 200, async text() { return JSON.stringify(reply); } };
    },
  });

  try {
    const result = await client.transcribe({
      job: { id: 'media-1', title: 'Fixture', instrument: 'piano' },
      preparedPath: audioPath,
      constraints: ['voice'],
    });
    assert.equal(result.notes[0].midi, 60);
    assert.deepEqual(s3Commands, ['PutObjectCommand', 'DeleteObjectCommand']);
    assert.match(requests[0].url, /endpoint-1\/run$/);
    assert.match(requests.at(-1).url, /status\/remote-job-1$/);
    const submitted = JSON.parse(requests[0].options.body);
    assert.equal(submitted.input.audio_path, '/runpod-volume/jobs/media-1.wav');
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('reports missing configuration without exposing secret values', () => {
  const client = createRunpodServerlessClient({});
  assert.equal(client.configured, false);
  assert.ok(client.missing.includes('RUNPOD_API_KEY'));
});

test('replicates each job to every configured volume before submission and cleans every copy', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'runpod-serverless-replica-test-'));
  const audioPath = path.join(directory, 'prepared.wav');
  fs.writeFileSync(audioPath, 'replicated-wav-fixture');
  const commands = [];
  const client = createRunpodServerlessClient({
    endpointId: 'endpoint-2',
    apiKey: 'secret',
    volumeId: 'volume-primary',
    region: 'US-MO-2',
    s3Endpoint: 'https://s3api-us-mo-2.runpod.io',
    s3AccessKeyId: 'user-test',
    s3SecretAccessKey: 'rps-test',
    replicas: JSON.stringify([{ volumeId: 'volume-failover', region: 'EU-RO-1', s3Endpoint: 'https://s3api-eu-ro-1.runpod.io' }]),
    timeoutMs: 60_000,
    pollIntervalMs: 1,
  }, {
    s3: {
      async send(command) {
        commands.push({ name: command.constructor.name, bucket: command.input.Bucket });
        if (command.constructor.name === 'PutObjectCommand') {
          for await (const chunk of command.input.Body) {
            assert.ok(chunk.length > 0);
          }
        }
      },
    },
    async fetchImpl(url) {
      const payload = url.endsWith('/run')
        ? { id: 'replicated-job' }
        : { status: 'COMPLETED', output: { notes: [{ midi: 64 }] } };
      return { ok: true, status: 200, async text() { return JSON.stringify(payload); } };
    },
  });

  try {
    const result = await client.transcribe({
      job: { id: 'media-replicated', title: 'Replicated fixture', instrument: 'piano' },
      preparedPath: audioPath,
    });
    assert.equal(result.notes[0].midi, 64);
    assert.equal(client.storageTargetCount, 2);
    assert.deepEqual(commands, [
      { name: 'PutObjectCommand', bucket: 'volume-primary' },
      { name: 'PutObjectCommand', bucket: 'volume-failover' },
      { name: 'DeleteObjectCommand', bucket: 'volume-primary' },
      { name: 'DeleteObjectCommand', bucket: 'volume-failover' },
    ]);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
