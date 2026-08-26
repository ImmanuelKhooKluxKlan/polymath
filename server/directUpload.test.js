const assert = require('node:assert/strict');
const test = require('node:test');
const { createDirectUploadService } = require('./directUpload');

function fixture(now = 1_800_000_000_000) {
  const objects = new Map();
  const artifactStore = {
    remote: true,
    async createPresignedPut(key) {
      return `https://uploads.example.test/${encodeURIComponent(key)}`;
    },
    async stat(key) {
      if (!objects.has(key)) throw new Error('missing');
      return objects.get(key);
    },
  };
  const service = createDirectUploadService({
    artifactStore,
    signingSecret: 'test-secret-with-at-least-thirty-two-characters',
    ttlSeconds: 900,
    now: () => now,
  });
  return { service, objects };
}

test('direct upload receipt is bound to user, purpose, size, and expiry', async () => {
  const { service, objects } = fixture();
  const intent = await service.create({
    userId: 'user-1',
    purpose: 'score-translation',
    key: 'pending/score-translation/user-1/file.pdf',
    filename: 'file.pdf',
    contentType: 'application/pdf',
    size: 1234,
  });
  assert.equal(intent.direct, true);
  assert.match(intent.uploadUrl, /^https:\/\/uploads\.example\.test\//);

  objects.set('pending/score-translation/user-1/file.pdf', {
    size: 1234,
    contentType: 'application/pdf',
  });
  const inspected = await service.inspect(intent.receipt, {
    userId: 'user-1',
    purpose: 'score-translation',
  });
  assert.equal(inspected.filename, 'file.pdf');
  assert.equal(inspected.stored.size, 1234);
  assert.throws(() => service.verify(intent.receipt, {
    userId: 'user-2',
    purpose: 'score-translation',
  }), /does not belong/i);
  service.now = () => 1_800_000_901_000;
  assert.throws(() => service.verify(intent.receipt, {
    userId: 'user-1',
    purpose: 'score-translation',
  }), /expired/i);
});

test('direct upload rejects tampered, missing, and size-mismatched objects', async () => {
  const { service, objects } = fixture();
  const intent = await service.create({
    userId: 'user-1',
    purpose: 'media-transcription',
    key: 'pending/media-transcription/user-1/recording.mp3',
    filename: 'recording.mp3',
    contentType: 'audio/mpeg',
    size: 4000,
  });

  await assert.rejects(service.inspect(intent.receipt, {
    userId: 'user-1',
    purpose: 'media-transcription',
  }), /not found/i);

  objects.set('pending/media-transcription/user-1/recording.mp3', {
    size: 3999,
    contentType: 'audio/mpeg',
  });
  await assert.rejects(service.inspect(intent.receipt, {
    userId: 'user-1',
    purpose: 'media-transcription',
  }), /size did not match/i);

  const [payload, signature] = intent.receipt.split('.');
  assert.throws(() => service.verify(`${payload.slice(0, -1)}x.${signature}`, {
    userId: 'user-1',
    purpose: 'media-transcription',
  }), /invalid/i);
});

test('direct uploads remain disabled for local storage or a missing secret', async () => {
  const local = createDirectUploadService({
    artifactStore: { remote: false },
    signingSecret: 'test-secret-with-at-least-thirty-two-characters',
  });
  assert.equal(local.enabled, false);
  assert.deepEqual(await local.create({}), { direct: false });
});
