const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const samplePath = path.resolve(process.argv[2] || '');
const remoteUrl = String(process.argv[3] || 'http://127.0.0.1:18222').replace(/\/+$/, '');
if (!fs.existsSync(samplePath)) {
  throw new Error('Usage: node runpod.smoke.js <audio-file> [remote-url]');
}

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-runpod-smoke-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'runpod-smoke@example.test';
process.env.MUSCRIPTOR_ENABLED = 'true';
process.env.MUSCRIPTOR_MODEL = 'large';
process.env.MUSCRIPTOR_REMOTE_URL = remoteUrl;

const { app } = require('./server');

async function main() {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  try {
    const registration = await fetch(`${baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'RunPod Smoke Test',
        email: 'runpod-smoke@example.test',
        phone: '+65 8000 0099',
        password: 'RunPodSmokePassword123',
      }),
    });
    assert.equal(registration.status, 201);
    const registered = await registration.json();
    const token = registered.token;

    const form = new FormData();
    form.append('media', new Blob([fs.readFileSync(samplePath)], { type: 'audio/wav' }), path.basename(samplePath));
    form.append('instrument', 'piano');
    form.append('title', 'RunPod integration smoke test');
    form.append('rightsConfirmed', 'true');
    const createdResponse = await fetch(`${baseUrl}/api/media-transcriptions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    assert.equal(createdResponse.status, 202);
    const created = await createdResponse.json();

    const deadline = Date.now() + 120000;
    let job = created.job;
    while (job.status === 'processing' && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const response = await fetch(`${baseUrl}/api/media-transcriptions/${job.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      assert.equal(response.status, 200);
      job = (await response.json()).job;
    }

    assert.equal(job.status, 'completed', job.error || 'RunPod job did not complete');
    assert.ok(job.noteCount > 0);
    const download = await fetch(`${baseUrl}/api/media-transcriptions/${job.id}/download`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    assert.equal(download.status, 200);
    const result = await download.json();
    assert.equal(result.readyToPlayFormat, 'polymath-musician-json-v1');
    assert.match(result.transcriptionProvider, /RunPod GPU/);
    assert.equal(result.notes.length, job.noteCount);
    console.log(JSON.stringify({
      status: job.status,
      execution: 'remote-gpu',
      noteCount: job.noteCount,
      instrumentGroups: job.instrumentGroups,
      provider: result.transcriptionProvider,
    }));
  } finally {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(testDataDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  fs.rmSync(testDataDir, { recursive: true, force: true });
  console.error(error);
  process.exitCode = 1;
});
