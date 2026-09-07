'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-artist-campaign-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.ADMIN_EMAILS = 'campaign-admin@example.test';
process.env.DATABASE_URL = '';
process.env.ARTIFACT_S3_BUCKET = '';

const { app } = require('./server');

test('admin creates a gated artist challenge and only a verified publication becomes public', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(testDataDir, { recursive: true, force: true });
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  async function jsonApi(pathname, { method = 'GET', token = '', body } = {}) {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { response, data: await response.json() };
  }

  const challenge = await jsonApi('/api/auth/register/otp', {
    method: 'POST', body: { channel: 'email', email: 'campaign-admin@example.test' },
  });
  const registration = await jsonApi('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Campaign Admin',
      email: 'campaign-admin@example.test',
      password: 'CampaignAdminPassword123',
      challengeId: challenge.data.challengeId,
      verificationCode: '123456',
    },
  });
  assert.equal(registration.response.status, 201);
  const token = registration.data.token;

  const song = Buffer.from(JSON.stringify({
    title: 'Midnight Practice', composer: 'Independent Artist', bpm: 92,
    privateArrangementNotes: 'This field must remain in the private source only.',
    notes: [
      { note: 'A3', time: 1, duration: 1, velocity: 0.5, hand: 'left' },
      { note: 'C4', time: 12, duration: 0.8, velocity: 0.72, hand: 'left' },
      { note: 'G4', time: 13, duration: 1.1, velocity: 0.88, hand: 'right' },
      { note: 'C6', time: 33, duration: 1, velocity: 0.9, hand: 'right' },
    ],
    pedals: [{ time: 12, down: true }, { time: 14, down: false }],
  }));
  const createForm = new FormData();
  createForm.append('artist', 'Independent Artist');
  createForm.append('title', 'Midnight Practice');
  createForm.append('hook', 'Can you play the chorus?');
  createForm.append('status', 'draft');
  createForm.append('previewStartSeconds', '12');
  createForm.append('previewDurationSeconds', '20');
  createForm.append('song', new Blob([song], { type: 'application/json' }), 'challenge.json');
  const createdResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns`, {
    method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: createForm,
  });
  const created = await createdResponse.json();
  assert.equal(createdResponse.status, 201);
  assert.equal(created.campaign.status, 'draft');
  assert.equal(created.campaign.publishProblems.length > 0, true);

  const unauthenticatedPreview = await jsonApi(`/api/admin/artist-campaigns/preview/${created.campaign.slug}`);
  assert.equal(unauthenticatedPreview.response.status, 401);
  const privatePreview = await jsonApi(`/api/admin/artist-campaigns/preview/${created.campaign.slug}`, { token });
  assert.equal(privatePreview.response.status, 200);
  assert.equal(privatePreview.data.campaign.adminPreview, true);
  assert.equal(privatePreview.data.campaign.live, false);
  const privateSongResponse = await fetch(`${baseUrl}${privatePreview.data.campaign.songUrl}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  assert.equal(privateSongResponse.status, 200);
  assert.equal(privateSongResponse.headers.get('cache-control'), 'private, no-store');

  const hidden = await jsonApi(`/api/artist-campaigns/${created.campaign.slug}`);
  assert.equal(hidden.response.status, 404);

  const premature = new FormData();
  premature.append('status', 'published');
  const prematureResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: premature,
  });
  const prematureData = await prematureResponse.json();
  assert.equal(prematureResponse.status, 409);
  assert.equal(prematureData.publishProblems.length > 0, true);

  const verify = new FormData();
  verify.append('qaScore', '92');
  verify.append('humanVerified', 'true');
  verify.append('verificationNotes', 'Pitches, timing, holds, dynamics, and pedal checked by a pianist.');
  verify.append('rightsConfirmed', 'true');
  verify.append('rightsHolder', 'Independent Artist');
  verify.append('rightsBasis', 'Written campaign agreement dated 2026-09-07');
  verify.append('artistApproved', 'true');
  verify.append('status', 'draft');
  const verifiedResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: verify,
  });
  const verified = await verifiedResponse.json();
  assert.equal(verifiedResponse.status, 200);
  assert.deepEqual(verified.campaign.publishProblems, []);

  const publish = new FormData();
  publish.append('status', 'published');
  const publishResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: publish,
  });
  assert.equal(publishResponse.status, 200);

  const publicResponse = await jsonApi(`/api/artist-campaigns/${created.campaign.slug}`);
  assert.equal(publicResponse.response.status, 200);
  assert.equal(publicResponse.data.campaign.live, true);
  assert.equal(publicResponse.data.campaign.verification.qaScore, 92);
  assert.equal(publicResponse.data.campaign.preview.startSeconds, 0);
  assert.equal('rightsBasis' in publicResponse.data.campaign, false);
  assert.equal(publicResponse.data.campaign.songFormat, 'JSON');
  assert.match(publicResponse.data.campaign.songFilename, /-challenge\.json$/);

  const songResponse = await fetch(`${baseUrl}/api/artist-campaigns/${created.campaign.slug}/song`);
  assert.equal(songResponse.status, 200);
  assert.equal(songResponse.headers.get('content-type'), 'application/json');
  assert.equal(songResponse.headers.get('cache-control'), 'private, no-store');
  const publicSong = await songResponse.json();
  assert.deepEqual(publicSong.notes.map((note) => [note.note, note.time]), [['C4', 0], ['G4', 1]]);
  assert.equal(publicSong.duration, 20);
  assert.equal('privateArrangementNotes' in publicSong, false);

  const pause = new FormData();
  pause.append('status', 'paused');
  const pauseResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: pause,
  });
  assert.equal(pauseResponse.status, 200);
  const pausedPublic = await jsonApi(`/api/artist-campaigns/${created.campaign.slug}`);
  assert.equal(pausedPublic.response.status, 404);

  const republish = new FormData();
  republish.append('status', 'published');
  const republishResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: republish,
  });
  assert.equal(republishResponse.status, 200);

  const unsafeLiveEdit = new FormData();
  unsafeLiveEdit.append('previewStartSeconds', '13');
  unsafeLiveEdit.append('status', 'published');
  unsafeLiveEdit.append('humanVerified', 'true');
  unsafeLiveEdit.append('qaScore', '99');
  unsafeLiveEdit.append('verificationNotes', 'Tried to reuse stale verification.');
  unsafeLiveEdit.append('artistApproved', 'true');
  const unsafeLiveEditResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns/${created.campaign.id}`, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}` }, body: unsafeLiveEdit,
  });
  const safetyDraft = await unsafeLiveEditResponse.json();
  assert.equal(unsafeLiveEditResponse.status, 200);
  assert.equal(safetyDraft.campaign.status, 'draft');
  assert.equal(safetyDraft.campaign.verification.humanVerified, false);
  assert.equal(safetyDraft.campaign.verification.qaScore, 0);
  assert.equal(safetyDraft.campaign.artistApproved, false);
  assert.equal(safetyDraft.campaign.publishProblems.some((problem) => /human pianist/i.test(problem)), true);
  const editedPublic = await jsonApi(`/api/artist-campaigns/${created.campaign.slug}`);
  assert.equal(editedPublic.response.status, 404);
});
