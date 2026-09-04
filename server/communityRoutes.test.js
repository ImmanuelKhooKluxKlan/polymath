const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-community-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'admin@example.test';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';

const { app, readDb, writeDb } = require('./server');

test('paid members can use Free Flow and private invite groups while free accounts cannot', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(testDataDir, { recursive: true, force: true });
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  async function api(pathname, { method = 'GET', token = '', body } = {}) {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { status: response.status, data: response.status === 204 ? null : await response.json() };
  }

  async function register(name, email) {
    const challenge = await api('/api/auth/register/otp', { method: 'POST', body: { channel: 'email', email } });
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: { name, email, password: 'CommunityPassword123', challengeId: challenge.data.challengeId, verificationCode: '123456' },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const owner = await register('Room Owner', 'owner@example.test');
  const invited = await register('Invited Member', 'invited@example.test');
  const free = await register('Free Member', 'free@example.test');
  const db = await readDb();
  [owner.user.user_id, invited.user.user_id].forEach((userId) => {
    const user = db.users.find((candidate) => candidate.id === userId);
    user.subscriptionTier = 'chill';
    user.proStatus = 'ACTIVE';
    user.pro = true;
  });
  await writeDb(db);

  const blocked = await api('/api/community/rooms', { token: free.token });
  assert.equal(blocked.status, 403);
  assert.equal(blocked.data.code, 'SUBSCRIPTION_REQUIRED');

  const rooms = await api('/api/community/rooms', { token: owner.token });
  assert.equal(rooms.status, 200);
  const global = rooms.data.rooms.find((room) => room.visibility === 'global');
  assert.ok(global);
  const sentGlobal = await api(`/api/community/rooms/${global.id}/messages`, {
    method: 'POST', token: owner.token, body: { text: 'Hello Polymath!' },
  });
  assert.equal(sentGlobal.status, 201);

  const created = await api('/api/community/rooms', {
    method: 'POST', token: owner.token, body: { name: 'Quiet Piano Club', topic: 'Practice together', visibility: 'private' },
  });
  assert.equal(created.status, 201);
  assert.ok(created.data.room.inviteCode);
  const privateId = created.data.room.id;
  assert.equal((await api(`/api/community/rooms/${privateId}/messages`, { token: invited.token })).status, 404);
  const joined = await api('/api/community/rooms/join', {
    method: 'POST', token: invited.token, body: { inviteCode: created.data.room.inviteCode },
  });
  assert.equal(joined.status, 200);
  assert.equal(joined.data.room.inviteCode, undefined);
  const privateMessage = await api(`/api/community/rooms/${privateId}/messages`, {
    method: 'POST', token: invited.token, body: { text: 'I joined safely.' },
  });
  assert.equal(privateMessage.status, 201);
  const reported = await api(`/api/community/messages/${sentGlobal.data.message.id}/report`, {
    method: 'POST', token: invited.token, body: {},
  });
  assert.equal(reported.status, 201);
});
