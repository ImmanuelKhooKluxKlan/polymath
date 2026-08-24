const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-band-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'host@example.test,member@example.test,outsider@example.test';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';

const { app } = require('./server');

test('band members can chat and creators can kick, ban, and unban accounts', async (context) => {
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
    return { status: response.status, data: await response.json() };
  }

  async function register(name, email) {
    const challenge = await api('/api/auth/register/otp', {
      method: 'POST',
      body: { channel: 'email', email },
    });
    assert.equal(challenge.status, 202);
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: {
        name,
        email,
        password: 'BandPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const host = await register('Band Host', 'host@example.test');
  const member = await register('Band Member', 'member@example.test');
  const outsider = await register('Band Outsider', 'outsider@example.test');
  const avatarDataUrl = `data:image/png;base64,${Buffer.from('small-avatar-fixture').toString('base64')}`;
  const avatar = await api('/api/profile/avatar', {
    method: 'PUT', token: member.token, body: { avatarDataUrl },
  });
  assert.equal(avatar.status, 200);
  assert.equal(avatar.data.user.avatarUrl, avatarDataUrl);

  const created = await api('/api/bands', {
    method: 'POST', token: host.token, body: { name: 'Public Test Band', accessMode: 'open' },
  });
  assert.equal(created.status, 201);
  const bandId = created.data.band.id;

  const joined = await api(`/api/bands/${bandId}/join`, { method: 'POST', token: member.token, body: {} });
  assert.equal(joined.status, 200);
  assert.equal(joined.data.band.members.find((item) => item.userId === member.user.user_id).avatarUrl, avatarDataUrl);

  const sent = await api(`/api/bands/${bandId}/chat`, {
    method: 'POST', token: member.token, body: { text: 'Ready for rehearsal!' },
  });
  assert.equal(sent.status, 201);
  assert.equal(sent.data.message.author.avatarUrl, avatarDataUrl);
  const chat = await api(`/api/bands/${bandId}/chat`, { token: host.token });
  assert.equal(chat.status, 200);
  assert.equal(chat.data.messages[0].text, 'Ready for rehearsal!');
  const privateChat = await api(`/api/bands/${bandId}/chat`, { token: outsider.token });
  assert.equal(privateChat.status, 403);

  const forbiddenKick = await api(`/api/bands/${bandId}/members/${host.user.user_id}`, {
    method: 'DELETE', token: member.token,
  });
  assert.equal(forbiddenKick.status, 403);
  const kicked = await api(`/api/bands/${bandId}/members/${member.user.user_id}`, {
    method: 'DELETE', token: host.token,
  });
  assert.equal(kicked.status, 200);
  const kickedChat = await api(`/api/bands/${bandId}/chat`, {
    method: 'POST', token: member.token, body: { text: 'Can I still chat?' },
  });
  assert.equal(kickedChat.status, 403);

  assert.equal((await api(`/api/bands/${bandId}/join`, { method: 'POST', token: member.token, body: {} })).status, 200);
  const banned = await api(`/api/bands/${bandId}/bans`, {
    method: 'POST', token: host.token, body: { userId: member.user.user_id },
  });
  assert.equal(banned.status, 200);
  assert.equal(banned.data.band.bannedMembers[0].userId, member.user.user_id);
  const bannedJoin = await api(`/api/bands/${bandId}/join`, { method: 'POST', token: member.token, body: {} });
  assert.equal(bannedJoin.status, 403);
  assert.match(bannedJoin.data.error, /banned/i);

  const unbanned = await api(`/api/bands/${bandId}/bans/${member.user.user_id}`, {
    method: 'DELETE', token: host.token,
  });
  assert.equal(unbanned.status, 200);
  assert.equal(unbanned.data.band.bannedMembers.length, 0);
  assert.equal((await api(`/api/bands/${bandId}/join`, { method: 'POST', token: member.token, body: {} })).status, 200);
});
