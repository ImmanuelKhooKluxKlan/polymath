'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-affiliate-program-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.REGISTRATION_OTP_SECRET = 'affiliate-program-test-secret-value-that-persists';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.ADMIN_EMAILS = 'growth-admin@example.test';
process.env.DATABASE_URL = '';
process.env.ARTIFACT_S3_BUCKET = '';
process.env.CLIENT_ORIGIN = 'https://polymathmusician67.com';

const { app, readDb } = require('./server');

test('admin creates an auditable percentage referral program and creators see aggregate results', async (context) => {
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
      method: 'POST', body: { channel: 'email', email },
    });
    const result = await api('/api/auth/register', {
      method: 'POST',
      body: {
        name,
        email,
        password: 'AffiliateProgramPassword123',
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
    assert.equal(result.status, 201);
    return result.data;
  }

  const admin = await register('Growth Admin', 'growth-admin@example.test');
  const creator = await register('Piano Creator', 'creator@example.test');
  const buyer = await register('Music Student', 'buyer@example.test');

  const created = await api('/api/admin/promotions', {
    method: 'POST',
    token: admin.token,
    body: {
      code: 'PIANO20',
      name: 'Piano creator launch',
      kind: 'subscription_percent',
      value: 20,
      perUserLimit: 1,
      affiliateUserId: creator.user.user_id,
      affiliateRewardMode: 'percent',
      affiliateRewardValue: 10,
    },
  });
  assert.equal(created.status, 201);
  assert.equal(created.data.promotion.affiliateRewardMode, 'percent');
  assert.equal(created.data.promotion.affiliateRewardValue, 10);
  assert.equal(created.data.promotion.affiliateRewardMcoins, 0);
  assert.equal(created.data.promotion.shareUrl, 'https://polymathmusician67.com/#payment?promo=PIANO20');

  const publicPreview = await api('/api/promotions/subscription-preview?code=piano20');
  assert.equal(publicPreview.status, 200);
  assert.deepEqual(publicPreview.data.promotion, {
    code: 'PIANO20', kind: 'subscription_percent', value: 20,
  });
  assert.equal('affiliateUserId' in publicPreview.data.promotion, false);

  const selfReferral = await api('/api/promotions/claim-subscription', {
    method: 'POST', token: creator.token, body: { code: 'PIANO20' },
  });
  assert.equal(selfReferral.status, 400);
  assert.match(selfReferral.data.error, /own referral code/i);

  const claimed = await api('/api/promotions/claim-subscription', {
    method: 'POST', token: buyer.token, body: { code: 'piano20' },
  });
  assert.equal(claimed.status, 200);
  assert.deepEqual(claimed.data.user.subscriptionOffer, {
    code: 'PIANO20', kind: 'subscription_percent', value: 20, claimedAt: claimed.data.user.subscriptionOffer.claimedAt,
  });

  const duplicate = await api('/api/promotions/claim-subscription', {
    method: 'POST', token: buyer.token, body: { code: 'PIANO20' },
  });
  assert.equal(duplicate.status, 200);
  const database = await readDb();
  assert.equal(database.promotionRedemptions.filter((entry) => !entry.supersededAt).length, 1);

  const dashboard = await api('/api/referrals/me', { token: creator.token });
  assert.equal(dashboard.status, 200);
  assert.equal(dashboard.data.programs.length, 1);
  assert.equal(dashboard.data.programs[0].claims, 1);
  assert.equal(dashboard.data.programs[0].commission.value, 10);
  assert.match(dashboard.data.programs[0].disclosure, /may earn Mcoins/i);
  assert.equal('buyers' in dashboard.data.programs[0], false, 'buyer identity must remain private');

});
