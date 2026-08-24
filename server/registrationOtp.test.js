const test = require('node:test');
const assert = require('node:assert/strict');
const {
  createRegistrationOtpService,
  normalizeRegistrationContact,
} = require('./registrationOtp');

function testService() {
  return createRegistrationOtpService({
    NODE_ENV: 'test',
    REGISTRATION_OTP_TEST_CODE: '123456',
    REGISTRATION_OTP_RESEND_SECONDS: '30',
  });
}

test('registration OTP is hashed, contact-bound, one-time, and case-normalized', async () => {
  const service = testService();
  const db = { registrationVerifications: [] };
  const challenge = await service.requestCode(db, {
    channel: 'email',
    destination: 'NEW.USER@Example.COM',
  });

  assert.equal(challenge.destination, 'new.user@example.com');
  assert.equal(db.registrationVerifications.length, 1);
  assert.equal(db.registrationVerifications[0].code, undefined);
  assert.equal(db.registrationVerifications[0].destination, undefined);
  assert.doesNotMatch(JSON.stringify(db.registrationVerifications), /123456|new\.user@example\.com/);

  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'email',
    destination: 'different@example.com',
    code: '123456',
  }), /invalid or expired/);
  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'email',
    destination: 'new.user@example.com',
    code: '654321',
  }), /incorrect/);

  const verified = service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'email',
    destination: 'new.user@example.com',
    code: '123456',
  });
  assert.equal(verified.destination, 'new.user@example.com');
  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'email',
    destination: 'new.user@example.com',
    code: '123456',
  }), /already been used/);
});

test('registration OTP locks after five wrong attempts', async () => {
  const service = testService();
  const db = { registrationVerifications: [] };
  const challenge = await service.requestCode(db, {
    channel: 'phone',
    destination: '+65 8123 4567',
  });

  for (let attempt = 1; attempt <= 4; attempt += 1) {
    assert.throws(() => service.verifyCode(db, {
      challengeId: challenge.challengeId,
      channel: 'phone',
      destination: '+6581234567',
      code: '000000',
    }), /incorrect/);
  }
  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'phone',
    destination: '+6581234567',
    code: '000000',
  }), (error) => error.status === 429);
  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'phone',
    destination: '+6581234567',
    code: '123456',
  }), (error) => error.status === 429);
});

test('registration OTP rejects expired challenges', async () => {
  const service = testService();
  const db = { registrationVerifications: [] };
  const challenge = await service.requestCode(db, {
    channel: 'email',
    destination: 'expired@example.com',
  });
  db.registrationVerifications[0].expiresAt = new Date(Date.now() - 1000).toISOString();

  assert.throws(() => service.verifyCode(db, {
    challengeId: challenge.challengeId,
    channel: 'email',
    destination: 'expired@example.com',
    code: '123456',
  }), /expired/);
});

test('registration contacts require a valid email or international phone number', () => {
  assert.equal(normalizeRegistrationContact('phone', '+65 8123-4567'), '+6581234567');
  assert.throws(() => normalizeRegistrationContact('email', 'not-an-email'), /valid email/);
  assert.throws(() => normalizeRegistrationContact('phone', '81234567'), /country code/);
  assert.throws(() => normalizeRegistrationContact('other', 'value'), /Choose email or phone/);
});
