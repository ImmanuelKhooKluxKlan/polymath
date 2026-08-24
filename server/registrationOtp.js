const crypto = require('node:crypto');
const { SESv2Client, SendEmailCommand } = require('@aws-sdk/client-sesv2');
const { SNSClient, PublishCommand } = require('@aws-sdk/client-sns');

const OTP_LENGTH = 6;
const DEFAULT_TTL_MINUTES = 10;
const DEFAULT_RESEND_SECONDS = 60;
const DEFAULT_HOURLY_LIMIT = 5;
const MAX_ATTEMPTS = 5;

class RegistrationOtpError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = 'RegistrationOtpError';
    this.status = status;
  }
}

function clampInteger(value, minimum, maximum, fallback) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function normalizeRegistrationContact(channel, value) {
  if (channel === 'email') {
    const email = String(value || '').trim().toLowerCase();
    if (!/^\S+@\S+\.\S+$/.test(email) || email.length > 254) {
      throw new RegistrationOtpError('Enter a valid email address.');
    }
    return email;
  }

  if (channel === 'phone') {
    const phone = String(value || '').trim().replace(/[\s()-]/g, '');
    if (!/^\+[1-9]\d{7,14}$/.test(phone)) {
      throw new RegistrationOtpError('Enter a valid phone number with its country code, for example +6581234567.');
    }
    return phone;
  }

  throw new RegistrationOtpError('Choose email or phone verification.');
}

function maskContact(channel, destination) {
  if (channel === 'email') {
    const [local, domain] = destination.split('@');
    return `${local.slice(0, 2)}${'*'.repeat(Math.max(1, Math.min(5, local.length - 2)))}@${domain}`;
  }
  return `${destination.slice(0, 3)}${'*'.repeat(Math.max(4, destination.length - 7))}${destination.slice(-4)}`;
}

function destinationHash(channel, destination) {
  return crypto.createHash('sha256').update(`${channel}:${destination}`).digest('hex');
}

function codeHash(secret, challengeId, destinationDigest, code) {
  return crypto
    .createHmac('sha256', secret)
    .update(`${challengeId}:${destinationDigest}:${code}`)
    .digest('hex');
}

function safeEqualHex(left, right) {
  const leftBuffer = Buffer.from(String(left || ''), 'hex');
  const rightBuffer = Buffer.from(String(right || ''), 'hex');
  return leftBuffer.length > 0
    && leftBuffer.length === rightBuffer.length
    && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

function createRegistrationOtpService(environment = process.env) {
  const nodeEnvironment = String(environment.NODE_ENV || '').trim().toLowerCase();
  const testCode = nodeEnvironment === 'test'
    ? String(environment.REGISTRATION_OTP_TEST_CODE || '').trim()
    : '';
  if (testCode && !/^\d{6}$/.test(testCode)) {
    throw new Error('REGISTRATION_OTP_TEST_CODE must contain exactly six digits.');
  }

  const secret = String(environment.REGISTRATION_OTP_SECRET || '').trim()
    || (testCode ? 'polymath-registration-otp-test-secret-only' : '');
  const region = String(environment.OTP_AWS_REGION || environment.AWS_REGION || 'us-east-2').trim();
  const accessKeyId = String(environment.OTP_AWS_ACCESS_KEY_ID || '').trim();
  const secretAccessKey = String(environment.OTP_AWS_SECRET_ACCESS_KEY || '').trim();
  const sessionToken = String(environment.OTP_AWS_SESSION_TOKEN || '').trim();
  const emailFrom = String(environment.OTP_EMAIL_FROM || '').trim();
  const smsEnabled = String(environment.OTP_SMS_ENABLED || '').trim().toLowerCase() === 'true';
  const ttlMinutes = clampInteger(environment.REGISTRATION_OTP_TTL_MINUTES, 3, 30, DEFAULT_TTL_MINUTES);
  const resendSeconds = clampInteger(environment.REGISTRATION_OTP_RESEND_SECONDS, 30, 600, DEFAULT_RESEND_SECONDS);
  const hourlyLimit = clampInteger(environment.REGISTRATION_OTP_HOURLY_LIMIT, 2, 20, DEFAULT_HOURLY_LIMIT);

  function requireSecurityConfiguration() {
    if (secret.length < 32) {
      throw new RegistrationOtpError('Account verification is temporarily unavailable.', 503);
    }
  }

  function awsConfiguration() {
    const configuration = { region };
    if (!accessKeyId || !secretAccessKey) return configuration;
    const credentials = {
      accessKeyId,
      secretAccessKey,
      ...(sessionToken ? { sessionToken } : {}),
    };
    return { ...configuration, credentials };
  }

  async function deliverCode(channel, destination, code) {
    if (testCode) return;

    try {
      if (channel === 'email') {
        if (!emailFrom) {
          throw new RegistrationOtpError('Email verification is temporarily unavailable.', 503);
        }
        const client = new SESv2Client(awsConfiguration());
        await client.send(new SendEmailCommand({
          FromEmailAddress: emailFrom,
          Destination: { ToAddresses: [destination] },
          Content: {
            Simple: {
              Subject: { Data: 'Your Polymath Musician verification code', Charset: 'UTF-8' },
              Body: {
                Text: {
                  Data: `Your Polymath Musician verification code is ${code}. It expires in ${ttlMinutes} minutes. If you did not request this code, ignore this message.`,
                  Charset: 'UTF-8',
                },
              },
            },
          },
        }));
        return;
      }

      if (!smsEnabled) {
        throw new RegistrationOtpError('Phone verification is temporarily unavailable.', 503);
      }
      const client = new SNSClient(awsConfiguration());
      await client.send(new PublishCommand({
        PhoneNumber: destination,
        Message: `Polymath Musician code: ${code}. Expires in ${ttlMinutes} minutes.`,
        MessageAttributes: {
          'AWS.SNS.SMS.SMSType': { DataType: 'String', StringValue: 'Transactional' },
        },
      }));
    } catch (error) {
      if (error instanceof RegistrationOtpError) throw error;
      console.error(`Registration OTP ${channel} delivery failed:`, error?.name || error?.message || error);
      throw new RegistrationOtpError(`We could not send the ${channel === 'email' ? 'email' : 'text message'} verification code. Try again later.`, 503);
    }
  }

  async function requestCode(db, { channel, destination }) {
    requireSecurityConfiguration();
    const normalizedDestination = normalizeRegistrationContact(channel, destination);
    const digest = destinationHash(channel, normalizedDestination);
    const now = Date.now();
    const oneHourAgo = now - 60 * 60 * 1000;
    db.registrationVerifications = (db.registrationVerifications || []).filter((challenge) => (
      new Date(challenge.createdAt).getTime() >= now - 24 * 60 * 60 * 1000
    ));

    const recent = db.registrationVerifications
      .filter((challenge) => challenge.destinationHash === digest
        && new Date(challenge.createdAt).getTime() >= oneHourAgo)
      .sort((left, right) => new Date(right.createdAt) - new Date(left.createdAt));
    const latestCreatedAt = recent[0] ? new Date(recent[0].createdAt).getTime() : 0;
    if (latestCreatedAt && now - latestCreatedAt < resendSeconds * 1000) {
      const waitSeconds = Math.max(1, Math.ceil((resendSeconds * 1000 - (now - latestCreatedAt)) / 1000));
      throw new RegistrationOtpError(`Wait ${waitSeconds} seconds before requesting another code.`, 429);
    }
    if (recent.length >= hourlyLimit) {
      throw new RegistrationOtpError('Too many verification codes were requested. Try again in one hour.', 429);
    }

    const code = testCode || String(crypto.randomInt(0, 10 ** OTP_LENGTH)).padStart(OTP_LENGTH, '0');
    const challengeId = `otp_${crypto.randomUUID()}`;
    const createdAt = new Date(now).toISOString();
    const expiresAt = new Date(now + ttlMinutes * 60 * 1000).toISOString();
    await deliverCode(channel, normalizedDestination, code);
    db.registrationVerifications.push({
      id: challengeId,
      channel,
      destinationHash: digest,
      codeHash: codeHash(secret, challengeId, digest, code),
      attempts: 0,
      createdAt,
      expiresAt,
      verifiedAt: null,
    });

    return {
      challengeId,
      channel,
      destination: normalizedDestination,
      destinationHint: maskContact(channel, normalizedDestination),
      expiresInSeconds: ttlMinutes * 60,
    };
  }

  function verifyCode(db, { challengeId, channel, destination, code }) {
    requireSecurityConfiguration();
    const normalizedDestination = normalizeRegistrationContact(channel, destination);
    const digest = destinationHash(channel, normalizedDestination);
    const challenge = (db.registrationVerifications || []).find((candidate) => candidate.id === challengeId);
    if (!challenge || challenge.channel !== channel || challenge.destinationHash !== digest) {
      throw new RegistrationOtpError('The verification request is invalid or expired. Request a new code.');
    }
    if (challenge.verifiedAt) {
      throw new RegistrationOtpError('That verification code has already been used.');
    }
    if (new Date(challenge.expiresAt).getTime() < Date.now()) {
      throw new RegistrationOtpError('The verification code expired. Request a new code.');
    }
    if (Number(challenge.attempts || 0) >= MAX_ATTEMPTS) {
      throw new RegistrationOtpError('Too many incorrect attempts. Request a new code.', 429);
    }

    challenge.attempts = Number(challenge.attempts || 0) + 1;
    const submittedCode = String(code || '').trim();
    const submittedHash = /^\d{6}$/.test(submittedCode)
      ? codeHash(secret, challenge.id, digest, submittedCode)
      : '';
    if (!safeEqualHex(challenge.codeHash, submittedHash)) {
      if (challenge.attempts >= MAX_ATTEMPTS) {
        throw new RegistrationOtpError('Too many incorrect attempts. Request a new code.', 429);
      }
      throw new RegistrationOtpError('The verification code is incorrect.');
    }

    challenge.verifiedAt = new Date().toISOString();
    return { channel, destination: normalizedDestination, verifiedAt: challenge.verifiedAt };
  }

  return {
    requestCode,
    verifyCode,
    normalizeContact: normalizeRegistrationContact,
  };
}

module.exports = {
  RegistrationOtpError,
  createRegistrationOtpService,
  normalizeRegistrationContact,
};
