const crypto = require('crypto');

class DirectUploadError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = 'DirectUploadError';
    this.status = status;
  }
}

function encodeJson(value) {
  return Buffer.from(JSON.stringify(value), 'utf8').toString('base64url');
}

function decodeJson(value) {
  try {
    return JSON.parse(Buffer.from(value, 'base64url').toString('utf8'));
  } catch {
    throw new DirectUploadError('The direct-upload receipt is invalid. Upload the file again.');
  }
}

function safeEqual(left, right) {
  const a = Buffer.from(String(left || ''), 'base64url');
  const b = Buffer.from(String(right || ''), 'base64url');
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

class DirectUploadService {
  constructor({ artifactStore, signingSecret, ttlSeconds = 900, now = () => Date.now() }) {
    this.artifactStore = artifactStore;
    this.signingSecret = String(signingSecret || '');
    this.ttlSeconds = Math.max(60, Math.min(3600, Number(ttlSeconds) || 900));
    this.now = now;
  }

  get enabled() {
    return Boolean(this.artifactStore?.remote && this.signingSecret.length >= 32);
  }

  signature(encodedPayload) {
    return crypto.createHmac('sha256', this.signingSecret)
      .update(encodedPayload)
      .digest('base64url');
  }

  sign(payload) {
    const encoded = encodeJson(payload);
    return `${encoded}.${this.signature(encoded)}`;
  }

  verify(receipt, { userId, purpose }) {
    if (!this.enabled) {
      throw new DirectUploadError('Direct uploads are unavailable on this server.', 503);
    }
    const [encoded, suppliedSignature, extra] = String(receipt || '').split('.');
    if (!encoded || !suppliedSignature || extra || !safeEqual(suppliedSignature, this.signature(encoded))) {
      throw new DirectUploadError('The direct-upload receipt is invalid. Upload the file again.');
    }
    const payload = decodeJson(encoded);
    if (payload.version !== 1 || payload.userId !== userId || payload.purpose !== purpose) {
      throw new DirectUploadError('The direct-upload receipt does not belong to this request.', 403);
    }
    if (!Number.isFinite(payload.expiresAt) || payload.expiresAt < this.now()) {
      throw new DirectUploadError('The direct-upload receipt expired. Upload the file again.');
    }
    if (!payload.key || !payload.filename || !Number.isFinite(payload.size) || payload.size <= 0) {
      throw new DirectUploadError('The direct-upload receipt is incomplete. Upload the file again.');
    }
    return payload;
  }

  async create({ userId, purpose, key, filename, contentType, size }) {
    if (!this.enabled) return { direct: false };
    const expiresAt = this.now() + (this.ttlSeconds * 1000);
    const payload = {
      version: 1,
      uploadId: crypto.randomUUID(),
      userId,
      purpose,
      key,
      filename,
      contentType,
      size,
      expiresAt,
    };
    const uploadUrl = await this.artifactStore.createPresignedPut(
      key,
      contentType,
      this.ttlSeconds,
    );
    return {
      direct: true,
      uploadUrl,
      receipt: this.sign(payload),
      contentType,
      expiresAt: new Date(expiresAt).toISOString(),
    };
  }

  async inspect(receipt, expected) {
    const payload = this.verify(receipt, expected);
    let stored;
    try {
      stored = await this.artifactStore.stat(payload.key);
    } catch {
      throw new DirectUploadError('The uploaded file was not found. Upload it again.');
    }
    if (stored.size !== payload.size) {
      throw new DirectUploadError('The uploaded file size did not match the upload ticket. Upload it again.');
    }
    return { ...payload, stored };
  }
}

function createDirectUploadService(options) {
  return new DirectUploadService(options);
}

module.exports = {
  DirectUploadError,
  createDirectUploadService,
};
