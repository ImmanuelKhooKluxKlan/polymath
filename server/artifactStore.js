const fs = require('fs');
const path = require('path');
const {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
} = require('@aws-sdk/client-s3');

function safeKey(value) {
  const normalized = String(value || '').replace(/^\/+/, '').replace(/\\/g, '/');
  if (!normalized || normalized.split('/').includes('..')) throw new Error('Invalid artifact path.');
  return normalized;
}

function contentDisposition(filename) {
  const safe = String(filename || 'download').replace(/[\r\n"\\]/g, '_');
  return `attachment; filename="${safe}"`;
}

class ArtifactStore {
  constructor({ localRoot, bucket, region, endpoint, accessKeyId, secretAccessKey }) {
    this.localRoot = path.resolve(localRoot);
    this.bucket = String(bucket || '').trim();
    this.remote = Boolean(this.bucket);
    this.client = this.remote ? new S3Client({
      region: String(region || 'auto'),
      endpoint: String(endpoint || '').trim() || undefined,
      forcePathStyle: Boolean(endpoint),
      credentials: accessKeyId && secretAccessKey ? { accessKeyId, secretAccessKey } : undefined,
    }) : null;
  }

  get provider() {
    return this.remote ? 's3-compatible' : 'local-disk';
  }

  localPath(key) {
    const resolved = path.resolve(this.localRoot, safeKey(key));
    const root = `${this.localRoot}${path.sep}`;
    if (!resolved.startsWith(root)) throw new Error('Invalid artifact path.');
    return resolved;
  }

  async putBuffer(key, bytes, contentType = 'application/octet-stream') {
    const normalized = safeKey(key);
    if (!this.remote) {
      const target = this.localPath(normalized);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, bytes);
      return normalized;
    }
    await this.client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: normalized,
      Body: bytes,
      ContentType: contentType,
      ServerSideEncryption: String(process.env.ARTIFACT_S3_SSE || '').trim() || undefined,
    }));
    return normalized;
  }

  async putFile(key, filename, contentType) {
    const normalized = safeKey(key);
    if (!this.remote) {
      const target = this.localPath(normalized);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      if (path.resolve(filename) !== target) fs.copyFileSync(filename, target);
      return normalized;
    }
    const stat = fs.statSync(filename);
    await this.client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: normalized,
      Body: fs.createReadStream(filename),
      ContentLength: stat.size,
      ContentType: contentType || 'application/octet-stream',
      ServerSideEncryption: String(process.env.ARTIFACT_S3_SSE || '').trim() || undefined,
    }));
    return normalized;
  }

  async materialize(key, targetPath) {
    const normalized = safeKey(key);
    if (!this.remote) return this.localPath(normalized);
    const response = await this.client.send(new GetObjectCommand({ Bucket: this.bucket, Key: normalized }));
    fs.mkdirSync(path.dirname(targetPath), { recursive: true });
    const bytes = Buffer.from(await response.Body.transformToByteArray());
    fs.writeFileSync(targetPath, bytes);
    return targetPath;
  }

  async sendDownload(res, key, filename, contentType = 'application/octet-stream') {
    const normalized = safeKey(key);
    if (!this.remote) return res.download(this.localPath(normalized), filename);
    const response = await this.client.send(new GetObjectCommand({ Bucket: this.bucket, Key: normalized }));
    res.setHeader('Content-Type', response.ContentType || contentType);
    res.setHeader('Content-Disposition', contentDisposition(filename));
    if (response.ContentLength !== undefined) res.setHeader('Content-Length', response.ContentLength);
    response.Body.on('error', (error) => res.destroy(error));
    response.Body.pipe(res);
    return undefined;
  }

  async remove(key) {
    if (!key) return;
    const normalized = safeKey(key);
    if (!this.remote) {
      fs.rmSync(this.localPath(normalized), { force: true });
      return;
    }
    await this.client.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: normalized }));
  }
}

function createArtifactStore(options) {
  return new ArtifactStore(options);
}

module.exports = { createArtifactStore };
