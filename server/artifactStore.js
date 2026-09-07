const fs = require('fs');
const path = require('path');
const {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
  CopyObjectCommand,
  ListObjectsV2Command,
} = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

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
      requestChecksumCalculation: 'WHEN_REQUIRED',
      responseChecksumValidation: 'WHEN_REQUIRED',
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

  async createPresignedPut(key, contentType, expiresInSeconds = 900) {
    if (!this.remote) return null;
    const normalized = safeKey(key);
    return getSignedUrl(this.client, new PutObjectCommand({
      Bucket: this.bucket,
      Key: normalized,
      ContentType: contentType || 'application/octet-stream',
      ServerSideEncryption: String(process.env.ARTIFACT_S3_SSE || '').trim() || undefined,
    }), {
      expiresIn: Math.max(60, Math.min(3600, Number(expiresInSeconds) || 900)),
    });
  }

  async stat(key) {
    const normalized = safeKey(key);
    if (!this.remote) {
      const details = fs.statSync(this.localPath(normalized));
      return { size: details.size, contentType: 'application/octet-stream', etag: '' };
    }
    const response = await this.client.send(new HeadObjectCommand({
      Bucket: this.bucket,
      Key: normalized,
    }));
    return {
      size: Number(response.ContentLength || 0),
      contentType: String(response.ContentType || 'application/octet-stream'),
      etag: String(response.ETag || '').replace(/^"|"$/g, ''),
    };
  }

  async list(prefix = '') {
    const normalizedPrefix = prefix ? safeKey(prefix).replace(/\/?$/, '/') : '';
    if (!this.remote) {
      const base = normalizedPrefix ? this.localPath(normalizedPrefix) : this.localRoot;
      if (!fs.existsSync(base)) return [];
      const keys = [];
      const visit = (directory) => {
        for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
          const filename = path.join(directory, entry.name);
          if (entry.isDirectory()) visit(filename);
          else if (entry.isFile()) keys.push(path.relative(this.localRoot, filename).replace(/\\/g, '/'));
        }
      };
      visit(base);
      return keys.sort();
    }
    const keys = [];
    let continuationToken;
    do {
      const response = await this.client.send(new ListObjectsV2Command({
        Bucket: this.bucket,
        Prefix: normalizedPrefix,
        ContinuationToken: continuationToken,
      }));
      for (const item of response.Contents || []) {
        if (item.Key) keys.push(String(item.Key));
      }
      continuationToken = response.IsTruncated ? response.NextContinuationToken : undefined;
    } while (continuationToken);
    return keys.sort();
  }

  async promote(sourceKey, targetKey) {
    const source = safeKey(sourceKey);
    const target = safeKey(targetKey);
    if (!this.remote) {
      const sourcePath = this.localPath(source);
      const targetPath = this.localPath(target);
      fs.mkdirSync(path.dirname(targetPath), { recursive: true });
      fs.renameSync(sourcePath, targetPath);
      return target;
    }
    const encodedSource = source.split('/').map(encodeURIComponent).join('/');
    await this.client.send(new CopyObjectCommand({
      Bucket: this.bucket,
      Key: target,
      CopySource: `${encodeURIComponent(this.bucket)}/${encodedSource}`,
      MetadataDirective: 'COPY',
      ServerSideEncryption: String(process.env.ARTIFACT_S3_SSE || '').trim() || undefined,
    }));
    await this.client.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: source }));
    return target;
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

  async getBuffer(key) {
    const normalized = safeKey(key);
    if (!this.remote) return fs.readFileSync(this.localPath(normalized));
    const response = await this.client.send(new GetObjectCommand({ Bucket: this.bucket, Key: normalized }));
    return Buffer.from(await response.Body.transformToByteArray());
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
