const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const {
  HeadObjectCommand,
  PutBucketCorsCommand,
  PutObjectCommand,
  S3Client,
} = require('@aws-sdk/client-s3');

const SOURCE_DIR = path.resolve(__dirname, '..', 'public', 'samples');
const DRY_RUN = process.argv.includes('--dry-run');
const RELEASE = String(process.env.INSTRUMENT_ASSET_RELEASE || 'v1').trim().replace(/^\/+|\/+$/g, '');
const BUCKET = String(process.env.INSTRUMENT_R2_BUCKET || '').trim();
const ACCOUNT_ID = String(process.env.CLOUDFLARE_ACCOUNT_ID || '').trim();
const ENDPOINT = String(
  process.env.INSTRUMENT_R2_ENDPOINT
    || (ACCOUNT_ID ? `https://${ACCOUNT_ID}.r2.cloudflarestorage.com` : ''),
).trim().replace(/\/+$/, '');
const ACCESS_KEY_ID = String(process.env.INSTRUMENT_R2_ACCESS_KEY_ID || '').trim();
const SECRET_ACCESS_KEY = String(process.env.INSTRUMENT_R2_SECRET_ACCESS_KEY || '').trim();
const CONCURRENCY = Math.max(1, Math.min(16, Number(process.env.INSTRUMENT_SYNC_CONCURRENCY || 6)));
const CONFIGURE_CORS = !/^(0|false|no)$/i.test(
  String(process.env.INSTRUMENT_CONFIGURE_CORS || 'true').trim(),
);

if (!/^[a-z0-9][a-z0-9._-]{0,62}$/i.test(RELEASE)) {
  throw new Error('INSTRUMENT_ASSET_RELEASE must be a simple version such as v1 or 2026-08-25.');
}

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolutePath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(absolutePath) : [absolutePath];
  });
}

function contentType(filename) {
  switch (path.extname(filename).toLowerCase()) {
    case '.wav': return 'audio/wav';
    case '.json': return 'application/json; charset=utf-8';
    case '.md': return 'text/markdown; charset=utf-8';
    case '.txt': return 'text/plain; charset=utf-8';
    default: return 'application/octet-stream';
  }
}

function cacheControl(filename) {
  return path.extname(filename).toLowerCase() === '.wav'
    ? 'public, max-age=31536000, immutable'
    : 'public, max-age=300, must-revalidate';
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function objectKey(filename) {
  const relative = path.relative(path.resolve(SOURCE_DIR, '..'), filename).split(path.sep).join('/');
  return `${RELEASE}/${relative}`;
}

async function mapConcurrent(items, worker) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(CONCURRENCY, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      await worker(items[index], index);
    }
  });
  await Promise.all(runners);
}

async function configureCors(client) {
  const origins = String(
    process.env.INSTRUMENT_ALLOWED_ORIGINS
      || 'https://polymathmusician67.com,http://localhost:5173,http://127.0.0.1:5173',
  )
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  await client.send(new PutBucketCorsCommand({
    Bucket: BUCKET,
    CORSConfiguration: {
      CORSRules: [{
        AllowedHeaders: ['*'],
        AllowedMethods: ['GET', 'HEAD'],
        AllowedOrigins: origins,
        ExposeHeaders: ['ETag'],
        MaxAgeSeconds: 86400,
      }],
    },
  }));
}

async function main() {
  const files = walk(SOURCE_DIR);
  const totalBytes = files.reduce((sum, filename) => sum + fs.statSync(filename).size, 0);
  console.log(`Instrument release ${RELEASE}: ${files.length} files, ${(totalBytes / 1024 / 1024).toFixed(1)} MiB.`);

  if (DRY_RUN) {
    console.log('Dry run complete; no network calls or uploads were made.');
    return;
  }

  if (!BUCKET || !ENDPOINT || !ACCESS_KEY_ID || !SECRET_ACCESS_KEY) {
    throw new Error(
      'R2 configuration is incomplete. Set INSTRUMENT_R2_BUCKET, CLOUDFLARE_ACCOUNT_ID '
      + '(or INSTRUMENT_R2_ENDPOINT), INSTRUMENT_R2_ACCESS_KEY_ID, and INSTRUMENT_R2_SECRET_ACCESS_KEY.',
    );
  }

  const client = new S3Client({
    region: 'auto',
    endpoint: ENDPOINT,
    credentials: {
      accessKeyId: ACCESS_KEY_ID,
      secretAccessKey: SECRET_ACCESS_KEY,
    },
  });

  if (CONFIGURE_CORS) {
    await configureCors(client);
  } else {
    console.log('CORS configuration skipped; expecting bucket CORS to be managed separately.');
  }
  let uploaded = 0;
  let unchanged = 0;

  await mapConcurrent(files, async (filename) => {
    const body = fs.readFileSync(filename);
    const digest = sha256(body);
    const Key = objectKey(filename);

    try {
      const current = await client.send(new HeadObjectCommand({ Bucket: BUCKET, Key }));
      if (String(current.Metadata?.sha256 || '') === digest) {
        unchanged += 1;
        return;
      }
    } catch (error) {
      const status = Number(error?.$metadata?.httpStatusCode || 0);
      if (status !== 404 && error?.name !== 'NotFound' && error?.name !== 'NoSuchKey') throw error;
    }

    await client.send(new PutObjectCommand({
      Bucket: BUCKET,
      Key,
      Body: body,
      ContentType: contentType(filename),
      CacheControl: cacheControl(filename),
      Metadata: { sha256: digest },
    }));
    uploaded += 1;
    if (uploaded % 25 === 0) console.log(`Uploaded ${uploaded} changed files...`);
  });

  console.log(`R2 sync complete: ${uploaded} uploaded, ${unchanged} unchanged.`);
}

main().catch((error) => {
  console.error(`R2 instrument sync failed: ${error.message}`);
  process.exitCode = 1;
});
