const fs = require('fs');
const path = require('path');
const {
  S3Client,
  HeadObjectCommand,
  PutObjectCommand,
} = require('@aws-sdk/client-s3');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const dryRun = process.argv.includes('--dry-run');
const uploadRoot = path.resolve(
  process.env.POLYMATH_DATA_DIR || path.join(__dirname, 'data'),
  'uploads',
);
const bucket = String(process.env.ARTIFACT_S3_BUCKET || '').trim();

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const filename = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(filename) : [filename];
  });
}

function contentType(filename) {
  const extension = path.extname(filename).toLowerCase();
  if (extension === '.json') return 'application/json';
  if (extension === '.pdf') return 'application/pdf';
  if (extension === '.wav') return 'audio/wav';
  if (extension === '.mp3') return 'audio/mpeg';
  return 'application/octet-stream';
}

async function main() {
  const files = walk(uploadRoot);
  const bytes = files.reduce((total, file) => total + fs.statSync(file).size, 0);
  console.log(`Legacy artifact plan: ${files.length} files, ${(bytes / 1024 / 1024).toFixed(2)} MiB.`);
  if (dryRun) return;
  if (!bucket) throw new Error('ARTIFACT_S3_BUCKET is required.');

  const endpoint = String(process.env.ARTIFACT_S3_ENDPOINT || '').trim();
  const accessKeyId = String(process.env.ARTIFACT_S3_ACCESS_KEY_ID || '').trim();
  const secretAccessKey = String(process.env.ARTIFACT_S3_SECRET_ACCESS_KEY || '').trim();
  const client = new S3Client({
    region: process.env.ARTIFACT_S3_REGION || 'us-east-2',
    endpoint: endpoint || undefined,
    forcePathStyle: Boolean(endpoint),
    credentials: accessKeyId && secretAccessKey ? { accessKeyId, secretAccessKey } : undefined,
  });

  let uploaded = 0;
  let skipped = 0;
  for (const filename of files) {
    const key = path.relative(uploadRoot, filename).replace(/\\/g, '/');
    const size = fs.statSync(filename).size;
    try {
      const existing = await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
      if (Number(existing.ContentLength) === size) {
        skipped += 1;
        continue;
      }
    } catch (error) {
      if (![403, 404].includes(error?.$metadata?.httpStatusCode)) throw error;
    }
    await client.send(new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: fs.createReadStream(filename),
      ContentLength: size,
      ContentType: contentType(filename),
      ServerSideEncryption: String(process.env.ARTIFACT_S3_SSE || '').trim() || undefined,
    }));
    uploaded += 1;
    console.log(`Uploaded ${key}`);
  }
  console.log(`Legacy artifact migration complete: ${uploaded} uploaded, ${skipped} unchanged.`);
}

main().catch((error) => {
  console.error(error.message || error);
  process.exitCode = 1;
});
