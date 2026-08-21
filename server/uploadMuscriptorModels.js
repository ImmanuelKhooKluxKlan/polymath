const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {
  CompleteMultipartUploadCommand,
  CopyObjectCommand,
  CreateMultipartUploadCommand,
  HeadObjectCommand,
  ListMultipartUploadsCommand,
  ListPartsCommand,
  PutObjectCommand,
  S3Client,
  UploadPartCommand,
} = require('@aws-sdk/client-s3');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '.env'), quiet: true });

const WEIGHTS_FILE = 'model.safetensors';
const CONFIG_FILE = 'config.json';
const PART_SIZE = 8 * 1024 * 1024;
const ORIGINAL_PREFIX = 'models/original';
const testVersion = String(process.env.MUSCRIPTOR_TEST_VERSION || 'v001').trim();
const TEST_PREFIX = `models/muscriptor-tester/${testVersion}`;

function clean(value) {
  return String(value || '').trim();
}

function required(name) {
  const value = clean(process.env[name]);
  if (!value) throw new Error(`${name} is missing from server/.env`);
  return value;
}

function findLocalModelDirectory() {
  const override = clean(process.env.MUSCRIPTOR_LOCAL_MODEL_DIR);
  if (override) return path.resolve(override);

  const snapshots = path.join(
    os.homedir(),
    '.cache',
    'huggingface',
    'hub',
    'models--MuScriptor--muscriptor-large',
    'snapshots',
  );
  const candidates = fs.readdirSync(snapshots, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(snapshots, entry.name))
    .filter((directory) => (
      fs.existsSync(path.join(directory, WEIGHTS_FILE))
      && fs.existsSync(path.join(directory, CONFIG_FILE))
    ));
  if (!candidates.length) {
    throw new Error('MuScriptor Large was not found in the local Hugging Face cache.');
  }
  return candidates[0];
}

function formatBytes(bytes) {
  return `${(Number(bytes) / (1024 ** 3)).toFixed(2)} GiB`;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function remoteMetadata(client, bucket, key) {
  try {
    return await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
  } catch (error) {
    const status = error?.$metadata?.httpStatusCode;
    if (status === 404 || error?.name === 'NotFound' || error?.name === 'NoSuchKey') {
      return null;
    }
    throw error;
  }
}

async function verifySize(client, bucket, key, expectedSize) {
  const metadata = await remoteMetadata(client, bucket, key);
  if (!metadata || Number(metadata.ContentLength) !== expectedSize) {
    throw new Error(`Remote verification failed for ${key}`);
  }
}

async function findMultipartUpload(client, bucket, key) {
  const response = await client.send(new ListMultipartUploadsCommand({
    Bucket: bucket,
    Prefix: key,
  }));
  return (response.Uploads || [])
    .filter((upload) => upload.Key === key && upload.UploadId)
    .sort((left, right) => new Date(right.Initiated || 0) - new Date(left.Initiated || 0))[0] || null;
}

async function listUploadedParts(client, bucket, key, uploadId) {
  const uploaded = new Map();
  let marker;
  do {
    const response = await client.send(new ListPartsCommand({
      Bucket: bucket,
      Key: key,
      UploadId: uploadId,
      PartNumberMarker: marker,
    }));
    for (const part of response.Parts || []) {
      uploaded.set(Number(part.PartNumber), part);
    }
    marker = response.IsTruncated ? response.NextPartNumberMarker : undefined;
  } while (marker);
  return uploaded;
}

async function multipartUploadFile(client, bucket, localPath, key, expectedSize) {
  const existingUpload = await findMultipartUpload(client, bucket, key);
  const uploadId = existingUpload?.UploadId || (await client.send(
    new CreateMultipartUploadCommand({
      Bucket: bucket,
      Key: key,
      ContentType: 'application/octet-stream',
    }),
  )).UploadId;
  const uploadedParts = existingUpload
    ? await listUploadedParts(client, bucket, key, uploadId)
    : new Map();
  const parts = [];
  const partCount = Math.ceil(expectedSize / PART_SIZE);
  let lastPercent = -1;
  console.log(existingUpload
    ? `  resuming multipart; ${uploadedParts.size}/${partCount} parts already stored`
    : `  multipart created; ${partCount} parts`);

  try {
    for (let index = 0; index < partCount; index += 1) {
      const partNumber = index + 1;
      const start = index * PART_SIZE;
      const length = Math.min(PART_SIZE, expectedSize - start);
      const existingPart = uploadedParts.get(partNumber);
      if (existingPart && Number(existingPart.Size) === length && existingPart.ETag) {
        parts.push({ PartNumber: partNumber, ETag: existingPart.ETag });
        continue;
      }
      let uploaded;
      for (let attempt = 1; attempt <= 5; attempt += 1) {
        const body = fs.createReadStream(localPath, {
          start,
          end: start + length - 1,
        });
        try {
          uploaded = await client.send(new UploadPartCommand({
            Bucket: bucket,
            Key: key,
            UploadId: uploadId,
            PartNumber: partNumber,
            Body: body,
            ContentLength: length,
          }));
          break;
        } catch (error) {
          body.destroy();
          if (attempt === 5) {
            error.message = `part ${partNumber}/${partCount}: ${error.message}`;
            throw error;
          }
          console.warn(`  retrying part ${partNumber} (attempt ${attempt + 1}/5)`);
          await sleep(1_000 * (2 ** (attempt - 1)));
        }
      }
      parts.push({ PartNumber: partNumber, ETag: uploaded.ETag });
      const percent = Math.floor((partNumber / partCount) * 100);
      if (percent > lastPercent || partNumber === partCount) {
        console.log(`  ${percent}% (part ${partNumber}/${partCount})`);
        lastPercent = percent;
      }
    }

    await client.send(new CompleteMultipartUploadCommand({
      Bucket: bucket,
      Key: key,
      UploadId: uploadId,
      MultipartUpload: { Parts: parts },
    }));
  } catch (error) {
    console.warn(`  multipart session preserved for resume: ${uploadId}`);
    throw error;
  }
}

async function uploadFile(client, bucket, localPath, key) {
  const expectedSize = fs.statSync(localPath).size;
  const existing = await remoteMetadata(client, bucket, key);
  if (existing) {
    if (Number(existing.ContentLength) === expectedSize) {
      console.log(`Already verified: ${key} (${formatBytes(expectedSize)})`);
      return;
    }
    throw new Error(`Refusing to overwrite ${key}; its remote size is different.`);
  }

  console.log(`Uploading ${key} (${formatBytes(expectedSize)})`);
  if (expectedSize > 500 * 1024 * 1024) {
    await multipartUploadFile(client, bucket, localPath, key, expectedSize);
  } else {
    await client.send(new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: fs.createReadStream(localPath),
      ContentLength: expectedSize,
      ContentType: key.endsWith('.json') ? 'application/json' : 'application/octet-stream',
    }));
  }
  await verifySize(client, bucket, key, expectedSize);
}

function encodedCopySource(bucket, key) {
  return encodeURIComponent(`${bucket}/${key}`).replace(/%2F/g, '/');
}

async function copyOrUpload(client, bucket, sourceKey, targetKey, localPath) {
  const expectedSize = fs.statSync(localPath).size;
  const existing = await remoteMetadata(client, bucket, targetKey);
  if (existing) {
    if (Number(existing.ContentLength) === expectedSize) {
      console.log(`Already verified: ${targetKey} (${formatBytes(expectedSize)})`);
      return;
    }
    throw new Error(`Refusing to overwrite ${targetKey}; its remote size is different.`);
  }

  console.log(`Copying inside RunPod: ${sourceKey} -> ${targetKey}`);
  try {
    await client.send(new CopyObjectCommand({
      Bucket: bucket,
      CopySource: encodedCopySource(bucket, sourceKey),
      Key: targetKey,
    }));
    await verifySize(client, bucket, targetKey, expectedSize);
  } catch (error) {
    console.warn(`Server-side copy failed (${error.message}); using multipart upload fallback.`);
    await uploadFile(client, bucket, localPath, targetKey);
  }
}

async function main() {
  if (!/^v\d{3,}$/.test(testVersion)) {
    throw new Error('MUSCRIPTOR_TEST_VERSION must look like v001, v002, and so on.');
  }

  const bucket = required('RUNPOD_NETWORK_VOLUME_ID');
  const endpoint = required('RUNPOD_S3_ENDPOINT').replace(/\/+$/, '');
  const region = required('RUNPOD_S3_REGION');
  const accessKeyId = required('RUNPOD_S3_ACCESS_KEY_ID');
  const secretAccessKey = required('RUNPOD_S3_SECRET_ACCESS_KEY');
  const modelDirectory = findLocalModelDirectory();
  const weightsPath = path.join(modelDirectory, WEIGHTS_FILE);
  const configPath = path.join(modelDirectory, CONFIG_FILE);

  const client = new S3Client({
    region,
    endpoint,
    forcePathStyle: true,
    credentials: { accessKeyId, secretAccessKey },
    maxAttempts: 10,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
  });
  client.middlewareStack.addRelativeTo(
    (next) => async (args) => {
      const output = await next(args);
      const headers = output.response?.headers || {};
      for (const name of ['date', 'last-modified']) {
        if (typeof headers[name] === 'string') {
          headers[name] = headers[name].replace(/ UTC$/, ' GMT');
        }
      }
      return output;
    },
    {
      relation: 'after',
      toMiddleware: 'deserializerMiddleware',
      name: 'normalizeRunpodDates',
    },
  );

  console.log(`Local model: ${modelDirectory}`);
  await uploadFile(client, bucket, weightsPath, `${ORIGINAL_PREFIX}/${WEIGHTS_FILE}`);
  await uploadFile(client, bucket, configPath, `${ORIGINAL_PREFIX}/${CONFIG_FILE}`);
  await copyOrUpload(
    client,
    bucket,
    `${ORIGINAL_PREFIX}/${WEIGHTS_FILE}`,
    `${TEST_PREFIX}/${WEIGHTS_FILE}`,
    weightsPath,
  );
  await copyOrUpload(
    client,
    bucket,
    `${ORIGINAL_PREFIX}/${CONFIG_FILE}`,
    `${TEST_PREFIX}/${CONFIG_FILE}`,
    configPath,
  );

  console.log('MuScriptor model bootstrap complete.');
  console.log(`Original: /runpod-volume/${ORIGINAL_PREFIX}/${WEIGHTS_FILE}`);
  console.log(`Tester:   /runpod-volume/${TEST_PREFIX}/${WEIGHTS_FILE}`);
}

main().catch((error) => {
  console.error(`MuScriptor upload failed: ${error.message}`);
  process.exitCode = 1;
});
