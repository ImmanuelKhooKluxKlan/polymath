const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { spawn } = require('node:child_process');
const express = require('express');
const multer = require('multer');
const bundledFfmpegPath = require('ffmpeg-static');
const { analyzeTranscription } = require('./modelLabAnalysis');
const { createRunpodServerlessClient } = require('./runpodServerless');

const MEDIA_EXTENSIONS = new Set([
  '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
  '.mp4', '.mov', '.webm', '.mkv', '.avi', '.mpeg', '.mpg',
]);
const NOTE_EXTENSIONS = new Set(['.mid', '.midi', '.json']);
const ALIGNMENT_MODULE_URL = pathToFileURL(path.join(
  __dirname,
  '..',
  'scripts',
  'alignment',
  'noteCoordinateAligner.mjs',
)).href;
const REMOTE_HISTORY_PREFIX = 'private/model-lab-history';

function isLoopback(request) {
  const addresses = [request.ip, request.socket?.remoteAddress]
    .map((value) => String(value || '').toLowerCase());
  return addresses.some((value) => (
    value === '127.0.0.1'
    || value === '::1'
    || value === '::ffff:127.0.0.1'
  ));
}

function runFfmpeg(ffmpegPath, source, destination, maximumSeconds) {
  return new Promise((resolve, reject) => {
    const child = spawn(ffmpegPath, [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-i', source,
      '-vn', '-ac', '1', '-ar', '16000',
      '-t', String(maximumSeconds),
      destination,
    ], {
      windowsHide: true,
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk.toString()}`.slice(-8000);
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr.trim() || `FFmpeg exited with code ${code}.`));
    });
  });
}

function readWavDurationSeconds(filename) {
  const buffer = fs.readFileSync(filename);
  if (buffer.length < 44 || buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Prepared audio is not a readable RIFF/WAVE file.');
  }
  let byteRate = 0;
  let dataBytes = 0;
  for (let offset = 12; offset + 8 <= buffer.length;) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkBytes = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;
    if (chunkId === 'fmt ' && chunkBytes >= 16 && chunkStart + 12 <= buffer.length) {
      byteRate = buffer.readUInt32LE(chunkStart + 8);
    } else if (chunkId === 'data') {
      dataBytes += Math.min(chunkBytes, Math.max(0, buffer.length - chunkStart));
    }
    offset = chunkStart + chunkBytes + (chunkBytes % 2);
  }
  if (!byteRate || !dataBytes) throw new Error('Prepared audio is missing WAV format or data chunks.');
  return dataBytes / byteRate;
}

function sha256File(filename) {
  return crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
}

function writeJsonAtomic(filename, value) {
  const temporary = `${filename}.${crypto.randomUUID()}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filename);
}

function copyFileAtomic(source, destination) {
  const temporary = `${destination}.${crypto.randomUUID()}.tmp`;
  fs.copyFileSync(source, temporary);
  fs.renameSync(temporary, destination);
}

function readJsonFile(filename) {
  return JSON.parse(fs.readFileSync(filename, 'utf8'));
}

function safeArchiveDirectory(archiveRoot, recordId) {
  const id = String(recordId || '').trim();
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(id)) return null;
  const root = path.resolve(archiveRoot);
  const directory = path.resolve(root, id);
  return directory.startsWith(`${root}${path.sep}`) ? directory : null;
}

function loadAlignmentArchive(archiveRoot, recordId) {
  const directory = safeArchiveDirectory(archiveRoot, recordId);
  if (!directory) return null;
  const manifestPath = path.join(directory, 'manifest.json');
  const analysisPath = path.join(directory, 'analysis.json');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(analysisPath)) return null;
  try {
    const manifest = readJsonFile(manifestPath);
    const analysis = readJsonFile(analysisPath);
    if (manifest?.schema !== 'polymath-supervision-archive-v1' || manifest.id !== recordId) return null;
    for (const artifact of [manifest.reference, manifest.observed]) {
      const filename = path.join(directory, path.basename(String(artifact?.archivedFilename || '')));
      if (!artifact?.sha256 || !fs.existsSync(filename) || sha256File(filename) !== artifact.sha256) return null;
    }
    return { directory, manifest, analysis };
  } catch {
    return null;
  }
}

function alignmentArchiveSummary(archive) {
  const { manifest, analysis } = archive;
  return {
    id: manifest.id,
    kind: 'supervision',
    title: manifest.title,
    createdAt: manifest.createdAt,
    updatedAt: manifest.updatedAt,
    referenceFilename: manifest.reference?.originalFilename || '',
    observedFilename: manifest.observed?.originalFilename || '',
    referenceNotes: manifest.reference?.noteCount || 0,
    observedNotes: manifest.observed?.noteCount || 0,
    matchedPercent: analysis.metrics?.matchedReferencePercent ?? null,
    exactPitchPercent: analysis.metrics?.exactPitchPercent ?? null,
    timingResidualMs: analysis.metrics?.medianTimingResidualMs ?? null,
    trainingEligiblePercent: analysis.metrics?.trainingEligiblePercent ?? null,
    readyForTraining: Boolean(analysis.supervisionPackage?.review?.readyForTraining),
    verdict: analysis.metrics?.verdict || 'unknown',
  };
}

function listAlignmentArchives(archiveRoot, limit = 100) {
  if (!fs.existsSync(archiveRoot)) return [];
  const records = [];
  for (const entry of fs.readdirSync(archiveRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const archive = loadAlignmentArchive(archiveRoot, entry.name);
    if (archive) records.push(alignmentArchiveSummary(archive));
  }
  return records
    .sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)))
    .slice(0, Math.max(1, Math.min(500, Number(limit) || 100)));
}

function archiveRawTestSnapshot({ archiveRoot, job }) {
  const directory = safeArchiveDirectory(archiveRoot, job.id);
  if (!directory) throw new Error('Unsafe Model Lab job archive identifier.');
  fs.mkdirSync(directory, { recursive: true });
  const rawPath = path.join(directory, 'raw-model-output.json');
  const analysisPath = path.join(directory, 'analysis.json');
  writeJsonAtomic(rawPath, job.result.raw);
  writeJsonAtomic(analysisPath, job.result.analysis);
  const manifest = {
    schema: 'polymath-model-test-archive-v1',
    id: job.id,
    kind: 'raw-test',
    title: job.title,
    originalFilename: job.filename,
    checkpoint: job.checkpoint || 'muscriptor-tester/v001',
    createdAt: job.createdAt,
    completedAt: job.completedAt,
    sourceDurationSeconds: job.sourceDurationSeconds,
    noteCount: job.result.analysis?.headline?.validNotes || 0,
    instrumentCount: job.result.analysis?.headline?.detectedInstrumentGroups || 0,
    rapidRepeats75ms: job.result.analysis?.headline?.rapidRepeats75ms || 0,
    rawOutput: { filename: path.basename(rawPath), sha256: sha256File(rawPath) },
    analysis: { filename: path.basename(analysisPath), sha256: sha256File(analysisPath) },
    sourceMediaRetained: false,
  };
  writeJsonAtomic(path.join(directory, 'manifest.json'), manifest);
  return { directory, manifest };
}

function loadRawTestArchive(archiveRoot, recordId) {
  const directory = safeArchiveDirectory(archiveRoot, recordId);
  if (!directory) return null;
  const manifestPath = path.join(directory, 'manifest.json');
  if (!fs.existsSync(manifestPath)) return null;
  try {
    const manifest = readJsonFile(manifestPath);
    if (manifest?.schema !== 'polymath-model-test-archive-v1' || manifest.id !== recordId) return null;
    const rawPath = path.join(directory, path.basename(String(manifest.rawOutput?.filename || '')));
    const analysisPath = path.join(directory, path.basename(String(manifest.analysis?.filename || '')));
    if (!manifest.rawOutput?.sha256 || sha256File(rawPath) !== manifest.rawOutput.sha256) return null;
    if (!manifest.analysis?.sha256 || sha256File(analysisPath) !== manifest.analysis.sha256) return null;
    const raw = readJsonFile(rawPath);
    const analysis = readJsonFile(analysisPath);
    return { directory, manifest, raw, analysis };
  } catch {
    return null;
  }
}

function listRawTestArchives(archiveRoot, limit = 100) {
  if (!fs.existsSync(archiveRoot)) return [];
  const records = [];
  for (const entry of fs.readdirSync(archiveRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const archive = loadRawTestArchive(archiveRoot, entry.name);
    if (!archive) continue;
    records.push({
      ...archive.manifest,
      rawOutput: undefined,
      analysis: undefined,
    });
  }
  return records
    .sort((left, right) => String(right.completedAt).localeCompare(String(left.completedAt)))
    .slice(0, Math.max(1, Math.min(500, Number(limit) || 100)));
}

async function mirrorArchiveDirectory(artifactStore, remotePrefix, directory) {
  if (!artifactStore?.remote) return false;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const contentType = entry.name.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    await artifactStore.putFile(`${remotePrefix}/${entry.name}`, path.join(directory, entry.name), contentType);
  }
  return true;
}

async function readRemoteJson(artifactStore, key, cacheRoot) {
  const target = path.join(cacheRoot, ...key.split('/'));
  await artifactStore.materialize(key, target);
  return readJsonFile(target);
}

async function listRemoteHistory(artifactStore, cacheRoot, limit = 100) {
  if (!artifactStore?.remote) return { rawTests: [], alignments: [] };
  const keys = await artifactStore.list(REMOTE_HISTORY_PREFIX);
  const manifestKeys = keys.filter((key) => key.endsWith('/manifest.json'));
  const rawTests = [];
  const alignments = [];
  for (const key of manifestKeys) {
    try {
      const manifest = await readRemoteJson(artifactStore, key, cacheRoot);
      if (manifest.schema === 'polymath-model-test-archive-v1') {
        rawTests.push({ ...manifest, rawOutput: undefined, analysis: undefined });
      } else if (manifest.schema === 'polymath-supervision-archive-v1') {
        const analysisKey = `${key.slice(0, -'manifest.json'.length)}analysis.json`;
        const analysis = await readRemoteJson(artifactStore, analysisKey, cacheRoot);
        alignments.push(alignmentArchiveSummary({ manifest, analysis }));
      }
    } catch {
      // A partial/corrupt remote record is omitted instead of poisoning history.
    }
  }
  const safeLimit = Math.max(1, Math.min(500, Number(limit) || 100));
  return {
    rawTests: rawTests
      .sort((left, right) => String(right.completedAt).localeCompare(String(left.completedAt)))
      .slice(0, safeLimit),
    alignments: alignments
      .sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)))
      .slice(0, safeLimit),
  };
}

async function restoreRemoteArchive(artifactStore, cacheRoot, archiveRoot, kind, recordId) {
  if (!artifactStore?.remote || !safeArchiveDirectory(archiveRoot, recordId)) return false;
  const remoteDirectory = `${REMOTE_HISTORY_PREFIX}/${kind}/${recordId}`;
  let manifest;
  try {
    manifest = await readRemoteJson(artifactStore, `${remoteDirectory}/manifest.json`, cacheRoot);
  } catch {
    return false;
  }
  const expectedSchema = kind === 'raw-tests'
    ? 'polymath-model-test-archive-v1'
    : 'polymath-supervision-archive-v1';
  if (manifest?.schema !== expectedSchema || manifest.id !== recordId) return false;
  const destination = safeArchiveDirectory(archiveRoot, recordId);
  fs.mkdirSync(destination, { recursive: true });
  const filenames = kind === 'raw-tests'
    ? ['manifest.json', manifest.rawOutput?.filename, manifest.analysis?.filename]
    : ['manifest.json', manifest.analysisFilename, manifest.reference?.archivedFilename, manifest.observed?.archivedFilename];
  for (const supplied of filenames) {
    const filename = path.basename(String(supplied || ''));
    if (!filename) return false;
    await artifactStore.materialize(`${remoteDirectory}/${filename}`, path.join(destination, filename));
  }
  return true;
}

function archiveAlignmentSnapshot({ archiveRoot, record, referenceFile, observedFile, analysis }) {
  const directory = path.join(path.resolve(archiveRoot), record.id);
  fs.mkdirSync(directory, { recursive: true });

  const referenceExtension = path.extname(referenceFile?.originalname || record.referenceFilename || '').toLowerCase();
  const observedExtension = path.extname(observedFile?.originalname || record.observedFilename || '').toLowerCase();
  const referenceArchive = path.join(directory, `desired-reference${NOTE_EXTENSIONS.has(referenceExtension) ? referenceExtension : '.json'}`);
  const observedArchive = path.join(directory, `model-output${NOTE_EXTENSIONS.has(observedExtension) ? observedExtension : '.json'}`);

  if (referenceFile?.path) copyFileAtomic(referenceFile.path, referenceArchive);
  else if (!fs.existsSync(referenceArchive)) writeJsonAtomic(referenceArchive, { notes: record.referenceNotes });
  if (observedFile?.path) copyFileAtomic(observedFile.path, observedArchive);
  else if (!fs.existsSync(observedArchive)) writeJsonAtomic(observedArchive, { notes: record.observedNotes });

  const manifest = {
    schema: 'polymath-supervision-archive-v1',
    id: record.id,
    title: record.title,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    privateLocalArchive: true,
    reference: {
      originalFilename: record.referenceFilename,
      archivedFilename: path.basename(referenceArchive),
      noteCount: record.referenceNotes.length,
      sha256: sha256File(referenceArchive),
    },
    observed: {
      originalFilename: record.observedFilename,
      archivedFilename: path.basename(observedArchive),
      noteCount: record.observedNotes.length,
      sha256: sha256File(observedArchive),
    },
    analysisFilename: 'analysis.json',
  };
  writeJsonAtomic(path.join(directory, 'analysis.json'), analysis);
  writeJsonAtomic(path.join(directory, 'manifest.json'), manifest);
  writeJsonAtomic(path.join(path.resolve(archiveRoot), 'latest.json'), {
    id: record.id,
    directory: record.id,
    updatedAt: record.updatedAt,
  });
  return { directory, manifest };
}

function publicJob(job) {
  return {
    id: job.id,
    title: job.title,
    filename: job.filename,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    createdAt: job.createdAt,
    completedAt: job.completedAt || null,
    sourceDurationSeconds: Number.isFinite(job.sourceDurationSeconds)
      ? Number(job.sourceDurationSeconds.toFixed(4))
      : null,
    error: job.error || '',
    archive: job.archive || { saved: false },
    result: job.status === 'completed' ? job.result : undefined,
  };
}

function createModelLab(environment = process.env, options = {}) {
  const localOnly = String(environment.MODEL_LAB_LOCAL_ONLY || 'false').trim().toLowerCase() === 'true';
  const production = String(environment.NODE_ENV || '').toLowerCase() === 'production';
  const dataRoot = path.resolve(
    options.dataRoot
      || String(environment.MODEL_LAB_DATA_ROOT || '').trim()
      || (production
        ? path.join(__dirname, 'data', 'model-lab')
        : path.join(os.tmpdir(), 'polymath-model-lab-runtime')),
  );
  const ffmpegPath = String(environment.FFMPEG_PATH || '').trim() || bundledFfmpegPath;
  const maximumSeconds = Math.max(10, Math.min(3600, Number(environment.MODEL_LAB_MAX_SECONDS) || 600));
  fs.mkdirSync(dataRoot, { recursive: true });
  const artifactStore = options.artifactStore || null;
  const remoteHistoryCacheRoot = path.join(dataRoot, 'remote-history-cache');

  const runpod = createRunpodServerlessClient({
    endpointId: environment.RUNPOD_SERVERLESS_ENDPOINT_ID || environment.RUNPOD_ENDPOINT_ID,
    apiKey: environment.RUNPOD_API_KEY,
    volumeId: environment.RUNPOD_NETWORK_VOLUME_ID,
    region: environment.RUNPOD_S3_REGION,
    s3Endpoint: environment.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: environment.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: environment.RUNPOD_S3_SECRET_ACCESS_KEY,
    replicas: environment.RUNPOD_S3_REPLICAS,
    timeoutMs: Math.max(10 * 60 * 1000, Number(environment.MODEL_LAB_TIMEOUT_MS) || 2 * 60 * 60 * 1000),
    pollIntervalMs: 2_000,
  }, options.dependencies);
  const enabled = runpod.configured && Boolean(ffmpegPath && fs.existsSync(ffmpegPath));
  const jobs = new Map();
  const alignments = new Map();
  const alignmentArchiveRoot = path.resolve(
    options.archiveRoot
      || String(environment.MODEL_LAB_ARCHIVE_ROOT || '').trim()
      || (production
        ? path.join(dataRoot, 'supervision-archive')
        : path.join(os.homedir(), 'Documents', 'Polymath Model Lab Archive')),
  );
  fs.mkdirSync(alignmentArchiveRoot, { recursive: true });
  const rawTestArchiveRoot = path.join(alignmentArchiveRoot, 'raw-tests');
  fs.mkdirSync(rawTestArchiveRoot, { recursive: true });
  const router = express.Router();
  const upload = multer({
    storage: multer.diskStorage({
      destination(request, file, callback) {
        callback(null, dataRoot);
      },
      filename(request, file, callback) {
        const extension = path.extname(file.originalname || '').toLowerCase();
        callback(null, `${crypto.randomUUID()}-source${extension}`);
      },
    }),
    limits: { files: 1, fields: 2 },
    fileFilter(request, file, callback) {
      callback(null, MEDIA_EXTENSIONS.has(path.extname(file.originalname || '').toLowerCase()));
    },
  });
  const alignmentUpload = multer({
    storage: multer.diskStorage({
      destination(request, file, callback) {
        callback(null, dataRoot);
      },
      filename(request, file, callback) {
        const extension = path.extname(file.originalname || '').toLowerCase();
        callback(null, `${crypto.randomUUID()}-${file.fieldname}${extension}`);
      },
    }),
    limits: { files: 2, fields: 2, fileSize: 32 * 1024 * 1024 },
    fileFilter(request, file, callback) {
      callback(null, NOTE_EXTENSIONS.has(path.extname(file.originalname || '').toLowerCase()));
    },
  });

  function capability() {
    const missing = [...runpod.missing];
    if (!ffmpegPath || !fs.existsSync(ffmpegPath)) missing.push('FFmpeg');
    return {
      enabled,
      adminOnly: true,
      localOnly,
      rawModelOutput: true,
      model: 'MuScriptor Large',
      checkpoint: String(environment.MODEL_LAB_MODEL_VERSION || 'muscriptor-tester/v001'),
      storageTargets: runpod.storageTargetCount,
      maximumSeconds,
      missing,
      limitations: {
        confidenceScores: false,
        measuredTempo: false,
        measuredVelocity: false,
        midiPrograms: false,
      },
      supervisedLearning: {
        available: true,
        acceptedReferenceFiles: [...NOTE_EXTENSIONS],
        supportsExistingRawOutput: true,
        manualAnchors: true,
        fiveSecondQualityWindows: true,
        sourceTimelineBound: true,
        historyStorage: artifactStore?.remote ? 'private-s3-compatible' : 'private-local-disk',
      },
    };
  }

  router.use((request, response, next) => {
    if (localOnly && !isLoopback(request)) {
      return response.status(404).json({ error: 'Model Lab is available only from localhost.' });
    }
    return next();
  });

  router.get('/capabilities', (request, response) => {
    response.json(capability());
  });

  async function processJob(job) {
    const preparedPath = path.join(dataRoot, `${job.id}-prepared.wav`);
    try {
      job.stage = 'Preparing mono 16 kHz test audio';
      job.progress = 8;
      await runFfmpeg(ffmpegPath, job.sourcePath, preparedPath, maximumSeconds);
      job.sourceDurationSeconds = readWavDurationSeconds(preparedPath);

      job.stage = 'Submitting raw audio to tester v001';
      job.progress = 15;
      const raw = await runpod.transcribe({
        job: {
          id: `model-lab-${job.id}`,
          title: job.title,
          instrument: 'band',
        },
        preparedPath,
        constraints: [],
        onProgress(remote) {
          const state = String(remote.state || '').toUpperCase();
          job.stage = state === 'IN_QUEUE'
            ? 'Waiting for a RunPod GPU worker'
            : state === 'IN_PROGRESS'
              ? 'MuScriptor is detecting instruments and notes'
              : `RunPod status: ${state || 'working'}`;
          job.progress = state === 'IN_QUEUE' ? 20 : state === 'IN_PROGRESS' ? 55 : job.progress;
        },
      });

      job.stage = 'Calculating instrument, MIDI, pitch, and timing statistics';
      job.progress = 92;
      job.result = {
        analysis: analyzeTranscription(raw),
        raw: {
          ...raw,
          sourceDurationSeconds: job.sourceDurationSeconds,
        },
      };
      job.status = 'completed';
      job.stage = 'Model analysis ready';
      job.progress = 100;
      job.completedAt = new Date().toISOString();
      job.checkpoint = capability().checkpoint;
      try {
        archiveRawTestSnapshot({ archiveRoot: rawTestArchiveRoot, job });
        try {
          const remoteSaved = await mirrorArchiveDirectory(
            artifactStore,
            `${REMOTE_HISTORY_PREFIX}/raw-tests/${job.id}`,
            path.join(rawTestArchiveRoot, job.id),
          );
          job.archive = { saved: true, remoteSaved, private: true, sourceMediaRetained: false };
        } catch (remoteError) {
          job.archive = {
            saved: true,
            remoteSaved: false,
            remoteError: remoteError.message || String(remoteError),
            private: true,
            sourceMediaRetained: false,
          };
        }
      } catch (archiveError) {
        job.archive = { saved: false, error: archiveError.message || String(archiveError) };
      }
    } catch (error) {
      job.status = 'failed';
      job.stage = 'Test failed';
      job.error = error.message || String(error);
    } finally {
      for (const candidate of [job.sourcePath, preparedPath]) {
        const resolved = path.resolve(candidate);
        if (!resolved.startsWith(`${dataRoot}${path.sep}`)) continue;
        try {
          fs.rmSync(resolved, { force: true });
        } catch {
          // Temporary lab files are cleaned on the next run if Windows still has a handle open.
        }
      }
    }
  }

  router.post('/jobs', (request, response, next) => {
    if (!enabled) {
      return response.status(503).json({
        error: `Model Lab is not configured${runpod.missing.length ? `: ${runpod.missing.join(', ')}` : '.'}`,
        capability: capability(),
      });
    }
    return upload.single('media')(request, response, (uploadError) => {
      if (uploadError) return next(uploadError);
      if (!request.file) return response.status(400).json({ error: 'Choose a supported audio or video file.' });
      const id = crypto.randomUUID();
      const title = String(request.body.title || path.parse(request.file.originalname).name || 'Model Lab test').slice(0, 120);
      const job = {
        id,
        title,
        filename: request.file.originalname,
        sourcePath: request.file.path,
        status: 'processing',
        stage: 'Queued locally',
        progress: 2,
        createdAt: new Date().toISOString(),
      };
      jobs.set(id, job);
      setImmediate(() => processJob(job));
      return response.status(202).json({ job: publicJob(job), capability: capability() });
    });
  });

  router.get('/jobs/:jobId', (request, response) => {
    const job = jobs.get(request.params.jobId);
    if (!job) return response.status(404).json({ error: 'Model Lab job not found. The local server may have restarted.' });
    return response.json({ job: publicJob(job), capability: capability() });
  });

  router.get('/history', async (request, response, next) => {
    const requestedLimit = Number(request.query.limit) || 100;
    const safeLimit = Math.max(1, Math.min(500, requestedLimit));
    try {
      let remote = { rawTests: [], alignments: [] };
      let remoteWarning = '';
      try {
        remote = await listRemoteHistory(artifactStore, remoteHistoryCacheRoot, safeLimit);
      } catch (remoteError) {
        remoteWarning = remoteError.message || String(remoteError);
      }
      const merge = (local, remoteRecords, dateField) => [...new Map(
        [...remoteRecords, ...local]
          .sort((left, right) => String(right[dateField]).localeCompare(String(left[dateField])))
          .map((record) => [record.id, record]),
      ).values()].slice(0, safeLimit);
      return response.json({
        rawTests: merge(listRawTestArchives(rawTestArchiveRoot, safeLimit), remote.rawTests, 'completedAt'),
        alignments: merge(listAlignmentArchives(alignmentArchiveRoot, safeLimit), remote.alignments, 'updatedAt'),
        storage: {
          private: true,
          persistent: Boolean(artifactStore?.remote || !production),
          provider: artifactStore?.remote ? 's3-compatible' : 'local-disk',
          sourceMediaRetained: false,
          warning: remoteWarning,
        },
      });
    } catch (error) {
      return next(error);
    }
  });

  router.get('/history/raw/:jobId', async (request, response, next) => {
    let archive = loadRawTestArchive(rawTestArchiveRoot, request.params.jobId);
    try {
      if (!archive && await restoreRemoteArchive(
        artifactStore, remoteHistoryCacheRoot, rawTestArchiveRoot, 'raw-tests', request.params.jobId,
      )) archive = loadRawTestArchive(rawTestArchiveRoot, request.params.jobId);
    } catch (error) {
      return next(error);
    }
    if (!archive) return response.status(404).json({ error: 'Archived model test not found.' });
    return response.json({
      job: {
        id: archive.manifest.id,
        title: archive.manifest.title,
        filename: archive.manifest.originalFilename,
        status: 'completed',
        stage: 'Loaded from private testing history',
        progress: 100,
        createdAt: archive.manifest.createdAt,
        completedAt: archive.manifest.completedAt,
        sourceDurationSeconds: archive.manifest.sourceDurationSeconds,
        result: { analysis: archive.analysis, raw: archive.raw },
        archived: true,
      },
      capability: capability(),
    });
  });

  function parseAlignmentMetadata(request) {
    if (!request.body?.metadata) return {};
    try {
      const parsed = JSON.parse(request.body.metadata);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      throw new Error('Alignment metadata must be valid JSON.');
    }
  }

  function extractRawNotes(raw) {
    const candidates = [raw?.notes, raw?.song?.notes, raw?.result?.notes, raw?.output?.notes];
    return candidates.find(Array.isArray) || [];
  }

  function samplePlotMatches(matches, limit = 1800) {
    if (matches.length <= limit) return matches;
    return Array.from({ length: limit }, (_, index) => matches[Math.round(
      (index / Math.max(1, limit - 1)) * (matches.length - 1),
    )]);
  }

  function publicAlignment(record) {
    const { result } = record;
    return {
      id: record.id,
      title: record.title,
      referenceFilename: record.referenceFilename,
      observedFilename: record.observedFilename,
      createdAt: record.createdAt,
      updatedAt: record.updatedAt,
      metrics: result.metrics,
      coarse: result.coarse,
      anchors: result.anchors,
      manualAnchors: result.manualAnchors,
      tempoSegments: result.tempoSegments,
      qualityWindows: result.qualityWindows,
      plot: {
        referenceDurationSeconds: result.supervisionPackage.timeline.referenceDurationSeconds,
        sourceDurationSeconds: result.supervisionPackage.timeline.sourceDurationSeconds
          || Math.max(...result.observed.map((note) => note.time + note.duration), 1),
        matches: samplePlotMatches(result.matches).map((match) => ({
          referenceTime: match.reference.time,
          sourceTime: match.observed.time,
          referenceMidi: match.reference.midi,
          observedMidi: match.observed.midi,
          exactPitch: match.exactPitch,
        })),
      },
      supervisionPackage: result.supervisionPackage,
      archive: record.archive ? {
        saved: true,
        id: record.id,
        privateLocalArchive: true,
      } : { saved: false },
    };
  }

  async function persistAlignment(record, files = {}) {
    record.archive = archiveAlignmentSnapshot({
      archiveRoot: alignmentArchiveRoot,
      record,
      referenceFile: files.referenceFile,
      observedFile: files.observedFile,
      analysis: publicAlignment(record),
    });
    try {
      record.archive.remoteSaved = await mirrorArchiveDirectory(
        artifactStore,
        `${REMOTE_HISTORY_PREFIX}/supervision/${record.id}`,
        record.archive.directory,
      );
    } catch (remoteError) {
      record.archive.remoteSaved = false;
      record.archive.remoteError = remoteError.message || String(remoteError);
    }
  }

  async function calculateAlignment(record, settings = {}) {
    const { alignNoteCoordinates } = await import(ALIGNMENT_MODULE_URL);
    record.settings = {
      ...record.settings,
      ...settings,
      sourceDurationSeconds: settings.sourceDurationSeconds
        ?? record.settings?.sourceDurationSeconds
        ?? null,
      manualAnchors: settings.manualAnchors ?? record.settings?.manualAnchors ?? [],
      excludedRanges: settings.excludedRanges ?? record.settings?.excludedRanges ?? [],
      reviewDecisions: settings.reviewDecisions ?? record.settings?.reviewDecisions ?? {},
    };
    record.result = alignNoteCoordinates(record.referenceNotes, record.observedNotes, record.settings);
    record.updatedAt = new Date().toISOString();
    return record;
  }

  router.post('/alignments', (request, response, next) => alignmentUpload.fields([
    { name: 'reference', maxCount: 1 },
    { name: 'observed', maxCount: 1 },
  ])(request, response, async (uploadError) => {
    const uploaded = Object.values(request.files || {}).flat();
    try {
      if (uploadError) throw uploadError;
      const metadata = parseAlignmentMetadata(request);
      const referenceFile = request.files?.reference?.[0];
      const observedFile = request.files?.observed?.[0];
      if (!referenceFile) throw new Error('Choose the desired/ideal MIDI or note JSON file.');
      const { loadNoteFile } = await import(ALIGNMENT_MODULE_URL);
      const referenceNotes = await loadNoteFile(referenceFile.path);
      let observedNotes;
      let observedFilename;
      let sourceDurationSeconds = Number(metadata.sourceDurationSeconds);
      if (observedFile) {
        observedNotes = await loadNoteFile(observedFile.path);
        observedFilename = observedFile.originalname;
      } else {
        const sourceJob = jobs.get(String(metadata.jobId || ''));
        if (!sourceJob || sourceJob.status !== 'completed') {
          throw new Error('Upload a raw MuScriptor MIDI/JSON file or finish a Model Lab transcription first.');
        }
        observedNotes = extractRawNotes(sourceJob.result?.raw);
        observedFilename = `${sourceJob.title || sourceJob.filename}-raw-model-output.json`;
        if (!Number.isFinite(sourceDurationSeconds) || sourceDurationSeconds <= 0) {
          sourceDurationSeconds = sourceJob.sourceDurationSeconds;
        }
      }
      if (!observedNotes?.length) throw new Error('The observed/model file contains no usable note array.');
      const id = crypto.randomUUID();
      const now = new Date().toISOString();
      const record = {
        id,
        title: String(metadata.title || path.parse(referenceFile.originalname).name || 'Supervised alignment').slice(0, 120),
        referenceFilename: referenceFile.originalname,
        observedFilename,
        referenceNotes,
        observedNotes,
        settings: {},
        createdAt: now,
        updatedAt: now,
      };
      await calculateAlignment(record, {
        sourceDurationSeconds: Number.isFinite(sourceDurationSeconds) && sourceDurationSeconds > 0
          ? sourceDurationSeconds
          : null,
        manualAnchors: Array.isArray(metadata.manualAnchors) ? metadata.manualAnchors : [],
        excludedRanges: Array.isArray(metadata.excludedRanges) ? metadata.excludedRanges : [],
        reviewDecisions: metadata.reviewDecisions && typeof metadata.reviewDecisions === 'object'
          ? metadata.reviewDecisions
          : {},
      });
      await persistAlignment(record, { referenceFile, observedFile });
      alignments.set(id, record);
      while (alignments.size > 50) alignments.delete(alignments.keys().next().value);
      return response.status(201).json({ alignment: publicAlignment(record) });
    } catch (error) {
      return next(error);
    } finally {
      for (const file of uploaded) {
        const resolved = path.resolve(file.path);
        if (!resolved.startsWith(`${dataRoot}${path.sep}`)) continue;
        try { fs.rmSync(resolved, { force: true }); } catch { /* cleaned later */ }
      }
    }
  }));

  router.get('/alignments', (request, response) => {
    return response.json({ alignments: listAlignmentArchives(alignmentArchiveRoot, request.query.limit) });
  });

  router.get('/alignments/:alignmentId', async (request, response, next) => {
    const record = alignments.get(request.params.alignmentId);
    if (record) return response.json({ alignment: publicAlignment(record) });
    let archive = loadAlignmentArchive(alignmentArchiveRoot, request.params.alignmentId);
    try {
      if (!archive && await restoreRemoteArchive(
        artifactStore, remoteHistoryCacheRoot, alignmentArchiveRoot, 'supervision', request.params.alignmentId,
      )) archive = loadAlignmentArchive(alignmentArchiveRoot, request.params.alignmentId);
    } catch (error) {
      return next(error);
    }
    if (!archive) return response.status(404).json({ error: 'Alignment not found in private testing history.' });
    return response.json({ alignment: { ...archive.analysis, archived: true } });
  });

  router.patch('/alignments/:alignmentId', async (request, response, next) => {
    try {
      const record = alignments.get(request.params.alignmentId);
      if (!record) return response.status(404).json({ error: 'Alignment not found. The server may have restarted.' });
      const sourceDurationSeconds = request.body?.sourceDurationSeconds == null
        ? record.settings.sourceDurationSeconds
        : Number(request.body.sourceDurationSeconds);
      await calculateAlignment(record, {
        sourceDurationSeconds: Number.isFinite(sourceDurationSeconds) && sourceDurationSeconds > 0
          ? sourceDurationSeconds
          : null,
        manualAnchors: Array.isArray(request.body?.manualAnchors)
          ? request.body.manualAnchors
          : record.settings.manualAnchors,
        excludedRanges: Array.isArray(request.body?.excludedRanges)
          ? request.body.excludedRanges
          : record.settings.excludedRanges,
        reviewDecisions: request.body?.reviewDecisions && typeof request.body.reviewDecisions === 'object'
          ? request.body.reviewDecisions
          : record.settings.reviewDecisions,
      });
      await persistAlignment(record);
      return response.json({ alignment: publicAlignment(record) });
    } catch (error) {
      return next(error);
    }
  });

  return { alignments, capability, enabled, jobs, router };
}

module.exports = {
  archiveRawTestSnapshot,
  archiveAlignmentSnapshot,
  createModelLab,
  isLoopback,
  listAlignmentArchives,
  listRawTestArchives,
  loadAlignmentArchive,
  loadRawTestArchive,
  readWavDurationSeconds,
  runFfmpeg,
};
