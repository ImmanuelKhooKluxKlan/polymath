const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
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
    error: job.error || '',
    result: job.status === 'completed' ? job.result : undefined,
  };
}

function createModelLab(environment = process.env, options = {}) {
  const localOnly = String(environment.MODEL_LAB_LOCAL_ONLY || 'false').trim().toLowerCase() === 'true';
  const dataRoot = path.resolve(options.dataRoot || path.join(__dirname, 'data', 'model-lab'));
  const ffmpegPath = String(environment.FFMPEG_PATH || '').trim() || bundledFfmpegPath;
  const maximumSeconds = Math.max(10, Math.min(3600, Number(environment.MODEL_LAB_MAX_SECONDS) || 600));
  fs.mkdirSync(dataRoot, { recursive: true });

  const runpod = createRunpodServerlessClient({
    endpointId: environment.RUNPOD_SERVERLESS_ENDPOINT_ID || environment.RUNPOD_ENDPOINT_ID,
    apiKey: environment.RUNPOD_API_KEY,
    volumeId: environment.RUNPOD_NETWORK_VOLUME_ID,
    region: environment.RUNPOD_S3_REGION,
    s3Endpoint: environment.RUNPOD_S3_ENDPOINT,
    s3AccessKeyId: environment.RUNPOD_S3_ACCESS_KEY_ID,
    s3SecretAccessKey: environment.RUNPOD_S3_SECRET_ACCESS_KEY,
    timeoutMs: Math.max(10 * 60 * 1000, Number(environment.MODEL_LAB_TIMEOUT_MS) || 2 * 60 * 60 * 1000),
    pollIntervalMs: 2_000,
  }, options.dependencies);
  const enabled = runpod.configured && Boolean(ffmpegPath && fs.existsSync(ffmpegPath));
  const jobs = new Map();
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
      maximumSeconds,
      missing,
      limitations: {
        confidenceScores: false,
        measuredTempo: false,
        measuredVelocity: false,
        midiPrograms: false,
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
        raw,
      };
      job.status = 'completed';
      job.stage = 'Model analysis ready';
      job.progress = 100;
      job.completedAt = new Date().toISOString();
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

  return { capability, enabled, jobs, router };
}

module.exports = {
  createModelLab,
  isLoopback,
  runFfmpeg,
};
