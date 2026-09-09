'use strict';

const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const {
  DEFAULT_FINE_TUNE_MODEL,
  createOpenAiFineTuningClient,
} = require('../openAiFineTuning');
const { buildDatasets } = require('./buildDataset');

const SOURCE_PATH = path.join(__dirname, 'polymath-assistant.examples.json');
const GENERATED_DIRECTORY = path.join(__dirname, 'generated');
const RUN_DIRECTORY = path.join(__dirname, '..', '..', '.local-dev', 'openai-fine-tuning');

function client() {
  return createOpenAiFineTuningClient({
    apiKey: process.env.OPENAI_API_KEY,
    baseUrl: process.env.OPENAI_BASE_URL,
    organization: process.env.OPENAI_ORGANIZATION,
    project: process.env.OPENAI_PROJECT,
  });
}

function writeRun(name, value) {
  fs.mkdirSync(RUN_DIRECTORY, { recursive: true });
  const safeName = String(name || 'run').replace(/[^A-Za-z0-9_-]+/g, '-');
  const destination = path.join(RUN_DIRECTORY, `${safeName}.json`);
  fs.writeFileSync(destination, JSON.stringify(value, null, 2), 'utf8');
  return destination;
}

function safeJob(job) {
  return {
    id: job?.id || '',
    status: job?.status || '',
    model: job?.model || '',
    fineTunedModel: job?.fine_tuned_model || null,
    trainingFile: job?.training_file || '',
    validationFile: job?.validation_file || '',
    trainedTokens: job?.trained_tokens ?? null,
    createdAt: job?.created_at || null,
    finishedAt: job?.finished_at || null,
    error: job?.error?.message || null,
  };
}

async function submit() {
  const built = buildDatasets({ sourcePath: SOURCE_PATH, outputDirectory: GENERATED_DIRECTORY });
  const api = client();
  let trainingFile;
  let validationFile;
  try {
    trainingFile = await api.uploadTrainingFile({
      filename: 'polymath-assistant-train.jsonl',
      jsonl: fs.readFileSync(built.trainingPath, 'utf8'),
    });
    validationFile = await api.uploadTrainingFile({
      filename: 'polymath-assistant-validation.jsonl',
      jsonl: fs.readFileSync(built.validationPath, 'utf8'),
    });
    const job = await api.createJob({
      trainingFileId: trainingFile.id,
      validationFileId: validationFile.id,
      model: process.env.OPENAI_FINE_TUNE_BASE_MODEL || DEFAULT_FINE_TUNE_MODEL,
      suffix: process.env.OPENAI_FINE_TUNE_SUFFIX || 'polymath-assistant-v001',
      epochs: process.env.OPENAI_FINE_TUNE_EPOCHS || 'auto',
      metadata: { product: 'polymath', dataset: 'assistant-v001' },
    });
    const record = { createdAt: new Date().toISOString(), built, job: safeJob(job) };
    const recordPath = writeRun(job.id, record);
    return { recordPath, ...record };
  } catch (error) {
    await Promise.allSettled(
      [trainingFile?.id, validationFile?.id].filter(Boolean).map((fileId) => api.deleteFile(fileId)),
    );
    throw error;
  }
}

async function cleanup() {
  const api = client();
  const listed = await api.listFiles({ purpose: 'fine-tune', limit: 100 });
  const ownedNames = new Set([
    'polymath-assistant-train.jsonl',
    'polymath-assistant-validation.jsonl',
  ]);
  const matches = (Array.isArray(listed?.data) ? listed.data : [])
    .filter((file) => ownedNames.has(String(file?.filename || '').trim()) && file?.id);
  const deleted = [];
  for (const file of matches) {
    await api.deleteFile(file.id);
    deleted.push({ id: file.id, filename: file.filename });
  }
  return { deletedCount: deleted.length, deleted };
}

async function status(jobId) {
  const job = await client().retrieveJob(jobId);
  const record = { checkedAt: new Date().toISOString(), job: safeJob(job) };
  return { recordPath: writeRun(job.id, record), ...record };
}

async function cancel(jobId) {
  const job = await client().cancelJob(jobId);
  const record = { cancelledAt: new Date().toISOString(), job: safeJob(job) };
  return { recordPath: writeRun(job.id, record), ...record };
}

async function main() {
  const command = String(process.argv[2] || 'validate').toLowerCase();
  const jobId = process.argv[3];
  let result;
  if (command === 'validate') {
    result = buildDatasets({ sourcePath: SOURCE_PATH, outputDirectory: GENERATED_DIRECTORY });
  } else if (command === 'submit') {
    result = await submit();
  } else if (command === 'status') {
    if (!jobId) throw new Error('Usage: node fine-tuning/cli.js status <ftjob-id>');
    result = await status(jobId);
  } else if (command === 'cancel') {
    if (!jobId) throw new Error('Usage: node fine-tuning/cli.js cancel <ftjob-id>');
    result = await cancel(jobId);
  } else if (command === 'cleanup') {
    result = await cleanup();
  } else {
    throw new Error('Commands: validate, submit, status <ftjob-id>, cancel <ftjob-id>, cleanup.');
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${error.message || error}\n`);
    process.exitCode = 1;
  });
}

module.exports = { cancel, cleanup, safeJob, status, submit, writeRun };
