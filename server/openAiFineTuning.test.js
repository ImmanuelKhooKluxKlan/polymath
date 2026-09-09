'use strict';

const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');
const {
  createOpenAiFineTuningClient,
  formatJsonl,
  parseJsonl,
  validateTrainingExamples,
} = require('./openAiFineTuning');
const { buildDatasets } = require('./fine-tuning/buildDataset');
const { gradeReply } = require('./fine-tuning/evaluateAssistant');

function examples(count = 10) {
  return Array.from({ length: count }, (_, index) => ({
    messages: [
      { role: 'system', content: 'Be accurate.' },
      { role: 'user', content: `Question ${index + 1}` },
      { role: 'assistant', content: `Answer ${index + 1}` },
    ],
  }));
}

function response(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

test('fine-tuning data is valid JSONL with unique prompts and no credentials', () => {
  const formatted = formatJsonl(examples());
  assert.equal(formatted.summary.exampleCount, 10);
  assert.equal(parseJsonl(formatted.jsonl).length, 10);
  assert.throws(() => validateTrainingExamples(examples(9)), /At least 10/);

  const duplicated = examples();
  duplicated[9].messages[1].content = 'Question 1';
  assert.throws(() => validateTrainingExamples(duplicated), /duplicates/);

  const credential = examples();
  credential[0].messages[2].content = `Use ${['sk', 'proj', 'this_should_never_be_training_data'].join('-')}`;
  assert.throws(() => validateTrainingExamples(credential), /credential/);
});

test('curated Polymath data builds separate train and holdout files', (context) => {
  const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-openai-data-'));
  context.after(() => fs.rmSync(outputDirectory, { recursive: true, force: true }));
  const result = buildDatasets({
    sourcePath: path.join(__dirname, 'fine-tuning', 'polymath-assistant.examples.json'),
    outputDirectory,
  });
  assert.equal(result.training.exampleCount, 26);
  assert.equal(result.validation.exampleCount, 10);
  assert.equal(parseJsonl(fs.readFileSync(result.trainingPath, 'utf8')).length, 26);
  assert.equal(parseJsonl(fs.readFileSync(result.validationPath, 'utf8')).length, 10);
});

test('fine-tuning client uploads JSONL and submits a supervised job without exposing its key', async () => {
  const requests = [];
  const client = createOpenAiFineTuningClient({
    apiKey: 'test-never-log',
    async fetch(url, options) {
      requests.push({ url, options });
      if (url.endsWith('/files')) return response({ id: 'file-training123', purpose: 'fine-tune' });
      if (url.endsWith('/fine_tuning/jobs')) {
        return response({ id: 'ftjob-polymath123', status: 'queued', model: 'gpt-4.1-mini-2025-04-14' });
      }
      return response({ id: 'ftjob-polymath123', status: 'running' });
    },
  });
  const formatted = formatJsonl(examples());
  const file = await client.uploadTrainingFile({ filename: 'training.jsonl', jsonl: formatted.jsonl });
  assert.equal(file.id, 'file-training123');
  assert.equal(requests[0].options.body instanceof FormData, true);
  assert.equal(Object.hasOwn(requests[0].options.headers, 'Content-Type'), false);

  const job = await client.createJob({
    trainingFileId: file.id,
    model: 'gpt-4.1-mini-2025-04-14',
    suffix: 'polymath-test',
  });
  assert.equal(job.id, 'ftjob-polymath123');
  const jobBody = JSON.parse(requests[1].options.body);
  assert.equal(jobBody.method.type, 'supervised');
  assert.equal(jobBody.method.supervised.hyperparameters.n_epochs, 'auto');
  assert.equal(JSON.stringify(requests).includes('test-never-log'), true);
  assert.equal(JSON.stringify(job).includes('test-never-log'), false);
});

test('fine-tuning client lists and deletes only validated file IDs', async () => {
  const requests = [];
  const client = createOpenAiFineTuningClient({
    apiKey: 'test-never-log',
    async fetch(url, options) {
      requests.push({ url, options });
      if (options.method === 'DELETE') return response({ id: 'file-polymath123', deleted: true });
      return response({ data: [{ id: 'file-polymath123', filename: 'polymath-assistant-train.jsonl' }] });
    },
  });
  const files = await client.listFiles({ purpose: 'fine-tune', limit: 25 });
  assert.equal(files.data[0].id, 'file-polymath123');
  assert.match(requests[0].url, /purpose=fine-tune/);
  assert.match(requests[0].url, /limit=25/);
  assert.equal((await client.deleteFile('file-polymath123')).deleted, true);
  await assert.rejects(() => client.deleteFile('../secret'), /Invalid OpenAI file ID/);
});

test('deterministic eval grader enforces required facts, forbidden claims, and length', () => {
  const rubric = {
    mustInclude: ['C4', '500 ms'],
    mustIncludeAny: [['slowly', 'use a metronome']],
    mustNotInclude: ['I heard'],
    maxWords: 20,
  };
  assert.equal(gradeReply('C4 ended 500 ms early. Repeat it slowly.', rubric).passed, true);
  const failed = gradeReply('I heard C4 end early.', rubric);
  assert.equal(failed.passed, false);
  assert.deepEqual(failed.missing, ['500 ms']);
  assert.deepEqual(failed.missingAny, [['slowly', 'use a metronome']]);
  assert.deepEqual(failed.forbiddenFound, ['I heard']);
});

test('deterministic eval grader normalizes common contractions and supports safe negation', () => {
  const result = gradeReply("I can't verify that, so don't paste your password.", {
    mustInclude: ['cannot', 'do not', 'password'],
    mustNotInclude: ['yes, paste your password'],
    maxWords: 30,
  });
  assert.equal(result.passed, true);
});
