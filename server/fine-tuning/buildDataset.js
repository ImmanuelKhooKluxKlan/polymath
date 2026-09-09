'use strict';

const fs = require('fs');
const path = require('path');
const { formatJsonl } = require('../openAiFineTuning');

const SYSTEMS = Object.freeze({
  teacher: [
    'You are Polymath Virtual Teacher, an expert music teacher speaking naturally inside a lesson.',
    'Use plain language and give one actionable correction or exercise at a time.',
    'Never invent something you heard, saw, measured, or remember. State when evidence is missing.',
    'Accuracy and learner safety matter more than confidence. Keep most replies below 100 words.',
  ].join(' '),
  support: [
    'You are Polymath Support. Be concise and dyslexia-friendly.',
    'Never claim to change accounts, balances, payments, subscriptions, or jobs.',
    'Never request passwords, one-time codes, API keys, private keys, or full card details.',
    'When human account access is required, give the next safe step.',
  ].join(' '),
  companion: [
    'You are an adult-only, opted-in Polymath virtual companion and music teacher.',
    'You may be warm, playful, and lightly flirtatious while remaining clearly virtual.',
    'Never claim physical presence, pressure spending, encourage dependency, isolate the learner, or invent sensory evidence.',
    'Keep replies natural, concise, and useful.',
  ].join(' '),
  chatboss: [
    'You are Polymath Chat Boss, a precise technical and business thought partner.',
    'Lead with the answer, distinguish facts from assumptions, protect credentials, and never claim checks you did not perform.',
  ].join(' '),
});

function readSource(sourcePath) {
  const parsed = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
  if (!Array.isArray(parsed)) throw new Error('Fine-tuning source must be a JSON array.');
  return parsed;
}

function materialize(source) {
  return source.map((example, index) => {
    const task = String(example?.task || '').trim();
    const split = String(example?.split || '').trim();
    if (!SYSTEMS[task]) throw new Error(`Example ${index + 1} has an unknown task.`);
    if (!['train', 'validation'].includes(split)) throw new Error(`Example ${index + 1} has an invalid split.`);
    if (!String(example?.user || '').trim() || !String(example?.assistant || '').trim()) {
      throw new Error(`Example ${index + 1} is missing user or assistant text.`);
    }
    return {
      split,
      category: String(example.category || 'uncategorized').trim(),
      messages: [
        { role: 'system', content: SYSTEMS[task] },
        { role: 'user', content: String(example.user).trim() },
        { role: 'assistant', content: String(example.assistant).trim() },
      ],
    };
  });
}

function buildDatasets({ sourcePath, outputDirectory }) {
  const materialized = materialize(readSource(sourcePath));
  const training = formatJsonl(
    materialized.filter((example) => example.split === 'train'),
  );
  const validation = formatJsonl(
    materialized.filter((example) => example.split === 'validation'),
  );
  fs.mkdirSync(outputDirectory, { recursive: true });
  const trainingPath = path.join(outputDirectory, 'train.jsonl');
  const validationPath = path.join(outputDirectory, 'validation.jsonl');
  fs.writeFileSync(trainingPath, training.jsonl, 'utf8');
  fs.writeFileSync(validationPath, validation.jsonl, 'utf8');
  return {
    trainingPath,
    validationPath,
    training: training.summary,
    validation: validation.summary,
  };
}

if (require.main === module) {
  const sourcePath = path.join(__dirname, 'polymath-assistant.examples.json');
  const outputDirectory = path.join(__dirname, 'generated');
  const result = buildDatasets({ sourcePath, outputDirectory });
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

module.exports = { SYSTEMS, buildDatasets, materialize, readSource };
