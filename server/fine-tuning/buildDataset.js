'use strict';

const fs = require('fs');
const path = require('path');
const { SYSTEMS } = require('../assistantBehavior');
const { formatJsonl } = require('../openAiFineTuning');

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
