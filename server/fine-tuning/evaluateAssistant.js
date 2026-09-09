'use strict';

const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { createOpenAiResponsesClient, extractOutputText } = require('../openAiResponses');
const { SYSTEMS } = require('../assistantBehavior');

function clean(value) {
  return String(value || '').trim();
}

function normalizeForMatching(value) {
  return clean(value)
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\bcan(?:no|')t\b/g, 'cannot')
    .replace(/\bdon't\b/g, 'do not')
    .replace(/\bwon't\b/g, 'will not')
    .replace(/\bshouldn't\b/g, 'should not')
    .replace(/\s+/g, ' ');
}

function includesText(haystack, needle) {
  return normalizeForMatching(haystack).includes(normalizeForMatching(needle));
}

function gradeReply(reply, testCase) {
  const required = Array.isArray(testCase.mustInclude) ? testCase.mustInclude : [];
  const requiredAny = Array.isArray(testCase.mustIncludeAny) ? testCase.mustIncludeAny : [];
  const forbidden = Array.isArray(testCase.mustNotInclude) ? testCase.mustNotInclude : [];
  const missing = required.filter((phrase) => !includesText(reply, phrase));
  const missingAny = requiredAny
    .map((group) => (Array.isArray(group) ? group : [group]).map(clean).filter(Boolean))
    .filter((group) => group.length && !group.some((phrase) => includesText(reply, phrase)));
  const forbiddenFound = forbidden.filter((phrase) => includesText(reply, phrase));
  const words = clean(reply).split(/\s+/).filter(Boolean).length;
  const tooLong = words > Math.max(1, Number(testCase.maxWords) || 200);
  return {
    passed: !missing.length && !missingAny.length && !forbiddenFound.length && !tooLong,
    missing,
    missingAny,
    forbiddenFound,
    words,
    tooLong,
  };
}

async function evaluateModel({ model, cases, clientFactory = createOpenAiResponsesClient }) {
  const client = clientFactory({
    apiKey: process.env.OPENAI_API_KEY,
    baseUrl: process.env.OPENAI_BASE_URL,
    organization: process.env.OPENAI_ORGANIZATION,
    project: process.env.OPENAI_PROJECT,
    model,
    reasoningEffort: 'low',
    timeoutMs: process.env.OPENAI_TIMEOUT_MS,
  });
  const results = [];
  for (const testCase of cases) {
    const body = await client.chat([
      { role: 'system', content: SYSTEMS[testCase.task] },
      { role: 'user', content: testCase.user },
    ], {
      max_output_tokens: 500,
      reasoning_effort: 'low',
      metadata: { workload: 'fine-tune-eval', eval_id: testCase.id },
    });
    const reply = extractOutputText(body);
    results.push({ id: testCase.id, task: testCase.task, reply, ...gradeReply(reply, testCase) });
  }
  const passed = results.filter((result) => result.passed).length;
  return {
    model,
    passed,
    total: results.length,
    passRate: results.length ? Number((passed / results.length).toFixed(4)) : 0,
    results,
  };
}

async function main() {
  const models = process.argv.slice(2).map(clean).filter(Boolean);
  if (!models.length) {
    throw new Error('Usage: node fine-tuning/evaluateAssistant.js <baseline-model> [candidate-model]');
  }
  const casesPath = path.join(__dirname, 'assistant-evals.json');
  const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
  const reports = [];
  for (const model of models) reports.push(await evaluateModel({ model, cases }));
  const outputDirectory = path.join(__dirname, '..', '..', '.local-dev', 'openai-fine-tuning');
  fs.mkdirSync(outputDirectory, { recursive: true });
  const outputPath = path.join(outputDirectory, `eval-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify({ createdAt: new Date().toISOString(), reports }, null, 2));
  process.stdout.write(`${JSON.stringify({
    outputPath,
    reports: reports.map(({ model, passed, total, passRate }) => ({ model, passed, total, passRate })),
  }, null, 2)}\n`);
  if (reports.some((report) => report.passRate < 0.9)) process.exitCode = 2;
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${error.message || error}\n`);
    process.exitCode = 1;
  });
}

module.exports = { evaluateModel, gradeReply, includesText, normalizeForMatching };
