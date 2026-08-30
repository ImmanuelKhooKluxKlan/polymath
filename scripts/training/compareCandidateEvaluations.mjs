#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function repeatExcess(diagnostics, maximumGapMs = 250) {
  return diagnostics.patternRecognition.sameKeyRepeatProfile
    .find((entry) => entry.maximumGapMs === maximumGapMs)?.excessPredictedRepeats ?? Infinity;
}

function snapshot(label, metrics) {
  const diagnostics = metrics.diagnostics50ms;
  return {
    label,
    boundaryF1: diagnostics.f1,
    rawF1_50ms: metrics['50ms'].microF1,
    rawF1_100ms: metrics['100ms'].microF1,
    rawF1_250ms: metrics['250ms'].microF1,
    onsetOffsetF1: diagnostics.onsetAndOffset.f1,
    frameF1: diagnostics.frame.f1,
    matchedNotes: diagnostics.matchedNotes,
    ignoredNotes: diagnostics.ignoredNotes,
    extraNotes: diagnostics.falsePositiveNotes,
    cutOffNotes: diagnostics.cutOffNotes,
    overlongNotes: diagnostics.overlongNotes,
    octaveSubstitutions: diagnostics.errorCauses.octaveSubstitution,
    timingNearMisses: diagnostics.errorCauses.timingNearMiss,
    spuriousExtras: diagnostics.errorCauses.spuriousExtra,
    completeChords: diagnostics.patternRecognition.chords.complete,
    missedChords: diagnostics.patternRecognition.chords.missed,
    repeatExcess250ms: repeatExcess(diagnostics),
    medianOnsetErrorMs: diagnostics.timing.medianOnsetErrorMs,
    medianOffsetErrorMs: diagnostics.timing.medianOffsetErrorMs,
    p95OffsetErrorMs: diagnostics.timing.p95OffsetErrorMs,
  };
}

function gate(name, candidateValue, incumbentValue, predicate, direction) {
  return {
    name,
    passed: predicate(candidateValue, incumbentValue),
    candidate: candidateValue,
    incumbent: incumbentValue,
    desiredDirection: direction,
  };
}

function markdown(report) {
  const columns = [
    ['Boundary onset F1', 'boundaryF1'],
    ['Onset + offset F1', 'onsetOffsetF1'],
    ['Frame F1', 'frameF1'],
    ['Cut-off notes', 'cutOffNotes'],
    ['Overlong notes', 'overlongNotes'],
    ['250 ms repeat excess', 'repeatExcess250ms'],
    ['Complete chords', 'completeChords'],
    ['Octave substitutions', 'octaveSubstitutions'],
    ['Spurious extras', 'spuriousExtras'],
    ['P95 offset error (ms)', 'p95OffsetErrorMs'],
  ];
  const lines = [
    '# Frozen checkpoint comparison',
    '',
    `Decision: **${report.decision}**`,
    '',
    '| Metric | Original | Incumbent | Candidate |',
    '|---|---:|---:|---:|',
    ...columns.map(([label, key]) => `| ${label} | ${report.models.original[key]} | ${report.models.incumbent[key]} | ${report.models.candidate[key]} |`),
    '',
    '## Promotion gates',
    '',
    ...report.gates.map((item) => `- ${item.passed ? 'PASS' : 'FAIL'} - ${item.name}: candidate ${item.candidate}, incumbent ${item.incumbent} (${item.desiredDirection})`),
    '',
    'A rejected checkpoint remains append-only research evidence. It is never copied over the public weights.',
  ];
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.incumbent || !args.candidate || !args.out) {
    throw new Error('Usage: --incumbent phase1-result.json --candidate phase2-result.json --out comparison.json');
  }
  const incumbentRecord = JSON.parse(await fs.readFile(path.resolve(args.incumbent), 'utf8'));
  const candidateRecord = JSON.parse(await fs.readFile(path.resolve(args.candidate), 'utf8'));
  const models = {
    original: snapshot('original', candidateRecord.baseline),
    incumbent: snapshot(incumbentRecord.version || 'incumbent', incumbentRecord.candidate),
    candidate: snapshot(candidateRecord.version || 'candidate', candidateRecord.candidate),
  };
  const c = models.candidate;
  const i = models.incumbent;
  const gates = [
    gate('Boundary onset F1 improves', c.boundaryF1, i.boundaryF1, (a, b) => a > b, 'higher'),
    gate('Onset + offset F1 improves', c.onsetOffsetF1, i.onsetOffsetF1, (a, b) => a > b, 'higher'),
    gate('Frame F1 does not regress over 0.0005', c.frameF1, i.frameF1, (a, b) => a >= b - 0.0005, 'higher'),
    gate('Cut-off notes do not increase', c.cutOffNotes, i.cutOffNotes, (a, b) => a <= b, 'lower'),
    gate('Overlong notes do not increase', c.overlongNotes, i.overlongNotes, (a, b) => a <= b, 'lower'),
    gate('Repeat excess does not increase', c.repeatExcess250ms, i.repeatExcess250ms, (a, b) => a <= b, 'lower'),
    gate('Complete chords do not decrease', c.completeChords, i.completeChords, (a, b) => a >= b, 'higher'),
    gate('Octave substitutions do not increase', c.octaveSubstitutions, i.octaveSubstitutions, (a, b) => a <= b, 'lower'),
    gate('Spurious extras do not increase', c.spuriousExtras, i.spuriousExtras, (a, b) => a <= b, 'lower'),
  ];
  const report = {
    schema: 'polymath-checkpoint-promotion-decision-v1',
    generatedAt: new Date().toISOString(),
    decision: gates.every(({ passed }) => passed) ? 'KEEP_FOR_LISTENING_REVIEW' : 'REJECT_FOR_PROMOTION',
    models,
    gates,
  };
  const destination = path.resolve(args.out);
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, `${JSON.stringify(report, null, 2)}\n`);
  await fs.writeFile(destination.replace(/\.json$/i, '.md'), markdown(report));
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Checkpoint comparison failed: ${error.message}\n`);
  process.exitCode = 1;
});
