#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const next = argv[index + 1];
    values[token.slice(2)] = next && !next.startsWith('--') ? next : true;
    if (values[token.slice(2)] !== true) index += 1;
  }
  return values;
}

function finite(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function passes(window, thresholds) {
  return (
    finite(window.matchedNotes, 0) >= thresholds.minimumMatches
    && finite(window.matchedPercent, 0) >= thresholds.minimumCoveragePercent
    && finite(window.exactPitchPercent, 0) >= thresholds.minimumExactPitchPercent
    && finite(window.medianResidualMs, Infinity) <= thresholds.maximumMedianResidualMs
    && finite(window.p95ResidualMs, Infinity) <= thresholds.maximumP95ResidualMs
    && Math.abs(finite(window.localTempoDifferencePercent, Infinity)) <= thresholds.maximumTempoDifferencePercent
    && finite(window.structuralSimilarity, 0) >= thresholds.minimumStructuralSimilarity
  );
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.output) {
    throw new Error('Usage: node scripts/training/reviewAlignmentWindows.mjs --input aligned-training-labels.json --output reviewed-labels.json');
  }
  const source = path.resolve(String(args.input));
  const destination = path.resolve(String(args.output));
  if (source === destination) throw new Error('Refusing to overwrite the source supervision package.');

  const thresholds = {
    minimumMatches: finite(args['minimum-matches'], 8),
    minimumCoveragePercent: finite(args['minimum-coverage-percent'], 55),
    minimumExactPitchPercent: finite(args['minimum-exact-pitch-percent'], 72),
    maximumMedianResidualMs: finite(args['maximum-median-residual-ms'], 400),
    maximumP95ResidualMs: finite(args['maximum-p95-residual-ms'], 950),
    maximumTempoDifferencePercent: finite(args['maximum-tempo-difference-percent'], 6),
    minimumStructuralSimilarity: finite(args['minimum-structural-similarity'], 0.88),
  };

  const payload = JSON.parse(await fs.readFile(source, 'utf8'));
  const windows = payload?.alignment?.qualityWindows;
  if (!Array.isArray(windows) || !windows.length || !Array.isArray(payload.notes)) {
    throw new Error('The input is not a Polymath aligned-training-labels package.');
  }

  const acceptedIds = new Set();
  const rejectedReasons = {};
  let manualAccepted = 0;
  let manualRejected = 0;
  for (const window of windows) {
    const manualDecision = String(window.decision || '').toLowerCase();
    const manuallyAccepted = manualDecision === 'accept' || window.status === 'accepted-manually';
    const manuallyRejected = manualDecision === 'reject' || window.status === 'rejected-manually';
    const accepted = manuallyAccepted || (!manuallyRejected && passes(window, thresholds));
    if (accepted) acceptedIds.add(window.id);
    if (manuallyAccepted) manualAccepted += 1;
    if (manuallyRejected) manualRejected += 1;
    const reasons = [];
    if (finite(window.matchedNotes, 0) < thresholds.minimumMatches) reasons.push('too-few-matches');
    if (finite(window.matchedPercent, 0) < thresholds.minimumCoveragePercent) reasons.push('weak-coverage');
    if (finite(window.exactPitchPercent, 0) < thresholds.minimumExactPitchPercent) reasons.push('weak-exact-pitch');
    if (finite(window.medianResidualMs, Infinity) > thresholds.maximumMedianResidualMs) reasons.push('median-timing-residual');
    if (finite(window.p95ResidualMs, Infinity) > thresholds.maximumP95ResidualMs) reasons.push('tail-timing-residual');
    if (Math.abs(finite(window.localTempoDifferencePercent, Infinity)) > thresholds.maximumTempoDifferencePercent) reasons.push('local-tempo-warp');
    if (finite(window.structuralSimilarity, 0) < thresholds.minimumStructuralSimilarity) reasons.push('weak-structure');
    rejectedReasons[window.id] = manuallyRejected ? ['manual-rejection'] : reasons;
    window.preTrainingReview = {
      accepted,
      source: manuallyAccepted || manuallyRejected ? 'manual' : 'deterministic-quality-gate-v1',
      reasons: accepted ? [] : rejectedReasons[window.id],
    };
    window.status = accepted ? (manuallyAccepted ? 'accepted-manually' : 'trusted') : 'rejected-by-quality-gate';
    window.trainingEligible = accepted;
  }

  let eligibleNotes = 0;
  for (const note of payload.notes) {
    const eligible = acceptedIds.has(note.qualityWindowId);
    note.trainingEligible = eligible;
    note.qualityStatus = eligible
      ? (windows.find((window) => window.id === note.qualityWindowId)?.status || 'trusted')
      : 'rejected-by-quality-gate';
    if (eligible) eligibleNotes += 1;
  }

  payload.review = {
    ...(payload.review || {}),
    readyForTraining: acceptedIds.size > 0,
    deterministicQualityGate: {
      schema: 'polymath-alignment-quality-gate-v1',
      sourcePackage: source,
      thresholds,
      totalWindows: windows.length,
      acceptedWindows: acceptedIds.size,
      rejectedWindows: windows.length - acceptedIds.size,
      manualAccepted,
      manualRejected,
      totalNotes: payload.notes.length,
      eligibleNotes,
    },
  };
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, `${JSON.stringify(payload, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(payload.review.deterministicQualityGate, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Alignment review failed: ${error.message}\n`);
  process.exitCode = 1;
});
