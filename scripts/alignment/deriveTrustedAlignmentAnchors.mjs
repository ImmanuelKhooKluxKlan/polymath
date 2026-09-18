#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

function parseArguments(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`${token} requires a value.`);
    values[token.slice(2)] = value;
    index += 1;
  }
  return values;
}

export function deriveTrustedAlignmentAnchors(report, {
  cutoffSeconds,
  minimumSupport = 1,
  reason = 'Human-reviewed reference trust boundary.',
} = {}) {
  if (!Number.isFinite(cutoffSeconds) || cutoffSeconds <= 0) {
    throw new Error('cutoffSeconds must be a positive finite number.');
  }
  const sourceAnchors = Array.isArray(report?.anchors) ? report.anchors : [];
  const anchors = sourceAnchors
    .map((anchor) => ({
      referenceTime: Number(anchor?.referenceTime),
      observedTime: Number(anchor?.observedTime),
      support: Number(anchor?.support || 0),
      structuralSimilarity: Number(anchor?.structuralSimilarity),
      sourceKind: String(anchor?.kind || 'unknown'),
    }))
    .filter((anchor) => Number.isFinite(anchor.referenceTime) && Number.isFinite(anchor.observedTime))
    .filter((anchor) => anchor.referenceTime >= 0 && anchor.referenceTime < cutoffSeconds)
    .filter((anchor) => anchor.support >= minimumSupport)
    .sort((left, right) => left.referenceTime - right.referenceTime || left.observedTime - right.observedTime);

  const monotonic = [];
  for (const anchor of anchors) {
    const previous = monotonic.at(-1);
    if (!previous || (
      anchor.referenceTime > previous.referenceTime
      && anchor.observedTime > previous.observedTime
    )) monotonic.push(anchor);
  }
  if (monotonic.length < 2) {
    throw new Error('At least two supported pre-cutoff anchors are required.');
  }

  const left = monotonic.at(-2);
  const right = monotonic.at(-1);
  const slope = (right.observedTime - left.observedTime)
    / (right.referenceTime - left.referenceTime);
  if (!Number.isFinite(slope) || slope <= 0) {
    throw new Error('The last two trusted anchors cannot define a forward boundary mapping.');
  }
  const boundaryObservedTime = right.observedTime
    + (cutoffSeconds - right.referenceTime) * slope;
  const outputAnchors = [
    ...monotonic.map(({
      referenceTime,
      observedTime,
      support,
      structuralSimilarity,
      sourceKind,
    }) => ({
      referenceTime,
      observedTime,
      sourceSupport: support,
      sourceKind,
      ...(Number.isFinite(structuralSimilarity) ? { structuralSimilarity } : {}),
    })),
    {
      referenceTime: cutoffSeconds,
      observedTime: Number(boundaryObservedTime.toFixed(9)),
      sourceSupport: Math.min(left.support, right.support),
      sourceKind: 'pre-cutoff-linear-extrapolation',
      ...(Number.isFinite(left.structuralSimilarity) && Number.isFinite(right.structuralSimilarity)
        ? { structuralSimilarity: Math.min(left.structuralSimilarity, right.structuralSimilarity) }
        : {}),
    },
  ];

  return {
    schema: 'polymath-trusted-alignment-anchor-seeds-v1',
    cutoffSeconds,
    boundaryRule: 'The cutoff anchor is extrapolated from the last two supported anchors strictly before the cutoff; no post-cutoff anchor is read.',
    reason,
    sourceAnchorCount: sourceAnchors.length,
    retainedSourceAnchorCount: monotonic.length,
    excludedSourceAnchorCount: sourceAnchors.length - monotonic.length,
    minimumSupport,
    boundaryObservedTime: Number(boundaryObservedTime.toFixed(9)),
    anchors: outputAnchors,
  };
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.output || !args.cutoff) {
    throw new Error('Usage: --input alignment-report.json --output anchors.json --cutoff seconds [--minimum-support n] [--reason text]');
  }
  const source = path.resolve(args.input);
  const destination = path.resolve(args.output);
  const sourceBytes = await fs.readFile(source);
  const report = JSON.parse(sourceBytes.toString('utf8'));
  const result = deriveTrustedAlignmentAnchors(report, {
    cutoffSeconds: Number(args.cutoff),
    minimumSupport: args['minimum-support'] == null ? 1 : Number(args['minimum-support']),
    reason: args.reason || 'Human-reviewed reference trust boundary.',
  });
  result.sourceAlignment = source;
  result.sourceAlignmentSha256 = crypto.createHash('sha256').update(sourceBytes).digest('hex');
  await fs.mkdir(path.dirname(destination), { recursive: true });
  const temporary = `${destination}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  await fs.rename(temporary, destination);
  process.stdout.write(`${JSON.stringify({ output: destination, ...result }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`Trusted anchor derivation failed: ${error.message}\n`);
    process.exitCode = 1;
  });
}
