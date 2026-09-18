#!/usr/bin/env node

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

function boundary(value, name, fallback) {
  if (value == null || value === '') return fallback;
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new Error(`${name} must be a non-negative number of seconds.`);
  }
  return seconds;
}

function validNote(item) {
  return Number.isFinite(Number(item?.time))
    && Number.isFinite(Number(item?.duration))
    && Number.isFinite(Number(item?.midi ?? item?.pitch));
}

export function trimPianoJsonTarget(payload, {
  startSeconds = 0,
  endSeconds = Number.POSITIVE_INFINITY,
  reason = 'Human-reviewed trusted training excerpt.',
} = {}) {
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.notes)) {
    throw new Error('Input must be a JSON object containing a notes array.');
  }
  if (!Number.isFinite(startSeconds) || startSeconds < 0) {
    throw new Error('startSeconds must be a non-negative finite number.');
  }
  if (!(Number.isFinite(endSeconds) || endSeconds === Number.POSITIVE_INFINITY)
      || endSeconds <= startSeconds) {
    throw new Error('endSeconds must be greater than startSeconds.');
  }

  const notes = payload.notes
    .filter(validNote)
    .filter((item) => Number(item.time) >= startSeconds && Number(item.time) < endSeconds)
    .map((item) => {
      const onset = Number(item.time);
      const maximumDuration = Number.isFinite(endSeconds)
        ? Math.max(0.01, endSeconds - onset)
        : Number(item.duration);
      return {
        ...item,
        time: Number((onset - startSeconds).toFixed(6)),
        duration: Number(Math.min(Math.max(0.01, Number(item.duration)), maximumDuration).toFixed(6)),
      };
    })
    .sort((left, right) => left.time - right.time || Number(left.midi ?? left.pitch) - Number(right.midi ?? right.pitch));

  if (!notes.length) throw new Error('The trusted excerpt contains no valid notes.');

  return {
    ...payload,
    title: `${payload.title || 'Piano target'} - trusted excerpt`,
    notes,
    trainingExcerpt: {
      schema: 'polymath-human-reviewed-training-excerpt-v1',
      sourceTitle: payload.title || null,
      startSeconds,
      endSeconds: Number.isFinite(endSeconds) ? endSeconds : null,
      sourceNoteCount: payload.notes.length,
      retainedNoteCount: notes.length,
      excludedNoteCount: payload.notes.length - notes.length,
      timeRebasedToZero: startSeconds !== 0,
      reason,
      rule: 'Only notes whose onset lies inside [startSeconds, endSeconds) are eligible; crossing durations are clipped at the trusted boundary.',
    },
  };
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.input || !args.output) {
    throw new Error('Usage: --input target.json --output trusted.json [--start-time 0] --end-time seconds [--reason text]');
  }
  const startSeconds = boundary(args['start-time'], '--start-time', 0);
  const endSeconds = boundary(args['end-time'], '--end-time', Number.POSITIVE_INFINITY);
  const source = path.resolve(args.input);
  const destination = path.resolve(args.output);
  const payload = JSON.parse(await fs.readFile(source, 'utf8'));
  const result = trimPianoJsonTarget(payload, {
    startSeconds,
    endSeconds,
    reason: args.reason || 'Human-reviewed trusted training excerpt.',
  });
  await fs.mkdir(path.dirname(destination), { recursive: true });
  const temporary = `${destination}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  await fs.rename(temporary, destination);
  process.stdout.write(`${JSON.stringify({
    output: destination,
    ...result.trainingExcerpt,
  }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`Piano JSON trim failed: ${error.message}\n`);
    process.exitCode = 1;
  });
}
