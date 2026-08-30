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

async function readJsonl(filename) {
  const content = await fs.readFile(filename, 'utf8');
  return content.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function accepted(record, source) {
  const songId = String(record.songId || '');
  const include = Array.isArray(source.includeSongPrefixes) ? source.includeSongPrefixes : [];
  const exclude = Array.isArray(source.excludeSongPrefixes) ? source.excludeSongPrefixes : [];
  return (!include.length || include.some((prefix) => songId.startsWith(prefix)))
    && !exclude.some((prefix) => songId.startsWith(prefix));
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  if (!args.spec || !args.out) {
    throw new Error('Usage: node scripts/training/composePreparedDataset.mjs --spec composition.json --out runpod-upload');
  }
  const specPath = path.resolve(args.spec);
  const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
  const output = path.resolve(args.out);
  const remoteRoot = String(spec.remoteRoot || '').replace(/\/+$/, '');
  if (!remoteRoot.startsWith('/runpod-volume/training/')) {
    throw new Error('spec.remoteRoot must be a /runpod-volume/training/... path.');
  }

  const seenClips = new Set();
  const songSplits = new Map();
  const summary = { schema: 'polymath-prepared-dataset-composition-v1', spec: specPath, remoteRoot, splits: {} };
  for (const split of ['train', 'validation']) {
    const records = [];
    const sources = Array.isArray(spec.splits?.[split]) ? spec.splits[split] : [];
    for (const source of sources) {
      const manifest = path.resolve(path.dirname(specPath), source.manifest);
      const localRoot = path.resolve(path.dirname(specPath), source.localRoot);
      for (const record of await readJsonl(manifest)) {
        if (!accepted(record, source)) continue;
        const clipId = String(record.clipId || '');
        const songId = String(record.songId || '');
        if (!clipId || seenClips.has(clipId)) throw new Error(`Missing or duplicate clipId: ${clipId}`);
        if (!songId) throw new Error(`${clipId}: songId is missing`);
        const previousSplit = songSplits.get(songId);
        if (previousSplit && previousSplit !== split) {
          throw new Error(`${songId}: song leakage between ${previousSplit} and ${split}`);
        }
        songSplits.set(songId, split);
        seenClips.add(clipId);

        const sourceAudio = path.join(localRoot, 'audio', split, `${clipId}.wav`);
        const destinationAudio = path.join(output, 'audio', split, `${clipId}.wav`);
        const stat = await fs.stat(sourceAudio).catch(() => null);
        if (!stat?.isFile() || stat.size <= 44) throw new Error(`${clipId}: local prepared WAV is missing`);
        await fs.mkdir(path.dirname(destinationAudio), { recursive: true });
        await fs.copyFile(sourceAudio, destinationAudio);
        records.push({
          ...record,
          split,
          audioClip: `${remoteRoot}/audio/${split}/${clipId}.wav`,
          compositionSourceManifest: manifest,
        });
      }
    }
    if (!records.length) throw new Error(`No ${split} records were selected.`);
    await fs.mkdir(output, { recursive: true });
    await fs.writeFile(
      path.join(output, `prepared-${split}.jsonl`),
      `${records.map((record) => JSON.stringify(record)).join('\n')}\n`,
    );
    summary.splits[split] = {
      clips: records.length,
      songs: [...new Set(records.map(({ songId }) => songId))].sort(),
      audioSeconds: records.reduce((sum, record) => sum + Number(record.durationSeconds || 0), 0),
    };
  }
  await fs.writeFile(path.join(output, 'composition-summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Prepared dataset composition failed: ${error.message}\n`);
  process.exitCode = 1;
});
