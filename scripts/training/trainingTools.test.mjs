import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import ToneMidi from '@tonejs/midi';

const { Midi } = ToneMidi;
const run = promisify(execFile);
const reviewer = path.resolve('scripts/training/reviewAlignmentWindows.mjs');
const reducer = path.resolve('scripts/training/buildPianoReductionTarget.mjs');
const composer = path.resolve('scripts/training/composePreparedDataset.mjs');

test('alignment quality gate writes a new package and excludes weak windows', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-quality-gate-'));
  const input = path.join(root, 'input.json');
  const output = path.join(root, 'output.json');
  const fixture = {
    schema: 'polymath-aligned-training-labels-v1',
    review: { readyForTraining: true },
    alignment: {
      qualityWindows: [
        { id: 'good', matchedNotes: 12, matchedPercent: 80, exactPitchPercent: 90, medianResidualMs: 100, p95ResidualMs: 300, localTempoDifferencePercent: 1, structuralSimilarity: 0.95, status: 'trusted', trainingEligible: true },
        { id: 'bad', matchedNotes: 2, matchedPercent: 10, exactPitchPercent: 20, medianResidualMs: 1200, p95ResidualMs: 3000, localTempoDifferencePercent: 15, structuralSimilarity: 0.4, status: 'trusted', trainingEligible: true },
      ],
    },
    notes: [
      { midi: 60, time: 0, duration: 1, qualityWindowId: 'good', qualityStatus: 'trusted', trainingEligible: true },
      { midi: 61, time: 5, duration: 1, qualityWindowId: 'bad', qualityStatus: 'trusted', trainingEligible: true },
    ],
  };
  await fs.writeFile(input, JSON.stringify(fixture));
  await run(process.execPath, [reviewer, '--input', input, '--output', output]);
  const reviewed = JSON.parse(await fs.readFile(output, 'utf8'));
  const unchanged = JSON.parse(await fs.readFile(input, 'utf8'));
  assert.equal(reviewed.alignment.qualityWindows[0].trainingEligible, true);
  assert.equal(reviewed.alignment.qualityWindows[1].trainingEligible, false);
  assert.deepEqual(reviewed.notes.map((note) => note.trainingEligible), [true, false]);
  assert.equal(reviewed.review.deterministicQualityGate.acceptedWindows, 1);
  assert.equal(unchanged.alignment.qualityWindows[1].trainingEligible, true);
});

test('piano reduction keeps authored piano and selected melody but not drums or guitar', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-piano-target-'));
  const input = path.join(root, 'arrangement.mid');
  const output = path.join(root, 'target.json');
  const midi = new Midi();
  const piano = midi.addTrack();
  piano.instrument.number = 0;
  piano.addNote({ midi: 48, time: 0, duration: 1, velocity: 0.7 });
  const flute = midi.addTrack();
  flute.instrument.number = 73;
  flute.addNote({ midi: 72, time: 0.5, duration: 0.5, velocity: 0.8 });
  const guitar = midi.addTrack();
  guitar.instrument.number = 25;
  guitar.addNote({ midi: 52, time: 0, duration: 1, velocity: 0.7 });
  const drums = midi.addTrack();
  drums.channel = 9;
  drums.addNote({ midi: 36, time: 0, duration: 0.1, velocity: 0.9 });
  await fs.writeFile(input, Buffer.from(midi.toArray()));
  await run(process.execPath, [reducer, '--input', input, '--output', output, '--melody', 'flute']);
  const target = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.deepEqual(target.notes.map((note) => note.midi), [48, 72]);
  assert.ok(target.notes.every((note) => note.instrument === 'acoustic_piano'));
  assert.deepEqual(target.notes.map((note) => note.role), ['piano-accompaniment', 'melody-revoiced-on-piano']);
  assert.equal(target.reduction.selectedTrackCount, 2);
  assert.equal(target.reduction.transposeSemitones, 0);
});

test('piano reduction applies only an explicit whole-semitone key shift', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-piano-transpose-'));
  const input = path.join(root, 'melody.mid');
  const output = path.join(root, 'target.json');
  const midi = new Midi();
  const flute = midi.addTrack();
  flute.instrument.number = 73;
  flute.addNote({ midi: 60, time: 0, duration: 1, velocity: 0.8 });
  await fs.writeFile(input, Buffer.from(midi.toArray()));
  await run(process.execPath, [
    reducer,
    '--input', input,
    '--output', output,
    '--melody', 'flute',
    '--transpose', '3',
  ]);
  const target = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.equal(target.notes[0].midi, 63);
  assert.equal(target.notes[0].note, 'D#4');
  assert.equal(target.reduction.transposeSemitones, 3);
  assert.match(target.reduction.transposeRule, /Explicit/);
});

test('piano reduction can explicitly remove grace-note ornaments by duration', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-piano-ornaments-'));
  const input = path.join(root, 'melody.mid');
  const output = path.join(root, 'target.json');
  const midi = new Midi();
  const flute = midi.addTrack();
  flute.instrument.number = 73;
  flute.addNote({ midi: 61, time: 0, duration: 0.05, velocity: 0.6 });
  flute.addNote({ midi: 62, time: 0.05, duration: 0.5, velocity: 0.8 });
  await fs.writeFile(input, Buffer.from(midi.toArray()));
  await run(process.execPath, [
    reducer,
    '--input', input,
    '--output', output,
    '--melody', 'flute',
    '--minimum-duration', '0.1',
  ]);
  const target = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.deepEqual(target.notes.map((note) => note.midi), [62]);
  assert.equal(target.reduction.minimumDurationSeconds, 0.1);
  assert.match(target.reduction.durationFilterRule, /Explicit/);
});

test('piano reduction can freeze an explicit score excerpt and rebase its clock', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-piano-excerpt-'));
  const input = path.join(root, 'melody.mid');
  const output = path.join(root, 'target.json');
  const midi = new Midi();
  const piano = midi.addTrack();
  piano.instrument.number = 0;
  piano.addNote({ midi: 60, time: 1, duration: 1, velocity: 0.7 });
  piano.addNote({ midi: 62, time: 3, duration: 2, velocity: 0.7 });
  piano.addNote({ midi: 64, time: 5, duration: 1, velocity: 0.7 });
  await fs.writeFile(input, Buffer.from(midi.toArray()));
  await run(process.execPath, [
    reducer,
    '--input', input,
    '--output', output,
    '--start-time', '2',
    '--end-time', '4',
  ]);
  const target = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.deepEqual(target.notes.map((note) => note.midi), [62]);
  assert.equal(target.notes[0].time, 1);
  assert.equal(target.notes[0].duration, 1);
  assert.equal(target.reduction.excerptStartSeconds, 2);
  assert.equal(target.reduction.excerptEndSeconds, 4);
});

test('prepared dataset composition filters songs, copies audio, and writes RunPod paths', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'polymath-compose-dataset-'));
  const sourceRoot = path.join(root, 'source');
  const output = path.join(root, 'output');
  await fs.mkdir(path.join(sourceRoot, 'audio', 'train'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'audio', 'validation'), { recursive: true });
  const trainRecords = [
    { clipId: 'keep-001', songId: 'keep-song', durationSeconds: 5 },
    { clipId: 'drop-001', songId: 'drop-song', durationSeconds: 5 },
  ];
  const validationRecords = [{ clipId: 'valid-001', songId: 'valid-song', durationSeconds: 5 }];
  await fs.writeFile(path.join(root, 'train.jsonl'), `${trainRecords.map(JSON.stringify).join('\n')}\n`);
  await fs.writeFile(path.join(root, 'validation.jsonl'), `${validationRecords.map(JSON.stringify).join('\n')}\n`);
  for (const record of [...trainRecords, ...validationRecords]) {
    const split = record.clipId === 'valid-001' ? 'validation' : 'train';
    await fs.writeFile(path.join(sourceRoot, 'audio', split, `${record.clipId}.wav`), Buffer.alloc(80, 1));
  }
  const spec = path.join(root, 'composition.json');
  await fs.writeFile(spec, JSON.stringify({
    remoteRoot: '/runpod-volume/training/test-v001',
    splits: {
      train: [{ manifest: 'train.jsonl', localRoot: 'source', includeSongPrefixes: ['keep-'] }],
      validation: [{ manifest: 'validation.jsonl', localRoot: 'source' }],
    },
  }));
  await run(process.execPath, [composer, '--spec', spec, '--out', output]);
  const composedTrain = JSON.parse((await fs.readFile(path.join(output, 'prepared-train.jsonl'), 'utf8')).trim());
  assert.equal(composedTrain.clipId, 'keep-001');
  assert.equal(composedTrain.audioClip, '/runpod-volume/training/test-v001/audio/train/keep-001.wav');
  assert.equal(composedTrain.localAudioSource, path.join(output, 'audio', 'train', 'keep-001.wav'));
  assert.equal(await fs.stat(path.join(output, 'audio', 'train', 'keep-001.wav')).then((item) => item.size), 80);
  const summary = JSON.parse(await fs.readFile(path.join(output, 'composition-summary.json'), 'utf8'));
  assert.equal(summary.splits.train.clips, 1);
  assert.equal(summary.splits.validation.clips, 1);
});
