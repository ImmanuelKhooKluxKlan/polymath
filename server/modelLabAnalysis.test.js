const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { analyzeTranscription, midiToNote } = require('./modelLabAnalysis');
const {
  archiveAlignmentSnapshot,
  archiveRawTestSnapshot,
  isLoopback,
  listAlignmentArchives,
  listRawTestArchives,
  loadAlignmentArchive,
  loadRawTestArchive,
  readWavDurationSeconds,
} = require('./modelLab');

test('Model Lab reports instruments, pitch, MIDI, polyphony, and rapid repeats from raw notes', () => {
  const result = analyzeTranscription({
    transcriptionProvider: 'Tester v001',
    notes: [
      { instrument: 'acoustic_piano', midi: 60, time: 0, duration: 0.5, velocity: 0.78 },
      { instrument: 'acoustic_piano', midi: 60, time: 0.05, duration: 0.4, velocity: 0.78 },
      { instrument: 'acoustic_guitar', midi: 52, time: 0, duration: 1.1, velocity: 0.78 },
      { instrument: 'voice', midi: 67, time: 0.02, duration: 0.7, velocity: 0.78 },
    ],
  });

  assert.equal(result.headline.validNotes, 4);
  assert.equal(result.headline.detectedInstrumentGroups, 3);
  assert.equal(result.headline.pitchRange, 'E3–G4');
  assert.equal(result.pitch.minimumMidi, 52);
  assert.equal(result.pitch.maximumMidi, 67);
  assert.equal(result.midi.noteOnEvents, 4);
  assert.equal(result.midi.noteOffEvents, 4);
  assert.equal(result.timing.rapidRepeats75ms, 1);
  assert.equal(result.timing.samePitchOverlaps, 1);
  assert.equal(result.timing.maximumPolyphony, 4);
  assert.equal(result.instruments[0].instrument, 'acoustic_piano');
  assert.equal(result.model.cleanupApplied, false);
});

test('Model Lab rejects malformed notes without corrupting valid statistics', () => {
  const result = analyzeTranscription({
    notes: [
      { instrument: 'piano', midi: 21, time: 0, duration: 1 },
      { instrument: 'piano', midi: 130, time: 0, duration: 1 },
      { instrument: 'piano', midi: 22, time: -1, duration: 1 },
      { instrument: 'piano', midi: 23, time: 0, duration: 0 },
    ],
  });

  assert.equal(result.headline.validNotes, 1);
  assert.equal(result.headline.rejectedMalformedNotes, 3);
  assert.equal(result.headline.pitchRange, 'A0–A0');
});

test('MIDI note naming and localhost boundary are explicit', () => {
  assert.equal(midiToNote(21), 'A0');
  assert.equal(midiToNote(60), 'C4');
  assert.equal(midiToNote(108), 'C8');
  assert.equal(isLoopback({ ip: '::1', socket: {} }), true);
  assert.equal(isLoopback({ ip: '::ffff:127.0.0.1', socket: {} }), true);
  assert.equal(isLoopback({ ip: '203.0.113.8', socket: {} }), false);
});

test('Model Lab reads the prepared WAV duration used as the supervision timeline bound', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-wav-'));
  const filename = path.join(directory, 'five-seconds.wav');
  const sampleRate = 16_000;
  const dataBytes = sampleRate * 2 * 5;
  const wav = Buffer.alloc(44 + dataBytes);
  wav.write('RIFF', 0, 'ascii');
  wav.writeUInt32LE(36 + dataBytes, 4);
  wav.write('WAVE', 8, 'ascii');
  wav.write('fmt ', 12, 'ascii');
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(sampleRate * 2, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36, 'ascii');
  wav.writeUInt32LE(dataBytes, 40);
  fs.writeFileSync(filename, wav);
  try {
    assert.equal(readWavDurationSeconds(filename), 5);
  } finally {
    const resolved = path.resolve(directory);
    const temporaryRoot = `${path.resolve(os.tmpdir())}${path.sep}`;
    if (resolved.startsWith(temporaryRoot)) fs.rmSync(resolved, { recursive: true, force: true });
  }
});

test('supervision uploads and analysis are retained in a private, hashed local archive', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-supervision-'));
  const archiveRoot = path.join(directory, 'archive');
  const referencePath = path.join(directory, 'desired.json');
  const observedPath = path.join(directory, 'current.mid');
  fs.writeFileSync(referencePath, '{"notes":[]}\n');
  fs.writeFileSync(observedPath, Buffer.from([0x4d, 0x54, 0x68, 0x64]));
  const record = {
    id: 'alignment-001',
    title: 'Archive test',
    createdAt: '2026-08-27T00:00:00.000Z',
    updatedAt: '2026-08-27T00:01:00.000Z',
    referenceFilename: 'desired.json',
    observedFilename: 'current.mid',
    referenceNotes: [{ midi: 60, time: 0, duration: 1 }],
    observedNotes: [{ midi: 61, time: 0.1, duration: 0.8 }],
  };
  try {
    const archived = archiveAlignmentSnapshot({
      archiveRoot,
      record,
      referenceFile: { path: referencePath, originalname: 'desired.json' },
      observedFile: { path: observedPath, originalname: 'current.mid' },
      analysis: { metrics: { exactPitchPercent: 0 } },
    });
    assert.equal(archived.manifest.reference.noteCount, 1);
    assert.equal(archived.manifest.observed.noteCount, 1);
    assert.equal(archived.manifest.reference.sha256.length, 64);
    assert.equal(fs.existsSync(path.join(archived.directory, 'desired-reference.json')), true);
    assert.equal(fs.existsSync(path.join(archived.directory, 'model-output.mid')), true);
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(archived.directory, 'analysis.json'))), {
      metrics: { exactPitchPercent: 0 },
    });
    assert.equal(JSON.parse(fs.readFileSync(path.join(archiveRoot, 'latest.json'))).id, record.id);
  } finally {
    const resolved = path.resolve(directory);
    const temporaryRoot = `${path.resolve(os.tmpdir())}${path.sep}`;
    if (resolved.startsWith(temporaryRoot)) fs.rmSync(resolved, { recursive: true, force: true });
  }
});

test('private supervision history survives process memory and reloads its full analysis', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-supervision-history-'));
  const archiveRoot = path.join(directory, 'archive');
  const record = {
    id: 'history-001',
    title: 'History song',
    createdAt: '2026-08-27T00:00:00.000Z',
    updatedAt: '2026-08-27T00:02:00.000Z',
    referenceFilename: 'ideal.mid',
    observedFilename: 'model.json',
    referenceNotes: [{ midi: 60, time: 0, duration: 1 }],
    observedNotes: [{ midi: 60, time: 0.03, duration: 0.9 }],
  };
  const analysis = {
    id: record.id,
    metrics: {
      matchedReferencePercent: 100,
      exactPitchPercent: 100,
      medianTimingResidualMs: 30,
      trainingEligiblePercent: 100,
      verdict: 'candidate',
    },
    supervisionPackage: { review: { readyForTraining: true } },
  };
  try {
    archiveAlignmentSnapshot({ archiveRoot, record, analysis });
    assert.deepEqual(loadAlignmentArchive(archiveRoot, record.id).analysis, analysis);
    assert.deepEqual(listAlignmentArchives(archiveRoot), [{
      id: record.id,
      kind: 'supervision',
      title: record.title,
      createdAt: record.createdAt,
      updatedAt: record.updatedAt,
      referenceFilename: record.referenceFilename,
      observedFilename: record.observedFilename,
      referenceNotes: 1,
      observedNotes: 1,
      matchedPercent: 100,
      exactPitchPercent: 100,
      timingResidualMs: 30,
      trainingEligiblePercent: 100,
      readyForTraining: true,
      verdict: 'candidate',
    }]);
    assert.equal(loadAlignmentArchive(archiveRoot, '../outside'), null);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('raw model tests retain events and analytics but deliberately discard source media', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-raw-history-'));
  const job = {
    id: 'raw-001',
    title: 'Raw history song',
    filename: 'private-song.mp3',
    checkpoint: 'muscriptor-tester/v001',
    createdAt: '2026-08-27T00:00:00.000Z',
    completedAt: '2026-08-27T00:01:00.000Z',
    sourceDurationSeconds: 12.5,
    result: {
      raw: { notes: [{ midi: 60, time: 0, duration: 1 }] },
      analysis: { headline: { validNotes: 1, detectedInstrumentGroups: 1, rapidRepeats75ms: 0 } },
    },
  };
  try {
    const archived = archiveRawTestSnapshot({ archiveRoot: directory, job });
    assert.equal(archived.manifest.sourceMediaRetained, false);
    assert.equal(archived.manifest.rawOutput.sha256.length, 64);
    assert.deepEqual(loadRawTestArchive(directory, job.id).raw, job.result.raw);
    const history = listRawTestArchives(directory);
    assert.equal(history.length, 1);
    assert.equal(history[0].noteCount, 1);
    assert.equal(history[0].sourceMediaRetained, false);
    assert.equal(fs.existsSync(path.join(directory, job.id, 'private-song.mp3')), false);
    fs.appendFileSync(path.join(directory, job.id, 'raw-model-output.json'), 'tampered');
    assert.equal(loadRawTestArchive(directory, job.id), null);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
