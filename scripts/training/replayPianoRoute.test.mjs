import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const script = fileURLToPath(new URL('./replayPianoRoute.mjs', import.meta.url));

function sourceWave() {
  const rate = 16000;
  const frames = rate * 2;
  const wave = Buffer.alloc(44 + frames * 2);
  wave.write('RIFF', 0);
  wave.writeUInt32LE(wave.length - 8, 4);
  wave.write('WAVEfmt ', 8);
  wave.writeUInt32LE(16, 16);
  wave.writeUInt16LE(1, 20);
  wave.writeUInt16LE(1, 22);
  wave.writeUInt32LE(rate, 24);
  wave.writeUInt32LE(rate * 2, 28);
  wave.writeUInt16LE(2, 32);
  wave.writeUInt16LE(16, 34);
  wave.write('data', 36);
  wave.writeUInt32LE(frames * 2, 40);
  for (let frame = 0; frame < frames; frame += 1) {
    const amplitude = frame < rate ? 600 : 12000;
    wave.writeInt16LE(Math.round(amplitude * Math.sin(2 * Math.PI * 440 * frame / rate)), 44 + frame * 2);
  }
  return wave;
}

test('piano replay can reproduce source dynamics without altering note coordinates', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'piano-route-replay-'));
  try {
    const input = path.join(directory, 'raw.json');
    const audio = path.join(directory, 'prepared.wav');
    const output = path.join(directory, 'cleaned.json');
    const notes = [0.25, 0.55, 1.25, 1.55].map((time) => ({
      midi: 60, time, duration: 0.2, velocity: 0.78, instrument: 'acoustic_piano',
    }));
    fs.writeFileSync(input, JSON.stringify({ notes }));
    fs.writeFileSync(audio, sourceWave());
    execFileSync(process.execPath, [script, '--input', input, '--output', output, '--audio', audio]);
    const actual = JSON.parse(fs.readFileSync(output, 'utf8'));
    assert.equal(actual.transcriptionCleanup.sourceDynamicsApplied, true);
    assert.equal(actual.replayProvenance.sourceDynamicsApplied, true);
    assert.match(actual.replayProvenance.sourceAudioSha256, /^[a-f0-9]{64}$/);
    assert.ok(actual.notes[0].velocity < actual.notes[3].velocity);
    assert.deepEqual(actual.notes.map(({ midi, time, duration }) => ({ midi, time, duration })),
      notes.map(({ midi, time, duration }) => ({ midi, time, duration })));
    execFileSync(process.execPath, [script, '--input', input, '--output', output]);
    const legacy = JSON.parse(fs.readFileSync(output, 'utf8'));
    assert.equal(legacy.replayProvenance.sourceAudio, null);
    assert.equal(legacy.replayProvenance.sourceDynamicsApplied, false);
    assert.match(legacy.replayProvenance.onsetClockTreatment, /no tempo or onset-delay/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('explicit missing replay audio fails instead of silently dropping source dynamics', () => {
  assert.throws(() => execFileSync(process.execPath, [script,
    '--input', 'unused.json', '--output', 'unused-output.json',
    '--audio', path.join(os.tmpdir(), 'missing-polymath-replay-source.wav'),
  ], { stdio: 'pipe' }), /Piano route replay failed/);
});
