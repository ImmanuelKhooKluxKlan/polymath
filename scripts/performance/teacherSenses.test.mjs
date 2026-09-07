import assert from 'node:assert/strict';
import test from 'node:test';
import {
  analyzeAudioFrame,
  analyzeSpectralNotes,
  frequencyToPitch,
} from '../../src/engine/teacherHearingEngine.js';
import {
  buildKeyboardGeometry,
  detectKeyMotion,
  extractPerspectiveGray,
  findKeyboardCandidate,
  regionToCorners,
} from '../../src/engine/teacherVisionEngine.js';

function sineWave(frequency, sampleRate = 48000, length = 4096, amplitude = 0.35) {
  return Float32Array.from({ length }, (_, index) => (
    Math.sin((2 * Math.PI * frequency * index) / sampleRate) * amplitude
  ));
}

function rgbaFrame(width, height, painter) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = painter(x, y);
      const offset = ((y * width) + x) * 4;
      data[offset] = value;
      data[offset + 1] = value;
      data[offset + 2] = value;
      data[offset + 3] = 255;
    }
  }
  return { data };
}

test('hearing converts concert A to A4 and resolves a clean waveform', () => {
  assert.deepEqual(frequencyToPitch(440), {
    frequency: 440,
    midi: 69,
    note: 'A4',
    cents: 0,
  });
  const reading = analyzeAudioFrame(sineWave(440), 48000);
  assert.equal(reading.note, 'A4');
  assert.ok(Math.abs(reading.frequency - 440) < 2, `resolved ${reading.frequency} Hz`);
  assert.ok(reading.confidence > 0.8);
});

test('hearing covers the low and high endpoints of an 88-key piano', () => {
  const low = analyzeAudioFrame(sineWave(27.5), 48000);
  const high = analyzeAudioFrame(sineWave(4186.009), 48000);
  assert.equal(low.note, 'A0');
  assert.equal(high.note, 'C8');
});

test('hearing resolves every clean sine pitch across all 88 piano keys', () => {
  for (let midi = 21; midi <= 108; midi += 1) {
    const frequency = 440 * (2 ** ((midi - 69) / 12));
    const reading = analyzeAudioFrame(sineWave(frequency), 48000);
    assert.equal(reading.midi, midi, `${frequency.toFixed(2)} Hz resolved as ${reading.note}`);
  }
});

test('hearing leaves silence unclassified', () => {
  const reading = analyzeAudioFrame(new Float32Array(4096), 48000);
  assert.equal(reading.heard, false);
  assert.equal(reading.note, '—');
  assert.equal(reading.frequency, 0);
});

test('hearing exposes several spectral peaks for a clean major chord', () => {
  const sampleRate = 48000;
  const frequencies = [261.6256, 329.6276, 391.9954];
  const chord = Float32Array.from({ length: 4096 }, (_, index) => (
    frequencies.reduce((sum, frequency) => sum + Math.sin((2 * Math.PI * frequency * index) / sampleRate), 0) * 0.12
  ));
  const notes = analyzeSpectralNotes(chord, sampleRate, { maximumNotes: 6 }).map((entry) => entry.note);
  assert.ok(notes.includes('C4'), `peaks: ${notes.join(', ')}`);
  assert.ok(notes.includes('E4'), `peaks: ${notes.join(', ')}`);
  assert.ok(notes.includes('G4'), `peaks: ${notes.join(', ')}`);
});

test('vision ranks a keyboard-like stripe field above a flat scene', () => {
  const width = 320;
  const height = 180;
  const flat = rgbaFrame(width, height, () => 120);
  const keyboard = rgbaFrame(width, height, (x, y) => {
    if (y < 55 || y > 145) return 92;
    if (y < 108) return x % 22 < 7 ? 28 : 220;
    return x % 22 < 2 ? 68 : 224;
  });
  const flatResult = findKeyboardCandidate(flat, width, height);
  const keyboardResult = findKeyboardCandidate(keyboard, width, height);
  assert.ok(keyboardResult.confidence > flatResult.confidence + 0.2);
  assert.equal(keyboardResult.detected, true);
});

test('calibrated motion identifies the changed white-key zone', () => {
  const width = 140;
  const height = 60;
  const baseline = new Uint8Array(width * height).fill(220);
  const current = baseline.slice();
  const geometry = buildKeyboardGeometry({ firstWhiteMidi: 60, whiteKeyCount: 7 });
  const target = geometry.whites[2];
  for (let y = Math.floor(target.yStart * height); y < Math.floor(target.yEnd * height); y += 1) {
    for (let x = Math.floor(target.xStart * width); x < Math.floor(target.xEnd * width); x += 1) {
      current[(y * width) + x] = 78;
    }
  }
  const result = detectKeyMotion({ baseline, current, width, height, geometry });
  assert.equal(result.pressed[0].note, 'E4');
  assert.ok(result.pressed[0].confidence > 0.8);
});

test('uniform camera exposure changes do not look like pressed keys', () => {
  const width = 140;
  const height = 60;
  const baseline = new Uint8Array(width * height).fill(210);
  const darkerFrame = new Uint8Array(width * height).fill(168);
  const geometry = buildKeyboardGeometry({ firstWhiteMidi: 60, whiteKeyCount: 7 });
  const result = detectKeyMotion({ baseline, current: darkerFrame, width, height, geometry });
  assert.equal(result.pressed.length, 0);
  assert.ok(Math.abs(result.globalShift + 42) < 0.1);
});

test('keyboard geometry includes black keys only where semitones exist', () => {
  const geometry = buildKeyboardGeometry({ firstWhiteMidi: 60, whiteKeyCount: 8 });
  assert.equal(geometry.whites[0].note, 'C4');
  assert.equal(geometry.whites[7].note, 'C5');
  assert.deepEqual(geometry.blacks.map((key) => key.note), ['C#4', 'D#4', 'F#4', 'G#4', 'A#4']);
});

test('perspective extraction produces a stable flattened keyboard plane', () => {
  const width = 100;
  const height = 80;
  const frame = rgbaFrame(width, height, (x) => Math.min(255, x * 2));
  const corners = regionToCorners({ x: 0.2, y: 0.25, width: 0.6, height: 0.5 });
  const flattened = extractPerspectiveGray(frame, width, height, corners, 48, 20);
  assert.equal(flattened.width, 48);
  assert.equal(flattened.height, 20);
  assert.ok(flattened.gray[0] < 55);
  assert.ok(flattened.gray[47] > 140);
});
