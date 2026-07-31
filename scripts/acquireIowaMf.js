#!/usr/bin/env node
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const outDir = path.join(projectRoot, 'public', 'samples', 'iowa-mf');
const sourceDir = path.join(projectRoot, 'public', 'samples', '_iowa-mf-aiff-source');
const manifestPath = path.join(projectRoot, 'src', 'data', 'iowaSampleManifest.js');

const BASE_URL = 'https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano';
const GRAND_START_MIDI = 21; // A0
const GRAND_END_MIDI = 108; // C8
const FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const NATURAL_MIDI_INDEX = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function positiveMod(value, mod) {
  return ((value % mod) + mod) % mod;
}

function midiToFlatName(midi) {
  const octave = Math.floor(midi / 12) - 1;
  return `${FLAT_NAMES[positiveMod(midi, 12)]}${octave}`;
}

function parseNote(note) {
  const text = String(note).trim().replace('♭', 'b').replace('♯', '#');
  const match = /^([A-G])(#|b)?(-?\d+)$/.exec(text);
  if (!match) throw new Error(`Invalid note: ${note}`);
  const [, letter, accidental = '', octaveText] = match;
  let semitone = NATURAL_MIDI_INDEX[letter];
  if (accidental === '#') semitone += 1;
  if (accidental === 'b') semitone -= 1;
  return ((Number(octaveText) + 1) * 12) + semitone;
}

function getArg(name, fallback = null) {
  const prefix = `--${name}=`;
  const match = process.argv.find((arg) => arg.startsWith(prefix));
  return match ? match.slice(prefix.length) : fallback;
}

function parseRange(rangeText = 'A0-C8') {
  const [startText, endText] = rangeText.split('-');
  const startMidi = parseNote(startText || 'A0');
  const endMidi = parseNote(endText || 'C8');
  return {
    startMidi: Math.max(GRAND_START_MIDI, Math.min(startMidi, endMidi)),
    endMidi: Math.min(GRAND_END_MIDI, Math.max(startMidi, endMidi)),
  };
}

function readUInt32BE(buffer, offset) {
  return buffer.readUInt32BE(offset);
}

function readExtended80(buffer, offset) {
  const sign = (buffer[offset] & 0x80) ? -1 : 1;
  const exponent = (((buffer[offset] & 0x7f) << 8) | buffer[offset + 1]);
  const hiMant = buffer.readUInt32BE(offset + 2);
  const loMant = buffer.readUInt32BE(offset + 6);
  if (exponent === 0 && hiMant === 0 && loMant === 0) return 0;
  if (exponent === 0x7fff) return Infinity;
  const unbiased = exponent - 16383;
  return sign * ((hiMant * Math.pow(2, unbiased - 31)) + (loMant * Math.pow(2, unbiased - 63)));
}

function parseAiff(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'FORM') {
    throw new Error('Not an AIFF/FORM file');
  }
  const formType = buffer.toString('ascii', 8, 12);
  if (formType !== 'AIFF' && formType !== 'AIFC') {
    throw new Error(`Unsupported FORM type: ${formType}`);
  }

  let channels = 0;
  let sampleFrames = 0;
  let bitsPerSample = 0;
  let sampleRate = 44100;
  let soundData = null;
  let offset = 12;

  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = readUInt32BE(buffer, offset + 4);
    const dataStart = offset + 8;
    const dataEnd = dataStart + size;

    if (id === 'COMM') {
      channels = buffer.readUInt16BE(dataStart);
      sampleFrames = buffer.readUInt32BE(dataStart + 2);
      bitsPerSample = buffer.readUInt16BE(dataStart + 6);
      const parsedRate = readExtended80(buffer, dataStart + 8);
      if (Number.isFinite(parsedRate) && parsedRate > 0) sampleRate = Math.round(parsedRate);
    } else if (id === 'SSND') {
      const soundOffset = buffer.readUInt32BE(dataStart);
      soundData = buffer.subarray(dataStart + 8 + soundOffset, dataEnd);
    }

    offset = dataEnd + (size % 2);
  }

  if (!channels || !sampleFrames || bitsPerSample !== 16 || !soundData) {
    throw new Error(`Unsupported AIFF details: ${channels}ch, ${sampleFrames} frames, ${bitsPerSample} bit, sound=${Boolean(soundData)}`);
  }

  const frameBytes = channels * 2;
  const expectedBytes = sampleFrames * frameBytes;
  const usableData = soundData.subarray(0, Math.min(soundData.length, expectedBytes));
  const pcmLE = Buffer.alloc(usableData.length);
  for (let i = 0; i + 1 < usableData.length; i += 2) {
    const sample = usableData.readInt16BE(i);
    pcmLE.writeInt16LE(sample, i);
  }

  return { channels, sampleFrames: Math.floor(pcmLE.length / frameBytes), bitsPerSample, sampleRate, pcmLE };
}

function trimAndNormalizePcm16({ channels, sampleRate, pcmLE }) {
  const frameBytes = channels * 2;
  const totalFrames = Math.floor(pcmLE.length / frameBytes);
  const threshold = Math.round(32767 * 0.0018);
  let firstFrame = 0;
  let lastFrame = totalFrames - 1;
  let found = false;

  for (let frame = 0; frame < totalFrames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const value = Math.abs(pcmLE.readInt16LE((frame * frameBytes) + (channel * 2)));
      if (value >= threshold) {
        firstFrame = frame;
        found = true;
        break;
      }
    }
    if (found) break;
  }

  found = false;
  for (let frame = totalFrames - 1; frame >= 0; frame -= 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const value = Math.abs(pcmLE.readInt16LE((frame * frameBytes) + (channel * 2)));
      if (value >= threshold) {
        lastFrame = frame;
        found = true;
        break;
      }
    }
    if (found) break;
  }

  const prerollFrames = Math.round(sampleRate * 0.004);
  const tailFrames = Math.round(sampleRate * 0.42);
  const startFrame = Math.max(0, firstFrame - prerollFrames);
  const endFrame = Math.min(totalFrames, lastFrame + tailFrames);
  const trimmed = Buffer.from(pcmLE.subarray(startFrame * frameBytes, endFrame * frameBytes));

  let peak = 0;
  for (let i = 0; i + 1 < trimmed.length; i += 2) {
    peak = Math.max(peak, Math.abs(trimmed.readInt16LE(i)));
  }

  const targetPeak = Math.round(32767 * 0.89);
  const gain = peak ? Math.min(targetPeak / peak, 2.15) : 1;

  if (Math.abs(gain - 1) > 0.001) {
    for (let i = 0; i + 1 < trimmed.length; i += 2) {
      const value = Math.round(trimmed.readInt16LE(i) * gain);
      trimmed.writeInt16LE(Math.max(-32768, Math.min(32767, value)), i);
    }
  }

  return { pcmLE: trimmed, sampleFrames: Math.floor(trimmed.length / frameBytes), gain, peak };
}

function writeWav({ channels, sampleRate, bitsPerSample, pcmLE }) {
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);
  const riffSize = 36 + pcmLE.length;
  const output = Buffer.alloc(44 + pcmLE.length);

  output.write('RIFF', 0, 'ascii');
  output.writeUInt32LE(riffSize, 4);
  output.write('WAVE', 8, 'ascii');
  output.write('fmt ', 12, 'ascii');
  output.writeUInt32LE(16, 16);
  output.writeUInt16LE(1, 20);
  output.writeUInt16LE(channels, 22);
  output.writeUInt32LE(sampleRate, 24);
  output.writeUInt32LE(byteRate, 28);
  output.writeUInt16LE(blockAlign, 32);
  output.writeUInt16LE(bitsPerSample, 34);
  output.write('data', 36, 'ascii');
  output.writeUInt32LE(pcmLE.length, 40);
  pcmLE.copy(output, 44);
  return output;
}

async function downloadFile(url, destination) {
  const response = await fetch(url, {
    headers: { 'user-agent': 'FallingPianoPro/1.0 sample-acquisition' },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  await writeFile(destination, Buffer.from(arrayBuffer));
}

async function writeManifest() {
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await mkdir(outDir, { recursive: true });
  const files = await readdir(outDir).catch(() => []);
  const notes = files
    .filter((file) => file.toLowerCase().endsWith('.wav'))
    .map((file) => path.basename(file, path.extname(file)))
    .sort((a, b) => parseNote(a) - parseNote(b));

  const content = `// Generated from public/samples/iowa-mf. Re-run \`npm run acquire:iowa-88\` after adding samples.\nexport const IOWA_MF_SAMPLES = ${JSON.stringify(notes, null, 2)};\n`;
  await writeFile(manifestPath, content);
  return notes;
}

async function main() {
  await mkdir(outDir, { recursive: true });
  await mkdir(sourceDir, { recursive: true });

  const rangeText = getArg('range', 'A0-C8');
  const { startMidi, endMidi } = parseRange(rangeText);
  const notes = [];
  for (let midi = startMidi; midi <= endMidi; midi += 1) notes.push(midiToFlatName(midi));

  console.log(`Acquiring University of Iowa Steinway Model B mf samples (${notes[0]}-${notes.at(-1)}).`);
  console.log('Existing WAV files are skipped. The manifest is regenerated at the end.');

  const failures = [];

  for (const note of notes) {
    const aiffName = `Piano.mf.${note}.aiff`;
    const wavName = `${note}.wav`;
    const url = `${BASE_URL}/${aiffName}`;
    const aiffPath = path.join(sourceDir, aiffName);
    const wavPath = path.join(outDir, wavName);

    if (existsSync(wavPath)) {
      console.log(`✓ ${wavName} already exists`);
      continue;
    }

    try {
      if (!existsSync(aiffPath)) {
        process.stdout.write(`↓ ${aiffName} ... `);
        await downloadFile(url, aiffPath);
        console.log('downloaded');
      } else {
        console.log(`• ${aiffName} already downloaded`);
      }

      const aiff = await readFile(aiffPath);
      const parsed = parseAiff(aiff);
      const processed = trimAndNormalizePcm16(parsed);
      const wav = writeWav({
        channels: parsed.channels,
        sampleRate: parsed.sampleRate,
        bitsPerSample: parsed.bitsPerSample,
        pcmLE: processed.pcmLE,
      });
      await writeFile(wavPath, wav);
      console.log(`✓ wrote ${wavName} (${parsed.channels}ch/${parsed.sampleRate}Hz, gain ${processed.gain.toFixed(2)})`);
    } catch (error) {
      failures.push({ note, error });
      console.error(`✗ ${note} failed: ${error.message}`);
    }
  }

  const installed = await writeManifest();
  console.log(`\nManifest updated with ${installed.length} local WAV samples.`);
  console.log('Restart `npm run dev` if it is already open.');

  if (failures.length) {
    console.error(`\n${failures.length} sample(s) failed. Re-run the command later; already downloaded files will be skipped.`);
    process.exitCode = 1;
  } else {
    console.log('Done.');
  }
}

main().catch((error) => {
  console.error('\nSample acquisition failed.');
  console.error(error);
  process.exitCode = 1;
});
