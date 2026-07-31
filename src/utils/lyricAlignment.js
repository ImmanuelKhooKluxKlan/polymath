import { apiRequest, fileToBase64 } from '../services/api.js';
import { midiToNote } from '../engine/noteMath.js';

const SPEECH_SAMPLE_RATE = 16000;
const MAX_MODEL_AUDIO_SECONDS = 12 * 60;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const output = new Float32Array(Math.max(1, Math.floor(input.length * outputRate / inputRate)));
  const ratio = inputRate / outputRate;
  for (let index = 0; index < output.length; index += 1) {
    const source = index * ratio;
    const left = Math.floor(source);
    const mix = source - left;
    output[index] = (input[left] || 0) * (1 - mix) + (input[left + 1] || input[left] || 0) * mix;
  }
  return output;
}

function writeText(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) view.setUint8(offset + index, text.charCodeAt(index));
}

function encodeMonoWav(samples) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeText(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(view, 8, 'WAVE');
  writeText(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SPEECH_SAMPLE_RATE, true);
  view.setUint32(28, SPEECH_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const safe = clamp(sample, -1, 1);
    view.setInt16(44 + index * 2, safe < 0 ? safe * 32768 : safe * 32767, true);
  });
  return buffer;
}

async function mediaToSpeechFile(file) {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (decoded.duration > MAX_MODEL_AUDIO_SECONDS) {
      throw new Error('Model lyric alignment currently supports recordings up to 12 minutes.');
    }
    const mono = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const samples = decoded.getChannelData(channel);
      for (let index = 0; index < samples.length; index += 1) {
        mono[index] += samples[index] / decoded.numberOfChannels;
      }
    }
    const wav = encodeMonoWav(resampleLinear(mono, decoded.sampleRate, SPEECH_SAMPLE_RATE));
    return new globalThis.File([wav], `${file.name.replace(/\.[^.]+$/, '') || 'recording'}-speech.wav`, {
      type: 'audio/wav',
    });
  } finally {
    await context.close().catch(() => {});
  }
}

export async function transcribeTimedLyrics(file, options = {}) {
  const speechFile = await mediaToSpeechFile(file);
  return apiRequest('/api/audio/lyrics', {
    method: 'POST',
    body: JSON.stringify({
      filename: speechFile.name,
      contentBase64: await fileToBase64(speechFile),
      language: options.language || 'en',
      lyricsHint: String(options.lyricsHint || '').slice(0, 4000),
    }),
  });
}

function nearestOctave(midi, reference) {
  return [midi - 24, midi - 12, midi, midi + 12, midi + 24]
    .reduce((best, candidate) => (
      Math.abs(candidate - reference) < Math.abs(best - reference) ? candidate : best
    ), midi);
}

function weightedMedianMidi(notes) {
  if (!notes.length) return null;
  const sorted = [...notes].sort((first, second) => first.midi - second.midi);
  const total = sorted.reduce((sum, note) => sum + Math.max(0.05, note.confidence || 0.5), 0);
  let accumulated = 0;
  for (const note of sorted) {
    accumulated += Math.max(0.05, note.confidence || 0.5);
    if (accumulated >= total / 2) return note.midi;
  }
  return sorted[sorted.length - 1].midi;
}

export function alignSongToLyrics(song, lyricResult) {
  const words = (lyricResult?.words || [])
    .map((word) => ({
      word: String(word.word || '').trim(),
      start: Math.max(0, Number(word.start) || 0),
      end: Math.max(0, Number(word.end) || 0),
    }))
    .filter((word) => word.word && word.end > word.start);
  if (!words.length) throw new Error('The model did not return usable word timestamps.');

  let previousMidi = null;
  let carriedMidi = song.notes[Math.floor(song.notes.length / 2)]?.midi || 60;
  const beatSeconds = 60 / Math.max(40, Number(song.bpm) || 120);
  const notes = words.map((word, wordIndex) => {
    const center = (word.start + word.end) / 2;
    const candidates = song.notes.filter((note) => {
      const noteEnd = note.time + note.duration;
      return note.time <= word.end + 0.08 && noteEnd >= word.start - 0.08;
    });
    if (!candidates.length) {
      const nearest = song.notes.reduce((best, note) => (
        Math.abs(note.time - center) < Math.abs((best?.time ?? Infinity) - center) ? note : best
      ), null);
      if (nearest && Math.abs(nearest.time - center) <= 0.4) candidates.push(nearest);
    }
    let midi = weightedMedianMidi(candidates);
    let pitchSource = 'detected-during-word';
    if (!Number.isFinite(midi)) {
      midi = previousMidi ?? carriedMidi;
      pitchSource = 'carried-phrase-pitch';
    }
    if (previousMidi !== null) midi = nearestOctave(midi, previousMidi);
    if (previousMidi !== null && Math.abs(midi - previousMidi) > 7) {
      midi = previousMidi + clamp(midi - previousMidi, -7, 7);
    }
    midi = Math.round(midi);
    carriedMidi = midi;
    previousMidi = midi;
    const averageVelocity = candidates.length
      ? candidates.reduce((sum, note) => sum + (note.velocity || 0.75), 0) / candidates.length
      : 0.7;
    return {
      id: `lyric-word-${wordIndex}`,
      note: midiToNote(midi),
      midi,
      time: Number(word.start.toFixed(3)),
      duration: Number(Math.max(0.1, (word.end - word.start) * 0.92).toFixed(3)),
      velocity: Number(clamp(averageVelocity, 0.42, 0.9).toFixed(3)),
      confidence: candidates.length
        ? Number((candidates.reduce((sum, note) => sum + (note.confidence || 0.4), 0) / candidates.length).toFixed(3))
        : 0.2,
      lyric: word.word,
      wordStart: word.start,
      wordEnd: word.end,
      lyricBeat: Number((word.start / beatSeconds).toFixed(3)),
      lyricBeatOffset: Number(((word.start / beatSeconds) - Math.round(word.start / beatSeconds)).toFixed(3)),
      pitchSource,
      source: 'model-word-aligned-audio-analysis',
    };
  });

  return {
    ...song,
    notes,
    lyrics: {
      text: lyricResult.text || words.map((word) => word.word).join(' '),
      language: lyricResult.language || '',
      words,
      provider: lyricResult.provider || 'OpenAI',
      model: lyricResult.model || 'whisper-1',
      hintAlignment: lyricResult.lyricHintAlignment || null,
      tempoTracking: {
        bpm: Number(song.bpm) || 120,
        beatSeconds: Number(beatSeconds.toFixed(4)),
        wordsPerMinute: Number((words.length / Math.max(1, words[words.length - 1].end) * 60).toFixed(1)),
        strategy: 'preserve model word onset; record position relative to detected song beat',
      },
    },
    transcription: {
      ...song.transcription,
      mode: 'model-word-aligned-vocal-draft',
      lyricWordCount: words.length,
      pitchStrategy: 'one stabilized instrument note per timed lyric word',
      limitations: [
        'Word timing is model-generated and can be wrong when vocals are masked by the band.',
        'Pitch still comes from the dominant-pitch detector until vocal stem separation is installed.',
        ...(song.transcription?.limitations || []),
      ],
    },
  };
}
