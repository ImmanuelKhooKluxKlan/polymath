import { parseNote } from '../engine/noteMath.js';

function safeFilename(name, extension) {
  const base = String(name || 'song').replace(/[^a-z0-9-_]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'song';
  return `${base}.${extension}`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadSongJson(song) {
  const payload = {
    ...song,
    exportedAt: new Date().toISOString(),
    exportedBy: 'Polymath Musician Studio',
  };
  downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }), safeFilename(song?.title, 'json'));
}

function writeVarLen(value) {
  let buffer = value & 0x7f;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= ((value & 0x7f) | 0x80);
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
  return bytes;
}

function textEvent(type, text) {
  const encoded = Array.from(new TextEncoder().encode(String(text || '').slice(0, 120)));
  return [0xff, type, ...writeVarLen(encoded.length), ...encoded];
}

function uint32(value) {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function uint16(value) {
  return [(value >>> 8) & 0xff, value & 0xff];
}

export function songToMidiBytes(song) {
  const ticksPerQuarter = 480;
  const bpm = Math.max(20, Math.min(260, Number(song?.bpm || song?.tempo || 100)));
  const usPerQuarter = Math.round(60000000 / bpm);
  const secondsPerTick = (60 / bpm) / ticksPerQuarter;
  const events = [];

  for (const note of song?.notes || []) {
    try {
      const midi = parseNote(note.note).midi;
      const startTick = Math.max(0, Math.round(Number(note.time || 0) / secondsPerTick));
      const durationSeconds = Math.max(0.035, Number(note.audioDuration ?? note.duration ?? 0.35));
      const endTick = Math.max(startTick + 1, Math.round((Number(note.time || 0) + durationSeconds) / secondsPerTick));
      const velocity = Math.max(1, Math.min(127, Math.round(Number(note.velocity || 0.8) * 127)));
      events.push({ tick: startTick, bytes: [0x90, midi, velocity], order: 1 });
      events.push({ tick: endTick, bytes: [0x80, midi, 0], order: 0 });
    } catch {
      // Skip invalid notes.
    }
  }

  for (const pedal of song?.pedals || []) {
    const tick = Math.max(0, Math.round(Number(pedal.time || 0) / secondsPerTick));
    events.push({
      tick,
      bytes: [0xb0, 64, pedal.down ? 127 : 0],
      order: pedal.down ? 0 : 2,
    });
  }

  events.sort((a, b) => (a.tick - b.tick) || (a.order - b.order));

  const track = [];
  track.push(0x00, ...textEvent(0x03, song?.title || 'Polymath Musician export'));
  track.push(0x00, 0xff, 0x51, 0x03, (usPerQuarter >> 16) & 0xff, (usPerQuarter >> 8) & 0xff, usPerQuarter & 0xff);
  track.push(0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08);

  let lastTick = 0;
  for (const event of events) {
    track.push(...writeVarLen(event.tick - lastTick), ...event.bytes);
    lastTick = event.tick;
  }
  track.push(0x00, 0xff, 0x2f, 0x00);

  const header = [
    ...Array.from(new TextEncoder().encode('MThd')),
    ...uint32(6),
    ...uint16(0),
    ...uint16(1),
    ...uint16(ticksPerQuarter),
  ];
  const trackHeader = [
    ...Array.from(new TextEncoder().encode('MTrk')),
    ...uint32(track.length),
  ];
  return new Uint8Array([...header, ...trackHeader, ...track]);
}

export function downloadSongMidi(song) {
  const bytes = songToMidiBytes(song);
  downloadBlob(new Blob([bytes], { type: 'audio/midi' }), safeFilename(song?.title, 'mid'));
}
