import { normalizeSong } from '../engine/scheduler.js';
import { midiToNote } from '../engine/noteMath.js';
import { parseMidiArrayBuffer } from './midiParser.js';
import { parseMusicXmlText } from './musicXmlParser.js';

function fileExtension(filename = '') {
  return filename.split('.').pop()?.toLowerCase() || '';
}

export async function parseUploadedSongFile(file) {
  const extension = fileExtension(file.name);
  let parsedSong;

  if (extension === 'mid' || extension === 'midi') {
    parsedSong = parseMidiArrayBuffer(await file.arrayBuffer(), file.name);
  } else if (extension === 'musicxml' || extension === 'xml') {
    parsedSong = parseMusicXmlText(await file.text(), file.name);
  } else if (extension === 'mxl') {
    throw new Error('Compressed .mxl is not supported in-browser yet. Export or unzip it as plain .musicxml first, then upload that file.');
  } else if (extension === 'pdf') {
    throw new Error('Use Translate to a Ready-to-Play Sheet for PDF music sheets. PDF files cannot be played directly.');
  } else {
    parsedSong = parseSongText(await file.text(), file.name);
  }

  const relativePath = String(file.webkitRelativePath || '');
  const folderName = relativePath.split('/').filter(Boolean)[0] || '';

  return {
    ...parsedSong,
    sourceFileName: file.name,
    sourceFolderName: parsedSong.sourceFolderName || folderName,
    youtubeSearchQuery:
      parsedSong.youtubeSearchQuery ||
      parsedSong.videoSearchQuery ||
      folderName ||
      '',
  };
}

export function parseSongText(text, filename = 'Uploaded Song') {
  const trimmed = text.trim();
  if (!trimmed) throw new Error('The uploaded file is empty.');

  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return normalizeSong({ title: filename, notes: parsed });
    if (!Array.isArray(parsed.notes) && Array.isArray(parsed.events)) {
      const openStringMidi = [40, 45, 50, 55, 59, 64];
      const notes = parsed.events.flatMap((event, eventIndex) => {
        const time = Number(event.time ?? event.at ?? event.start ?? 0);
        const duration = Number(event.duration ?? event.length ?? 0.7);
        const velocity = Number(event.velocity ?? 0.8);
        if (!Number.isFinite(time) || time < 0) return [];
        if (event.note) return [{ ...event, time, duration, velocity }];
        if (Array.isArray(event.frets)) {
          return event.frets.flatMap((fretValue, stringIndex) => {
            const fret = Number(fretValue);
            if (!Number.isFinite(fret) || fret < 0 || stringIndex >= openStringMidi.length) return [];
            const midi = openStringMidi[stringIndex] + Math.round(fret);
            return [{
              id: `guitar-event-${eventIndex}-${stringIndex}`,
              note: midiToNote(midi),
              time,
              duration,
              velocity,
            }];
          });
        }
        const stringIndex = Number(event.stringIndex ?? event.string);
        const fret = Number(event.fret);
        if (!Number.isInteger(stringIndex) || !Number.isFinite(fret)) return [];
        const zeroBasedString = stringIndex >= 0 && stringIndex <= 5
          ? stringIndex
          : stringIndex >= 1 && stringIndex <= 6
            ? stringIndex - 1
            : -1;
        if (zeroBasedString < 0 || zeroBasedString >= openStringMidi.length || fret < 0) return [];
        return [{
          id: `guitar-event-${eventIndex}`,
          note: midiToNote(openStringMidi[zeroBasedString] + Math.round(fret)),
          time,
          duration,
          velocity,
        }];
      });
      return normalizeSong({ ...parsed, notes });
    }
    return normalizeSong(parsed);
  }

  return parseCsv(trimmed, filename);
}

function splitCsvLine(line) {
  const cells = [];
  let current = '';
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if (char === ',' && !quoted) {
      cells.push(current.trim());
      current = '';
      continue;
    }
    current += char;
  }

  cells.push(current.trim());
  return cells;
}

function parseCsv(text, filename) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const headers = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
  const required = ['note', 'time'];
  for (const name of required) {
    if (!headers.includes(name)) throw new Error(`CSV missing required column: ${name}`);
  }

  const notes = lines.slice(1).map((line, index) => {
    const cells = splitCsvLine(line);
    const row = Object.fromEntries(headers.map((header, i) => [header, cells[i]]));
    return {
      id: row.id || `csv-${index}`,
      note: row.note,
      time: Number(row.time),
      duration: Number(row.duration || row.visualduration || row.audioduration || 0.45),
      visualDuration: row.visualduration ? Number(row.visualduration) : undefined,
      audioDuration: row.audioduration ? Number(row.audioduration) : undefined,
      velocity: Number(row.velocity || 0.82),
      hand: row.hand || undefined,
      measure: row.measure ? Number(row.measure) : undefined,
      source: row.source || 'CSV import',
    };
  });

  return normalizeSong({
    title: filename.replace(/\.[^.]+$/, ''),
    composer: 'CSV/JSON import',
    performance: {
      profile: 'csv-json-import-v9',
      preserveScoreDurations: true,
      preserveScoreTiming: true,
      noOctaveFolding: true,
      sameKeyRetriggerGapSeconds: 0.035,
    },
    notes,
  });
}

export function downloadSongTemplate() {
  const blob = new Blob([
    'note,time,duration,visualDuration,audioDuration,velocity,hand,measure\nEb4,0,0.3,0.3,0.42,0.8,right,1\nG4,0,0.3,0.3,0.42,0.82,right,1\nEb2,0,0.8,0.55,0.95,0.7,left,1\n'
  ], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'falling-piano-v9-template.csv';
  a.click();
  URL.revokeObjectURL(url);
}
