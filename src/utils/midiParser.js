import { midiToNote } from '../engine/noteMath.js';
import { normalizeSong } from '../engine/scheduler.js';

const DEFAULT_TEMPO_US_PER_QUARTER = 500000;
const TICKS_PER_QUARTER_FALLBACK = 480;
const MIN_DURATION_SECONDS = 0.035;
const MAX_DURATION_SECONDS = 32;
const PERCUSSION_CHANNEL = 9;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function readAscii(bytes, offset, length) {
  return String.fromCharCode(...bytes.slice(offset, offset + length));
}

function readUint16(bytes, offset) {
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function readUint32(bytes, offset) {
  return (
    bytes[offset] * 0x1000000
    + ((bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3])
  );
}

function readVariableLength(bytes, state) {
  let value = 0;
  for (let index = 0; index < 4; index += 1) {
    if (state.offset >= bytes.length) {
      throw new Error('Unexpected end of MIDI variable-length value.');
    }

    const byte = bytes[state.offset];
    state.offset += 1;
    value = (value << 7) | (byte & 0x7f);
    if ((byte & 0x80) === 0) return value;
  }
  return value;
}

function secondsAtTick(tick, tempoMap, ticksPerQuarter) {
  let seconds = 0;
  let previousTick = 0;
  let previousTempo = DEFAULT_TEMPO_US_PER_QUARTER;

  for (const event of tempoMap) {
    if (event.tick >= tick) break;
    seconds += ((event.tick - previousTick) * previousTempo) / (ticksPerQuarter * 1000000);
    previousTick = event.tick;
    previousTempo = event.usPerQuarter;
  }

  seconds += ((tick - previousTick) * previousTempo) / (ticksPerQuarter * 1000000);
  return seconds;
}

function decodeText(bytes) {
  try {
    return new TextDecoder('utf-8').decode(Uint8Array.from(bytes)).replace(/\0/g, '').trim();
  } catch {
    return String.fromCharCode(...bytes).replace(/\0/g, '').trim();
  }
}

function parseTrack(bytes, trackOffset, trackLength, trackIndex) {
  const end = Math.min(bytes.length, trackOffset + trackLength);
  const state = { offset: trackOffset };
  const events = [];
  const tempos = [];
  const trackNames = [];
  const timeSignatures = [];
  const keySignatures = [];
  let tick = 0;
  let runningStatus = null;

  while (state.offset < end) {
    const delta = readVariableLength(bytes, state);
    tick += delta;

    let status = bytes[state.offset];
    if (status & 0x80) {
      state.offset += 1;
      runningStatus = status;
    } else if (runningStatus !== null) {
      status = runningStatus;
    } else {
      throw new Error(`Bad MIDI running status in track ${trackIndex + 1}.`);
    }

    if (status === 0xff) {
      const type = bytes[state.offset];
      state.offset += 1;
      const length = readVariableLength(bytes, state);
      const data = Array.from(bytes.slice(state.offset, Math.min(end, state.offset + length)));
      state.offset += length;

      if (type === 0x2f) break;
      if (type === 0x51 && data.length >= 3) {
        tempos.push({
          tick,
          usPerQuarter: (data[0] << 16) | (data[1] << 8) | data[2],
        });
      } else if (type === 0x58 && data.length >= 2) {
        timeSignatures.push({
          tick,
          numerator: data[0],
          denominator: 2 ** data[1],
          clocksPerMetronome: data[2] ?? 24,
          thirtySecondsPerQuarter: data[3] ?? 8,
        });
      } else if (type === 0x59 && data.length >= 2) {
        const signedSharpsFlats = data[0] > 127 ? data[0] - 256 : data[0];
        keySignatures.push({
          tick,
          sharpsFlats: signedSharpsFlats,
          minor: data[1] === 1,
        });
      }

      if ((type === 0x03 || type === 0x01 || type === 0x04) && data.length) {
        const text = decodeText(data);
        if (text) trackNames.push(text);
      }
      continue;
    }

    if (status === 0xf0 || status === 0xf7) {
      const length = readVariableLength(bytes, state);
      state.offset += length;
      runningStatus = null;
      continue;
    }

    const command = status & 0xf0;
    const channel = status & 0x0f;

    if (command === 0xc0 || command === 0xd0) {
      const data1 = bytes[state.offset];
      state.offset += 1;
      events.push({ tick, command, channel, data1, data2: 0, trackIndex });
      continue;
    }

    if (state.offset + 1 >= end) break;
    const data1 = bytes[state.offset];
    const data2 = bytes[state.offset + 1];
    state.offset += 2;
    events.push({ tick, command, channel, data1, data2, trackIndex });
  }

  return {
    events,
    tempos,
    trackNames,
    timeSignatures,
    keySignatures,
    endTick: tick,
  };
}

function buildTempoMap(trackTempos) {
  const tempos = trackTempos
    .flat()
    .filter((event) => (
      Number.isFinite(event.tick)
      && Number.isFinite(event.usPerQuarter)
      && event.usPerQuarter > 0
    ))
    .sort((a, b) => a.tick - b.tick);

  if (!tempos.length || tempos[0].tick !== 0) {
    tempos.unshift({ tick: 0, usPerQuarter: DEFAULT_TEMPO_US_PER_QUARTER });
  }

  const deduped = [];
  for (const tempo of tempos) {
    const previous = deduped[deduped.length - 1];
    if (previous && previous.tick === tempo.tick) previous.usPerQuarter = tempo.usPerQuarter;
    else deduped.push({ ...tempo });
  }
  return deduped;
}

function velocityCurve(rawVelocity, channelVolume = 127, expression = 127) {
  const strike = clamp(Number(rawVelocity) / 127, 0, 1);
  const volumeGain = clamp(Number(channelVolume) / 127, 0, 1);
  const expressionGain = clamp(Number(expression) / 127, 0, 1);
  const shapedStrike = 0.11 + (0.89 * (strike ** 1.28));
  return clamp(shapedStrike * (0.35 + 0.65 * volumeGain) * (0.35 + 0.65 * expressionGain), 0.04, 1);
}

function keySignatureLabel(event) {
  if (!event) return undefined;
  const majorKeys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
  const minorKeys = ['Abm', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm', 'Dm', 'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m'];
  const index = clamp(event.sharpsFlats + 7, 0, 14);
  return event.minor ? minorKeys[index] : majorKeys[index];
}

function pairNoteEvents(trackParses, tempoMap, ticksPerQuarter, filename) {
  const allEvents = trackParses
    .flatMap((track) => track.events)
    .sort((a, b) => {
      if (a.tick !== b.tick) return a.tick - b.tick;
      const aOn = a.command === 0x90 && a.data2 > 0;
      const bOn = b.command === 0x90 && b.data2 > 0;
      if (aOn !== bOn) return aOn ? 1 : -1;
      return a.trackIndex - b.trackIndex || a.channel - b.channel;
    });

  const channelState = new Map();
  const stateFor = (trackIndex, channel) => {
    const key = `${trackIndex}:${channel}`;
    if (!channelState.has(key)) {
      channelState.set(key, {
        volume: 127,
        expression: 127,
        program: 0,
        softPedal: false,
      });
    }
    return channelState.get(key);
  };

  const controlChanges = [];
  const programChanges = [];
  const active = new Map();
  const melodicNotes = [];
  const percussionNotes = [];
  const finalTick = Math.max(0, ...trackParses.map((track) => track.endTick || 0));

  const finishNote = (start, endTick) => {
    const startSeconds = secondsAtTick(start.tick, tempoMap, ticksPerQuarter);
    const endSeconds = secondsAtTick(Math.max(start.tick + 1, endTick), tempoMap, ticksPerQuarter);
    const duration = clamp(endSeconds - startSeconds, MIN_DURATION_SECONDS, MAX_DURATION_SECONDS);
    const noteName = midiToNote(start.data1);
    const velocity = velocityCurve(start.data2, start.channelVolume, start.expression);
    const note = {
      id: `midi-t${start.trackIndex}-c${start.channel}-n${start.data1}-${start.tick}-${start.sequence}`,
      note: noteName,
      midi: start.data1,
      time: Number(startSeconds.toFixed(5)),
      duration: Number(duration.toFixed(5)),
      scoreDuration: Number(duration.toFixed(5)),
      visualDuration: Number(duration.toFixed(5)),
      audioDuration: Number(duration.toFixed(5)),
      velocity: Number(velocity.toFixed(4)),
      rawVelocity: start.data2,
      channel: start.channel,
      program: start.program,
      hand: start.data1 < 60 ? 'left' : 'right',
      source: `MIDI track ${start.trackIndex + 1}`,
      originalTick: start.tick,
      endTick,
      softPedal: start.softPedal,
    };

    if (start.channel === PERCUSSION_CHANNEL) percussionNotes.push(note);
    else melodicNotes.push(note);
  };

  let sequence = 0;
  for (const event of allEvents) {
    const state = stateFor(event.trackIndex, event.channel);

    if (event.command === 0xc0) {
      state.program = event.data1;
      programChanges.push({
        time: Number(secondsAtTick(event.tick, tempoMap, ticksPerQuarter).toFixed(5)),
        tick: event.tick,
        track: event.trackIndex,
        channel: event.channel,
        program: event.data1,
      });
      continue;
    }

    if (event.command === 0xb0) {
      if (event.data1 === 7) state.volume = event.data2;
      if (event.data1 === 11) state.expression = event.data2;
      if (event.data1 === 67) state.softPedal = event.data2 >= 64;

      if ([7, 10, 11, 64, 67].includes(event.data1)) {
        controlChanges.push({
          id: `midi-cc-${event.trackIndex}-${event.channel}-${event.data1}-${event.tick}-${controlChanges.length}`,
          time: Number(secondsAtTick(event.tick, tempoMap, ticksPerQuarter).toFixed(5)),
          tick: event.tick,
          controller: event.data1,
          value: event.data2,
          down: event.data1 === 64 ? event.data2 >= 64 : undefined,
          channel: event.channel,
          track: event.trackIndex,
          source: `MIDI track ${event.trackIndex + 1}`,
        });
      }
      continue;
    }

    const isNoteOn = event.command === 0x90 && event.data2 > 0;
    const isNoteOff = event.command === 0x80 || (event.command === 0x90 && event.data2 === 0);
    if (!isNoteOn && !isNoteOff) continue;

    const key = `${event.trackIndex}:${event.channel}:${event.data1}`;
    const queue = active.get(key) || [];

    if (isNoteOn) {
      queue.push({
        ...event,
        sequence,
        channelVolume: state.volume,
        expression: state.expression,
        program: state.program,
        softPedal: state.softPedal,
      });
      sequence += 1;
      active.set(key, queue);
      continue;
    }

    const start = queue.shift();
    if (!start) continue;
    if (queue.length) active.set(key, queue);
    else active.delete(key);
    finishNote(start, event.tick);
  }

  for (const queue of active.values()) {
    queue.forEach((start) => finishNote(start, Math.max(finalTick, start.tick + Math.round(ticksPerQuarter / 2))));
  }

  const notes = melodicNotes.length ? melodicNotes : percussionNotes;
  if (!notes.length) {
    throw new Error('No playable note-on/note-off pairs were found in this MIDI file.');
  }

  const firstTempo = tempoMap[0]?.usPerQuarter || DEFAULT_TEMPO_US_PER_QUARTER;
  const bpm = Math.round(60000000 / firstTempo);
  const names = trackParses.flatMap((track) => track.trackNames).filter(Boolean);
  const firstTimeSignature = trackParses
    .flatMap((track) => track.timeSignatures)
    .sort((a, b) => a.tick - b.tick)[0];
  const firstKeySignature = trackParses
    .flatMap((track) => track.keySignatures)
    .sort((a, b) => a.tick - b.tick)[0];

  const sustainEvents = controlChanges.filter((event) => event.controller === 64);
  const tempoEvents = tempoMap.map((event) => ({
    time: Number(secondsAtTick(event.tick, tempoMap, ticksPerQuarter).toFixed(5)),
    tick: event.tick,
    bpm: Number((60000000 / event.usPerQuarter).toFixed(4)),
  }));

  return normalizeSong({
    schemaVersion: 'falling-piano-song-v1',
    title: filename.replace(/\.(mid|midi)$/i, '') || 'Uploaded MIDI',
    composer: names[0] || 'MIDI import',
    bpm,
    tempo: bpm,
    tempoEvents,
    timeSignature: firstTimeSignature
      ? `${firstTimeSignature.numerator}/${firstTimeSignature.denominator}`
      : '4/4',
    key: keySignatureLabel(firstKeySignature),
    importedFromMidi: true,
    normalizedFormat: 'falling-piano-song-v1',
    performance: {
      profile: 'midi-import-v10-musical',
      preserveScoreDurations: true,
      preserveScoreTiming: true,
      noOctaveFolding: true,
      sameKeyRetriggerGapSeconds: 0.024,
      defaultAutoplayReleaseSeconds: 0.58,
      velocityCurve: 'polymath-musical-v1',
      source: 'Standard MIDI file normalized in-browser to the same performance JSON used by ready-to-play songs',
    },
    controlChanges: sustainEvents,
    pedalEvents: sustainEvents,
    notes,
    percussionEvents: melodicNotes.length ? percussionNotes : [],
    midiProgramChanges: programChanges,
  });
}

export function parseMidiArrayBuffer(arrayBuffer, filename = 'Uploaded MIDI.mid') {
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.length < 14 || readAscii(bytes, 0, 4) !== 'MThd') {
    throw new Error('This is not a valid Standard MIDI file.');
  }

  const headerLength = readUint32(bytes, 4);
  const format = readUint16(bytes, 8);
  const trackCount = readUint16(bytes, 10);
  const division = readUint16(bytes, 12);

  if (division & 0x8000) {
    throw new Error('SMPTE-timed MIDI files are not supported. Export as PPQ/ticks-per-quarter MIDI.');
  }

  const ticksPerQuarter = division || TICKS_PER_QUARTER_FALLBACK;
  let offset = 8 + headerLength;
  const trackParses = [];

  for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
    if (offset + 8 > bytes.length || readAscii(bytes, offset, 4) !== 'MTrk') break;
    const trackLength = readUint32(bytes, offset + 4);
    if (offset + 8 + trackLength > bytes.length) {
      throw new Error(`MIDI track ${trackIndex + 1} is truncated.`);
    }
    trackParses.push(parseTrack(bytes, offset + 8, trackLength, trackIndex));
    offset += 8 + trackLength;
  }

  if (!trackParses.length) throw new Error('No MIDI tracks were found.');

  const tempoMap = buildTempoMap(trackParses.map((track) => track.tempos));
  const song = pairNoteEvents(trackParses, tempoMap, ticksPerQuarter, filename);
  song.midi = {
    format,
    trackCount: trackParses.length,
    ticksPerQuarter,
    tempoChanges: tempoMap.length,
    normalizedToJson: true,
  };
  return song;
}
