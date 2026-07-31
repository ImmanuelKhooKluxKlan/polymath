import { normalizeSong } from '../engine/scheduler.js';

function childText(node, selector, fallback = '') {
  return node.querySelector(selector)?.textContent?.trim() || fallback;
}

function alterToAccidental(alterText) {
  const alter = Number(alterText || 0);
  if (alter === 1) return '#';
  if (alter === -1) return 'b';
  if (alter === 2) return '##';
  if (alter === -2) return 'bb';
  return '';
}

function noteNameFromPitch(pitch) {
  const step = childText(pitch, 'step');
  const alter = alterToAccidental(childText(pitch, 'alter', '0'));
  const octave = childText(pitch, 'octave');
  if (!step || octave === '') return null;
  if (alter === '##' || alter === 'bb') return null;
  return `${step}${alter}${octave}`;
}

function tempoFromScore(doc) {
  const perMinute = doc.querySelector('sound[tempo]')?.getAttribute('tempo')
    || doc.querySelector('metronome per-minute')?.textContent;
  const tempo = Number(perMinute);
  return Number.isFinite(tempo) && tempo > 0 ? tempo : 100;
}

function inferHandFromStaff(note, noteName) {
  const staff = Number(childText(note, 'staff', '0'));
  if (staff === 2) return 'left';
  if (staff === 1) return 'right';
  const octave = Number(String(noteName).match(/(-?\d+)$/)?.[1]);
  return Number.isFinite(octave) && octave < 4 ? 'left' : 'right';
}

export function parseMusicXmlText(text, filename = 'Uploaded MusicXML.musicxml') {
  const doc = new DOMParser().parseFromString(text, 'application/xml');
  const parserError = doc.querySelector('parsererror');
  if (parserError) throw new Error('The MusicXML file could not be parsed.');

  const scoreTitle = childText(doc, 'work-title')
    || childText(doc, 'movement-title')
    || filename.replace(/\.(musicxml|xml)$/i, '');
  const composer = doc.querySelector('creator[type="composer"]')?.textContent?.trim()
    || childText(doc, 'creator')
    || 'MusicXML import';
  const tempo = tempoFromScore(doc);
  const notes = [];

  let divisions = Number(childText(doc, 'attributes divisions', '1')) || 1;
  const secondsPerQuarter = 60 / tempo;

  for (const part of Array.from(doc.querySelectorAll('part'))) {
    let timeBeats = 0;
    let measureNumber = 0;
    const activeTies = new Map();

    for (const measure of Array.from(part.children).filter((node) => node.tagName === 'measure')) {
      measureNumber = Number(measure.getAttribute('number')) || measureNumber + 1;
      const nextDivisions = Number(childText(measure, 'attributes divisions', String(divisions)));
      if (Number.isFinite(nextDivisions) && nextDivisions > 0) divisions = nextDivisions;

      let previousChordStart = timeBeats;

      for (const element of Array.from(measure.children)) {
        if (element.tagName === 'backup') {
          const duration = Number(childText(element, 'duration', '0')) || 0;
          timeBeats -= duration / divisions;
          continue;
        }

        if (element.tagName === 'forward') {
          const duration = Number(childText(element, 'duration', '0')) || 0;
          timeBeats += duration / divisions;
          continue;
        }

        if (element.tagName !== 'note') continue;

        const durationDivisions = Number(childText(element, 'duration', '0')) || 0;
        const durationBeats = durationDivisions / divisions;
        const isChord = Boolean(element.querySelector('chord'));
        const isRest = Boolean(element.querySelector('rest'));
        const startBeat = isChord ? previousChordStart : timeBeats;
        if (!isChord) previousChordStart = timeBeats;

        if (!isRest) {
          const pitch = element.querySelector('pitch');
          const noteName = pitch ? noteNameFromPitch(pitch) : null;
          if (noteName) {
            const tieStart = Boolean(element.querySelector('tie[type="start"], tied[type="start"]'));
            const tieStop = Boolean(element.querySelector('tie[type="stop"], tied[type="stop"]'));
            const tieKey = `${noteName}:${childText(element, 'voice', '1')}:${childText(element, 'staff', '0')}`;
            const startSeconds = startBeat * secondsPerQuarter;
            const durationSeconds = Math.max(0.035, durationBeats * secondsPerQuarter);
            const velocity = element.querySelector('sound[dynamics]')
              ? Math.max(0.08, Math.min(1, Number(element.querySelector('sound[dynamics]').getAttribute('dynamics')) / 127))
              : 0.78;

            if (tieStop && activeTies.has(tieKey)) {
              const held = activeTies.get(tieKey);
              held.duration += durationSeconds;
              held.visualDuration = held.duration;
              held.audioDuration = held.duration;
              if (!tieStart) {
                notes.push({
                  ...held,
                  duration: Number(held.duration.toFixed(5)),
                  visualDuration: Number(held.visualDuration.toFixed(5)),
                  audioDuration: Number(held.audioDuration.toFixed(5)),
                });
                activeTies.delete(tieKey);
              }
            } else if (tieStart) {
              activeTies.set(tieKey, {
                id: `xml-${measureNumber}-${notes.length}-${noteName}`,
                note: noteName,
                time: Number(startSeconds.toFixed(5)),
                duration: durationSeconds,
                visualDuration: durationSeconds,
                audioDuration: durationSeconds,
                velocity,
                hand: inferHandFromStaff(element, noteName),
                measure: measureNumber,
                source: 'MusicXML tied note',
              });
            } else {
              notes.push({
                id: `xml-${measureNumber}-${notes.length}-${noteName}`,
                note: noteName,
                time: Number(startSeconds.toFixed(5)),
                duration: Number(durationSeconds.toFixed(5)),
                visualDuration: Number(durationSeconds.toFixed(5)),
                audioDuration: Number(durationSeconds.toFixed(5)),
                velocity,
                hand: inferHandFromStaff(element, noteName),
                measure: measureNumber,
                source: 'MusicXML note',
              });
            }
          }
        }

        if (!isChord) timeBeats += durationBeats;
      }
    }

    for (const held of activeTies.values()) {
      notes.push({
        ...held,
        duration: Number(held.duration.toFixed(5)),
        visualDuration: Number(held.visualDuration.toFixed(5)),
        audioDuration: Number(held.audioDuration.toFixed(5)),
      });
    }
  }

  if (!notes.length) throw new Error('No playable notes were found in this MusicXML file.');

  return normalizeSong({
    title: scoreTitle,
    composer,
    bpm: tempo,
    tempo,
    timeSignature: '4/4',
    performance: {
      profile: 'musicxml-import-v9',
      preserveScoreDurations: true,
      preserveScoreTiming: true,
      noOctaveFolding: true,
      sameKeyRetriggerGapSeconds: 0.035,
      defaultAutoplayReleaseSeconds: 0.45,
    },
    notes,
  });
}
