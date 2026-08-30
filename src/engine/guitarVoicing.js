import { parseNote } from './noteMath.js';

const OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64];

export function guitarCandidatesForMidi(midi, maximumFret = 24) {
  return OPEN_STRING_MIDI
    .map((openMidi, stringIndex) => ({ stringIndex, fret: midi - openMidi }))
    .filter(({ fret }) => Number.isInteger(fret) && fret >= 0 && fret <= maximumFret)
    .sort((a, b) => a.fret - b.fret || b.stringIndex - a.stringIndex);
}

function midiValue(note) {
  if (Number.isFinite(Number(note?.midi))) return Math.round(Number(note.midi));
  return parseNote(note?.note).midi;
}

function candidateScore(assignments, lowestMidi, highestMidi) {
  if (!assignments.length) return -Infinity;
  const frets = assignments.map(({ fret }) => fret);
  const midis = new Set(assignments.map(({ midi }) => midi));
  return assignments.length * 10_000
    + (midis.has(highestMidi) ? 500 : 0)
    + (midis.has(lowestMidi) ? 250 : 0)
    - (Math.max(...frets) - Math.min(...frets)) * 8
    - frets.reduce((sum, fret) => sum + fret, 0);
}

// A greedy choice can waste a string and silently drop a later melody note.
// Six strings make exhaustive backtracking cheap, so find the best complete
// fingering instead. The score prioritizes note count, then melody/bass, then a
// compact low-fret hand shape.
export function assignNotesToStrings(notes, { maximumFret = 24 } = {}) {
  const unique = new Map();
  for (const note of Array.isArray(notes) ? notes : []) {
    try {
      const midi = midiValue(note);
      const existing = unique.get(midi);
      const strength = Number(note?.velocity || 0.72) * Number(note?.duration || 0.45);
      const existingStrength = existing
        ? Number(existing.note?.velocity || 0.72) * Number(existing.note?.duration || 0.45)
        : -1;
      if (!existing || strength > existingStrength) {
        unique.set(midi, {
          note,
          midi,
          candidates: guitarCandidatesForMidi(midi, maximumFret),
        });
      }
    } catch {
      // Malformed pitches are ignored without sacrificing the rest of a chord.
    }
  }
  const prepared = [...unique.values()]
    .filter((item) => item.candidates.length)
    .sort((a, b) => a.candidates.length - b.candidates.length || b.midi - a.midi)
    .slice(0, 12);
  if (!prepared.length) return [];

  const lowestMidi = Math.min(...prepared.map(({ midi }) => midi));
  const highestMidi = Math.max(...prepared.map(({ midi }) => midi));
  let best = [];
  let bestScore = -Infinity;

  function search(index, usedStrings, assignments) {
    const remainingCapacity = Math.min(6 - usedStrings.size, prepared.length - index);
    if (assignments.length + remainingCapacity < best.length) return;
    if (index >= prepared.length || usedStrings.size >= 6) {
      const score = candidateScore(assignments, lowestMidi, highestMidi);
      if (score > bestScore) {
        bestScore = score;
        best = assignments.map((assignment) => ({ ...assignment }));
      }
      return;
    }

    const item = prepared[index];
    for (const candidate of item.candidates) {
      if (usedStrings.has(candidate.stringIndex)) continue;
      usedStrings.add(candidate.stringIndex);
      assignments.push({ ...candidate, note: item.note, midi: item.midi });
      search(index + 1, usedStrings, assignments);
      assignments.pop();
      usedStrings.delete(candidate.stringIndex);
    }
    search(index + 1, usedStrings, assignments);
  }

  search(0, new Set(), []);
  return best.sort((a, b) => a.stringIndex - b.stringIndex);
}
