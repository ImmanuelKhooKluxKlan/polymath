import { clamp, midiToNote, parseNote } from './noteMath.js';

export const TEACHER_PROFILES = Object.freeze([
  {
    id: 'padme',
    name: 'Padme',
    title: 'Performance coach',
    description: 'Warm, precise, and focused on expressive melody.',
    voice: 'Encouraging',
    palette: { skin: '#d9a07f', hair: '#24192a', primary: '#7857d8', secondary: '#cf5eaa' },
    look: 'athletic',
  },
  {
    id: 'anakin',
    name: 'Anakin',
    title: 'Technique coach',
    description: 'Direct coaching for timing, power, and confident movement.',
    voice: 'Focused',
    palette: { skin: '#d3a27f', hair: '#543524', primary: '#2b3152', secondary: '#8ca1d8' },
    look: 'athletic-male',
  },
  {
    id: 'celeste',
    name: 'Celeste',
    title: 'Musicality coach',
    description: 'Calm phrasing lessons with a concert-hall feel.',
    voice: 'Calm',
    palette: { skin: '#c98f73', hair: '#27182c', primary: '#3569d4', secondary: '#7a78ed' },
    look: 'blue-dress',
  },
  {
    id: 'mace',
    name: 'Mace Windu',
    title: 'Piano master',
    description: 'Clear, disciplined guidance for difficult passages.',
    voice: 'Exact',
    palette: { skin: '#75452f', hair: '#20191b', primary: '#4c347a', secondary: '#c4b8ea' },
    look: 'master',
  },
]);

const GRAND_START_MIDI = 21;
const GRAND_END_MIDI = 108;

function midiForEvent(event) {
  const numeric = Number(event?.midi);
  if (Number.isFinite(numeric)) return Math.round(numeric);
  try {
    return parseNote(event?.note).midi;
  } catch {
    return null;
  }
}

export function teacherHandForEvent(event) {
  const explicit = String(event?.hand || '').toLowerCase();
  const role = String(event?.scoreRole || '').toLowerCase();
  if (explicit === 'left' || role.includes('left') || role.includes('bass')) return 'left';
  if (explicit === 'right' || role.includes('right') || role.includes('melody') || role.includes('top')) return 'right';
  const midi = midiForEvent(event);
  return midi !== null && midi < 60 ? 'left' : 'right';
}

export function pianoPercentForMidi(midi) {
  return clamp(((Number(midi) - GRAND_START_MIDI) / (GRAND_END_MIDI - GRAND_START_MIDI)) * 100, 2, 98);
}

export function fingerForMidi(midi, hand, handNotes = []) {
  const uniqueMidis = [...new Set(handNotes.map((note) => midiForEvent(note)).filter(Number.isFinite))]
    .sort((a, b) => a - b)
    .slice(0, 5);
  const index = Math.max(0, uniqueMidis.indexOf(Number(midi)));
  if (hand === 'left') return Math.max(1, 5 - index);
  return Math.min(5, index + 1);
}

function targetForHand(hand, relevant, currentTime) {
  const handNotes = relevant
    .filter((entry) => entry.hand === hand)
    .sort((a, b) => a.midi - b.midi);

  if (!handNotes.length) {
    return {
      hand,
      centerPercent: hand === 'left' ? 36 : 64,
      notes: [],
      isPressing: false,
      isUpcoming: false,
    };
  }

  const centerMidi = handNotes.reduce((sum, entry) => sum + entry.midi, 0) / handNotes.length;
  const notes = handNotes.slice(0, 5).map((entry) => ({
    ...entry,
    finger: fingerForMidi(entry.midi, hand, handNotes),
    percent: pianoPercentForMidi(entry.midi),
  }));

  return {
    hand,
    centerPercent: pianoPercentForMidi(centerMidi),
    notes,
    isPressing: handNotes.some((entry) => entry.time <= currentTime + 0.045 && entry.end >= currentTime - 0.045),
    isUpcoming: handNotes.every((entry) => entry.time > currentTime + 0.045),
  };
}

export function prepareTeacherHandTimeline(notes = []) {
  let maximumDuration = 0.2;
  const entries = [];

  for (const event of notes) {
    const midi = midiForEvent(event);
    if (!Number.isFinite(midi) || midi < GRAND_START_MIDI || midi > GRAND_END_MIDI) continue;
    const time = Math.max(0, Number(event?.time) || 0);
    const duration = Math.max(0.06, Number(event?.duration ?? event?.visualDuration) || 0.2);
    maximumDuration = Math.max(maximumDuration, duration);
    entries.push({
      id: event?.id || `${midi}-${time}`,
      midi,
      note: event?.note || midiToNote(midi),
      time,
      duration,
      hand: teacherHandForEvent(event),
    });
  }

  entries.sort((a, b) => a.time - b.time || a.midi - b.midi);
  return { type: 'teacher-hand-timeline', entries, maximumDuration };
}

function firstEntryAtOrAfter(entries, minimumTime) {
  let low = 0;
  let high = entries.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (entries[middle].time < minimumTime) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function buildTeacherHandTargets(notesOrTimeline = [], currentTime = 0, options = {}) {
  const lookAhead = Number.isFinite(options.lookAhead) ? options.lookAhead : 0.42;
  const releaseTail = Number.isFinite(options.releaseTail) ? options.releaseTail : 0.12;
  const selectedHand = options.handMode || 'both';
  const normalizedTime = Math.max(0, Number(currentTime) || 0);
  const timeline = notesOrTimeline?.type === 'teacher-hand-timeline'
    ? notesOrTimeline
    : prepareTeacherHandTimeline(notesOrTimeline);
  const relevant = [];
  const firstIndex = firstEntryAtOrAfter(timeline.entries, normalizedTime - timeline.maximumDuration - releaseTail);

  for (let index = firstIndex; index < timeline.entries.length; index += 1) {
    const event = timeline.entries[index];
    const end = event.time + event.duration + releaseTail;
    if (event.time > normalizedTime + lookAhead) break;
    const { hand } = event;
    if (selectedHand !== 'both' && selectedHand !== hand) continue;
    if (end >= normalizedTime) {
      relevant.push({
        id: event.id,
        midi: event.midi,
        note: event.note,
        time: event.time,
        end,
        hand,
      });
    }
  }

  // Prefer notes that are being held. If nothing is active, preview only the
  // nearest upcoming chord so the teacher does not jump across several phrases.
  const active = relevant.filter((entry) => entry.time <= normalizedTime + 0.045 && entry.end >= normalizedTime);
  let focus = active;
  if (!focus.length && relevant.length) {
    const nearestTime = Math.min(...relevant.map((entry) => entry.time));
    focus = relevant.filter((entry) => Math.abs(entry.time - nearestTime) <= 0.035);
  }

  return {
    left: targetForHand('left', focus, normalizedTime),
    right: targetForHand('right', focus, normalizedTime),
    hasTargets: focus.length > 0,
  };
}

export function teacherReply(teacher, message, targets) {
  const text = String(message || '').trim().toLowerCase();
  const played = [...(targets?.left?.notes || []), ...(targets?.right?.notes || [])];
  const noteSummary = played.length
    ? played.map((note) => `${note.note} (finger ${note.finger})`).join(', ')
    : 'the next falling notes';
  const name = teacher?.name || 'Your teacher';

  if (/hello|hi|hey/.test(text)) return `${name} here. Start the lesson when you are ready, and I will demonstrate each hand.`;
  if (/finger|hand|play|note|wrong|help/.test(text)) return `Watch ${noteSummary}. Keep your wrist relaxed and press only when the note reaches the line.`;
  if (/slow|fast|speed|tempo/.test(text)) return 'Use the speed control below the piano. Begin slowly, then raise it only after the movement feels easy.';
  return `I’m following this lesson with you. Right now, focus on ${noteSummary}.`;
}
