import { clamp, midiToNote, parseNote } from './noteMath.js';

export const TEACHER_PROFILES = Object.freeze([
  {
    id: 'aria',
    name: 'Aria',
    title: 'Piano performance teacher',
    description: 'Calm demonstrations with clear posture, phrasing, and connected movement.',
    voice: 'Warm and precise',
    voiceType: 'feminine',
    image: '/teachers/polymath-teacher-studio-v1.jpg',
    stageImage: '/teachers/polymath-teacher-studio-v1.jpg',
    portraitPosition: '48% 20%',
    handCameraImage: '/teachers/pianist-hands-overhead-v1.webp',
    pressedHandCameraImage: '/teachers/pianist-hands-pressed-v2.webp',
    stageShoulders: { left: [442, 265], right: [589, 270] },
    palette: { skin: '#d5a078', skinShadow: '#986047', hair: '#4a2d24', primary: '#4c477f', secondary: '#9c70c8' },
    look: 'professional',
  },
  {
    id: 'nova',
    name: 'Padme',
    title: 'Expressive performance coach',
    description: 'Warm, confident, and focused on expressive melody.',
    voice: 'Warm and expressive',
    voiceType: 'feminine',
    image: '/teachers/padme-teacher-studio-v1.jpg',
    stageImage: '/teachers/padme-teacher-studio-v1.jpg',
    portraitPosition: '35% 20%',
    handCameraImage: '/teachers/pianist-hands-overhead-v1.webp',
    pressedHandCameraImage: '/teachers/pianist-hands-pressed-v2.webp',
    armImage: '/teachers/arm-light-full-v1.webp',
    palette: { skin: '#d9a07f', skinShadow: '#a96e54', hair: '#24192a', primary: '#7857d8', secondary: '#cf5eaa' },
    look: 'athletic',
    requiresAdultConfirmation: true,
  },
  {
    id: 'anakin',
    name: 'Anakin',
    title: 'Technique coach',
    description: 'Direct coaching for timing, power, and confident movement.',
    voice: 'Focused',
    voiceType: 'masculine',
    image: '/teachers/anakin-teacher-studio-v2.jpg',
    stageImage: '/teachers/anakin-teacher-studio-v2.jpg',
    portraitPosition: '35% 17%',
    handCameraImage: '/teachers/pianist-hands-overhead-male-v1.webp',
    pressedHandCameraImage: '/teachers/pianist-hands-pressed-male-v2.webp',
    armImage: '/teachers/arm-light-full-v1.webp',
    palette: { skin: '#d3a27f', skinShadow: '#9c654a', hair: '#543524', primary: '#2b3152', secondary: '#8ca1d8' },
    look: 'athletic-male',
  },
  {
    id: 'taylor',
    name: 'Taylor',
    title: 'Songwriting coach',
    description: 'Friendly guidance for melody, phrasing, and storytelling.',
    voice: 'Thoughtful',
    voiceType: 'feminine',
    image: '/teachers/taylor-teacher-studio-v1.jpg',
    stageImage: '/teachers/taylor-teacher-studio-v1.jpg',
    portraitPosition: '35% 18%',
    handCameraImage: '/teachers/pianist-hands-overhead-v1.webp',
    pressedHandCameraImage: '/teachers/pianist-hands-pressed-v2.webp',
    armImage: '/teachers/arm-light-full-v1.webp',
    palette: { skin: '#e2b39e', skinShadow: '#ae7868', hair: '#d4bd9d', primary: '#b77c98', secondary: '#e4b5cf' },
    look: 'songwriter',
  },
  {
    id: 'mace',
    name: 'Mace Windu',
    title: 'Piano master',
    description: 'Clear, disciplined guidance for difficult passages.',
    voice: 'Exact',
    voiceType: 'masculine',
    image: '/teachers/mace-teacher-studio-v1.jpg',
    stageImage: '/teachers/mace-teacher-studio-v1.jpg',
    portraitPosition: '34% 18%',
    handCameraImage: '/teachers/pianist-hands-overhead-dark-v1.webp',
    pressedHandCameraImage: '/teachers/pianist-hands-pressed-dark-v2.webp',
    armImage: '/teachers/arm-dark-full-v1.webp',
    palette: { skin: '#75452f', skinShadow: '#4d2a1d', hair: '#20191b', primary: '#4c347a', secondary: '#c4b8ea' },
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

export function teacherRowHandPlacement(target, row, side, showAtRest = false) {
  const notes = (target?.notes || [])
    .map((note) => ({ ...note, position: row?.getPosition?.(note.midi) }))
    .filter((note) => note.position?.rowId === row?.id);
  if (!notes.length && !showAtRest) return null;

  const fallbackCenter = side === 'left' ? 36 : 64;
  const centers = notes.map((note) => note.position.centerPercent);
  const centerPercent = centers.length
    ? centers.reduce((sum, value) => sum + value, 0) / centers.length
    : fallbackCenter;
  const spread = centers.length > 1 ? Math.max(...centers) - Math.min(...centers) : 0;
  const blackNotes = notes.filter((note) => note.position.isBlack).length;
  const blackKeyBias = notes.length > 0 && blackNotes >= Math.ceil(notes.length / 2);
  const distanceFromHome = centerPercent - fallbackCenter;
  const chordSize = Math.min(5, notes.length);

  return {
    notes,
    centerPercent: clamp(centerPercent, 1.5, 98.5),
    // Keep the visual guide compact enough that learners can still see the
    // neighbouring keys. Wider chords may open the pose slightly, but never
    // restore the oversized hand crop used by the first prototype.
    widthPercent: clamp(21 + spread * 0.55, 21, 33),
    horizontalScale: 0.9,
    // Anchor the photographic fingertips inside the playable key bed. The
    // source image includes a full palm and forearm, so placing its anchor
    // near the front of a white key made the hands appear to hover below the
    // instrument. Black notes sit farther back; white notes and the relaxed
    // home position sit just in front of them.
    fingerDepthPercent: blackKeyBias ? 13 : 25,
    // A pianist leads lateral travel from the wrist instead of sliding a
    // rigid hand-shaped sticker. These small, bounded pose changes keep the
    // photographic hand natural without ever twisting it beyond a plausible
    // neutral playing posture.
    wristTiltDegrees: clamp(
      distanceFromHome * 0.085 + (side === 'left' ? -0.8 : 0.8),
      -4.5,
      4.5,
    ),
    verticalFlex: clamp(1 - chordSize * 0.004 - (blackKeyBias ? 0.012 : 0), 0.965, 1),
    approachLiftPixels: clamp(7 - chordSize * 0.45, 4.75, 7),
    pressDepthPixels: blackKeyBias ? 2.25 : 4,
  };
}

export function fingerForMidi(midi, hand, handNotes = []) {
  const uniqueMidis = [...new Set(handNotes.map((note) => midiForEvent(note)).filter(Number.isFinite))]
    .sort((a, b) => a - b)
    .slice(0, 5);
  const index = Math.max(0, uniqueMidis.indexOf(Number(midi)));
  const rightPlans = {
    1: [3],
    2: [1, 5],
    3: [1, 3, 5],
    4: [1, 2, 3, 5],
    5: [1, 2, 3, 4, 5],
  };
  const right = rightPlans[uniqueMidis.length] || rightPlans[5];
  const plan = hand === 'left' ? [...right].reverse() : right;
  return plan[Math.min(index, plan.length - 1)] || 3;
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

export function teacherReply(teacher, message, targets, practiceReport = null) {
  const text = String(message || '').trim().toLowerCase();
  const played = [...(targets?.left?.notes || []), ...(targets?.right?.notes || [])];
  const noteSummary = played.length
    ? played.map((note) => `${note.note} (finger ${note.finger})`).join(', ')
    : 'the next falling notes';
  const name = teacher?.name || 'Your teacher';

  if (/feedback|progress|score|improve|mistake|again/.test(text) && practiceReport) {
    return `${practiceReport.headline}. Focus on ${practiceReport.focus.toLowerCase()}: ${practiceReport.nextAction}`;
  }

  if (teacher?.id === 'mace') {
    if (/hello|hi|hey/.test(text)) return 'Sit properly. Wrists level. Begin when the timing line reaches the keys.';
    if (/slow|fast|speed|tempo/.test(text)) return 'Reduce the speed. Accuracy first. Increase it only after three clean repetitions.';
    if (/finger|hand|play|note|wrong|help/.test(text)) return `Watch ${noteSummary}. Again—and this time, do not rush the release.`;
    return `Focus on ${noteSummary}. I will compliment it when it is actually precise.`;
  }

  if (/hello|hi|hey/.test(text)) return `${name} here. Start the lesson when you are ready, and I will demonstrate each hand.`;
  if (/finger|hand|play|note|wrong|help/.test(text)) return `Watch ${noteSummary}. Keep your wrist relaxed and press only when the note reaches the line.`;
  if (/slow|fast|speed|tempo/.test(text)) return 'Use the speed control below the piano. Begin slowly, then raise it only after the movement feels easy.';
  return `I’m following this lesson with you. Right now, focus on ${noteSummary}.`;
}
