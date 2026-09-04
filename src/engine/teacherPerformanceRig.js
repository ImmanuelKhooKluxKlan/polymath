import { clamp } from './noteMath.js';

export const TEACHER_STAGE = Object.freeze({
  width: 1200,
  height: 680,
  keyboardLeft: 66,
  keyboardRight: 1134,
  keyTop: 500,
  whiteKeyHeight: 116,
  blackKeyHeight: 70,
});

const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);

function isBlackMidi(midi) {
  return BLACK_PITCH_CLASSES.has(((Number(midi) % 12) + 12) % 12);
}

export function buildTeacherKeyboardGeometry() {
  const whiteWidth = (TEACHER_STAGE.keyboardRight - TEACHER_STAGE.keyboardLeft) / 52;
  const keys = [];
  let whiteIndex = 0;

  for (let midi = 21; midi <= 108; midi += 1) {
    const black = isBlackMidi(midi);
    if (black) {
      const width = whiteWidth * 0.59;
      const center = TEACHER_STAGE.keyboardLeft + whiteIndex * whiteWidth;
      keys.push({
        midi,
        black: true,
        x: center - width / 2,
        center,
        width,
        y: TEACHER_STAGE.keyTop,
        height: TEACHER_STAGE.blackKeyHeight,
      });
      continue;
    }

    const x = TEACHER_STAGE.keyboardLeft + whiteIndex * whiteWidth;
    keys.push({
      midi,
      black: false,
      x,
      center: x + whiteWidth / 2,
      width: whiteWidth,
      y: TEACHER_STAGE.keyTop,
      height: TEACHER_STAGE.whiteKeyHeight,
    });
    whiteIndex += 1;
  }

  return keys;
}

export const TEACHER_KEYBOARD_GEOMETRY = Object.freeze(buildTeacherKeyboardGeometry());
const KEY_BY_MIDI = new Map(TEACHER_KEYBOARD_GEOMETRY.map((key) => [key.midi, key]));

export function teacherKeyGeometry(midi) {
  return KEY_BY_MIDI.get(Math.round(Number(midi))) || null;
}

export function teacherKeyX(midi) {
  const key = teacherKeyGeometry(midi);
  if (key) return key.center;
  return clamp(
    TEACHER_STAGE.keyboardLeft
      + ((Number(midi) - 21) / 87) * (TEACHER_STAGE.keyboardRight - TEACHER_STAGE.keyboardLeft),
    TEACHER_STAGE.keyboardLeft,
    TEACHER_STAGE.keyboardRight,
  );
}

export function teacherHandCenter(target, side) {
  const positions = (target?.notes || [])
    .map((note) => teacherKeyX(note.midi))
    .filter(Number.isFinite);
  if (positions.length) return positions.reduce((sum, value) => sum + value, 0) / positions.length;
  return side === 'left' ? 410 : 790;
}

export function separateTeacherHandCentres(left, right, minimumGap = 240) {
  const safeLeft = Number.isFinite(Number(left)) ? Number(left) : 410;
  const safeRight = Number.isFinite(Number(right)) ? Number(right) : 790;
  const gap = Math.max(140, Number(minimumGap) || 240);
  if (safeRight - safeLeft >= gap) return { left: safeLeft, right: safeRight };
  const middle = (safeLeft + safeRight) / 2;
  const separatedLeft = clamp(middle - gap / 2, 150, 1050 - gap);
  return { left: separatedLeft, right: separatedLeft + gap };
}
