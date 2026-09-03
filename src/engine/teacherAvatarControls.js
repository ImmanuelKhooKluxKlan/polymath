export const TEACHER_POSES = Object.freeze([
  { id: 'ready', label: 'Ready', rotation: 0, scale: 1 },
  { id: 'bow', label: 'Bend', rotation: 24, scale: 0.94 },
  { id: 'stretch', label: 'Stretch', rotation: 0, scale: 1 },
  { id: 'rest', label: 'Lie down', rotation: 88, scale: 0.82 },
  { id: 'jump', label: 'Jump', rotation: 0, scale: 1 },
]);

export const TEACHER_BODY_PARTS = Object.freeze([
  { id: 'head', label: 'Head' },
  { id: 'torso', label: 'Torso' },
  { id: 'leftArm', label: 'Left arm' },
  { id: 'rightArm', label: 'Right arm' },
  { id: 'lower', label: 'Lower body' },
]);

export function teacherPoseById(poseId) {
  return TEACHER_POSES.find((pose) => pose.id === poseId) || TEACHER_POSES[0];
}

export function clampTeacherOffset(offset, bounds = {}) {
  const maximumX = Math.max(0, Number(bounds.maximumX) || 0);
  const maximumY = Math.max(0, Number(bounds.maximumY) || 0);
  return {
    x: Math.max(-maximumX, Math.min(maximumX, Number(offset?.x) || 0)),
    y: Math.max(-maximumY, Math.min(maximumY, Number(offset?.y) || 0)),
  };
}

export function normalizeTeacherArmAngle(value) {
  return Math.max(-110, Math.min(110, Math.round(Number(value) || 0)));
}

export function teacherBodyPartById(partId) {
  return TEACHER_BODY_PARTS.find((part) => part.id === partId) || TEACHER_BODY_PARTS[1];
}
