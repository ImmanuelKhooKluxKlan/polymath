export const TEACHER_POSES = Object.freeze([
  { id: 'ready', label: 'Ready', rotation: 0, scaleX: 1, scaleY: 1, translateY: '0%' },
  { id: 'bend', label: 'Bend', rotation: 18, scaleX: 0.96, scaleY: 0.94, translateY: '4%' },
  { id: 'kneel', label: 'Kneel', rotation: 0, scaleX: 1.04, scaleY: 0.74, translateY: '17%' },
  { id: 'wide', label: 'Wide stance', rotation: 0, scaleX: 1.15, scaleY: 0.96, translateY: '2%' },
  { id: 'rest', label: 'Lie down', rotation: 88, scaleX: 0.82, scaleY: 0.82, translateY: '1%' },
  { id: 'jump', label: 'Jump', rotation: 0, scaleX: 1, scaleY: 1, translateY: '0%' },
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

export function clampTeacherDepth(value) {
  return Math.max(0.7, Math.min(1.45, Number(value) || 1));
}
