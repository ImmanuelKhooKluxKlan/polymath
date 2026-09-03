const HALF_PI = Math.PI / 2;

export const TEACHER_RIG_POSES = Object.freeze({
  ready: {
    root: { y: 0, x: 0, z: 0 },
    joints: {
      spine: { x: 0.04 },
      upperArmL: { x: -0.08, z: -0.16 },
      upperArmR: { x: -0.08, z: 0.16 },
      lowerArmL: { x: -0.12 },
      lowerArmR: { x: -0.12 },
    },
  },
  bend: {
    root: { y: -0.1, x: 0, z: 0 },
    joints: {
      hips: { x: 0.18 },
      spine: { x: 0.72 },
      head: { x: -0.28 },
      upperArmL: { x: -0.55, z: -0.12 },
      upperArmR: { x: -0.55, z: 0.12 },
      lowerArmL: { x: -0.38 },
      lowerArmR: { x: -0.38 },
    },
  },
  kneel: {
    root: { y: -0.55, x: 0, z: 0 },
    joints: {
      hips: { x: -0.12 },
      spine: { x: 0.12 },
      upperLegL: { x: -1.08, z: -0.12 },
      upperLegR: { x: -0.52, z: 0.22 },
      lowerLegL: { x: 2.08 },
      lowerLegR: { x: 1.08 },
      upperArmL: { x: -0.18, z: -0.22 },
      upperArmR: { x: -0.18, z: 0.22 },
    },
  },
  wide: {
    root: { y: -0.12, x: 0, z: 0 },
    joints: {
      upperLegL: { z: -0.38 },
      upperLegR: { z: 0.38 },
      lowerLegL: { z: 0.12 },
      lowerLegR: { z: -0.12 },
      upperArmL: { z: -0.32 },
      upperArmR: { z: 0.32 },
    },
  },
  rest: {
    root: { y: -1.18, x: 0, z: HALF_PI },
    joints: {
      spine: { x: 0.05 },
      upperArmL: { x: -0.1, z: -0.38 },
      upperArmR: { x: -0.1, z: 0.38 },
      upperLegL: { x: -0.12, z: -0.12 },
      upperLegR: { x: 0.12, z: 0.12 },
    },
  },
  jump: {
    root: { y: 0, x: 0, z: 0 },
    joints: {
      upperArmL: { x: 0.08, z: -2.62 },
      upperArmR: { x: 0.08, z: 2.62 },
      lowerArmL: { x: -0.18 },
      lowerArmR: { x: -0.18 },
      upperLegL: { z: -0.14 },
      upperLegR: { z: 0.14 },
      lowerLegL: { x: 0.28 },
      lowerLegR: { x: 0.28 },
    },
  },
});

export const TEACHER_JOINT_LABELS = Object.freeze({
  hips: 'Hips',
  spine: 'Torso',
  head: 'Head',
  upperArmL: 'Left upper arm',
  lowerArmL: 'Left forearm',
  handL: 'Left hand',
  upperArmR: 'Right upper arm',
  lowerArmR: 'Right forearm',
  handR: 'Right hand',
  upperLegL: 'Left thigh',
  lowerLegL: 'Left lower leg',
  footL: 'Left foot',
  upperLegR: 'Right thigh',
  lowerLegR: 'Right lower leg',
  footR: 'Right foot',
});

function normalizedBoneName(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function sideMatch(name, side) {
  if (side === 'L') return name.includes('left') || /(^|[^a-z])l([^a-z]|$)/.test(String(name));
  return name.includes('right') || /(^|[^a-z])r([^a-z]|$)/.test(String(name));
}

export function canonicalTeacherBone(nameValue) {
  const original = String(nameValue || '').toLowerCase();
  const name = normalizedBoneName(nameValue);
  if (!name) return '';
  if (/head(?!top|end)/.test(name)) return 'head';
  if (/hips|pelvis/.test(name)) return 'hips';
  if (/spine|chest|upperbody/.test(name)) return 'spine';

  for (const side of ['L', 'R']) {
    const sideName = side === 'L' ? 'left' : 'right';
    const isSide = name.includes(sideName) || sideMatch(original, side);
    if (!isSide) continue;
    if (/forearm|lowerarm|elbow/.test(name)) return `lowerArm${side}`;
    if (/upperarm|shoulder|(^|mixamorig)(left|right)arm/.test(name)) return `upperArm${side}`;
    if (/hand|wrist/.test(name)) return `hand${side}`;
    if (/upleg|upperleg|thigh/.test(name)) return `upperLeg${side}`;
    if (/lowerleg|shin|calf|knee|leftleg|rightleg/.test(name)) return `lowerLeg${side}`;
    if (/foot|ankle|toe/.test(name)) return `foot${side}`;
  }
  return '';
}

export function teacherRigPose(poseId) {
  return TEACHER_RIG_POSES[poseId] || TEACHER_RIG_POSES.ready;
}

export function clampJointRotation(jointKey, rotation = {}) {
  const isHead = jointKey === 'head';
  const isSpine = jointKey === 'spine' || jointKey === 'hips';
  const maximumX = isHead ? 0.75 : isSpine ? 1.05 : 2.7;
  const maximumY = isHead ? 1.05 : isSpine ? 0.85 : 1.7;
  const maximumZ = isSpine ? 0.75 : 2.8;
  const clamp = (value, maximum) => Math.max(-maximum, Math.min(maximum, Number(value) || 0));
  return {
    x: clamp(rotation.x, maximumX),
    y: clamp(rotation.y, maximumY),
    z: clamp(rotation.z, maximumZ),
  };
}

export function draggedJointRotation(jointKey, startRotation, deltaX, deltaY) {
  const start = startRotation || { x: 0, y: 0, z: 0 };
  let next;
  if (jointKey === 'head' || jointKey === 'spine' || jointKey === 'hips') {
    next = { x: start.x + deltaY * 1.7, y: start.y + deltaX * 2.1, z: start.z + deltaX * 0.35 };
  } else if (/Arm|hand/.test(jointKey)) {
    next = { x: start.x + deltaY * 2.6, y: start.y + deltaX * 0.8, z: start.z + deltaX * 2.8 };
  } else {
    next = { x: start.x + deltaY * 2.7, y: start.y + deltaX * 0.45, z: start.z + deltaX * 2.15 };
  }
  return clampJointRotation(jointKey, next);
}
