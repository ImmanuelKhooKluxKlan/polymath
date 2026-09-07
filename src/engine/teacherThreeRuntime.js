import * as THREE from 'three';
import {
  canonicalTeacherBone,
  draggedJointRotation,
  teacherRigPose,
  TEACHER_JOINT_LABELS,
} from './teacherRig.js';

function safeColour(value, fallback) {
  try {
    return new THREE.Color(value || fallback);
  } catch {
    return new THREE.Color(fallback);
  }
}

function partMaterial(colour, options = {}) {
  return new THREE.MeshStandardMaterial({
    color: colour,
    roughness: options.roughness ?? 0.72,
    metalness: options.metalness ?? 0.02,
  });
}

function attachPart(parent, geometry, material, jointKey, position, scale = [1, 1, 1]) {
  const mesh = new THREE.Mesh(geometry, material.clone());
  mesh.position.set(...position);
  mesh.scale.set(...scale);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.jointKey = jointKey;
  parent.add(mesh);
  return mesh;
}

function addJoint(parent, name, position) {
  const joint = new THREE.Group();
  joint.name = name;
  joint.position.set(...position);
  parent.add(joint);
  return joint;
}

function createProceduralTeacher(teacher, performanceTier = 'balanced') {
  const root = new THREE.Group();
  root.name = 'PolymathProceduralTeacher';
  const pickables = [];
  const joints = {};
  const palette = teacher?.palette || {};
  const skin = partMaterial(safeColour(palette.skin, '#c78f71'));
  const hair = partMaterial(safeColour(palette.hair, '#30232b'), { roughness: 0.88 });
  const primary = partMaterial(safeColour(palette.primary, '#7057d7'));
  const secondary = partMaterial(safeColour(palette.secondary, '#c45b9c'));
  const dark = partMaterial('#171929', { roughness: 0.8 });
  const white = partMaterial('#f4f0ff', { roughness: 0.55 });

  const radialSegments = performanceTier === 'lite' ? 10 : performanceTier === 'full' ? 22 : 16;
  const heightSegments = performanceTier === 'lite' ? 8 : 12;
  const sphere = new THREE.SphereGeometry(1, radialSegments, heightSegments);
  const limb = new THREE.CylinderGeometry(0.12, 0.14, 1, radialSegments);
  const slimLimb = new THREE.CylinderGeometry(0.1, 0.12, 1, radialSegments);
  const torsoGeometry = new THREE.SphereGeometry(1, radialSegments, heightSegments);
  const box = new THREE.BoxGeometry(1, 1, 1);

  const hips = addJoint(root, 'hips', [0, 0.18, 0]);
  joints.hips = hips;
  pickables.push(attachPart(hips, torsoGeometry, primary, 'hips', [0, 0.03, 0], [0.44, 0.27, 0.3]));

  const spine = addJoint(hips, 'spine', [0, 0.28, 0]);
  joints.spine = spine;
  pickables.push(attachPart(spine, torsoGeometry, primary, 'spine', [0, 0.55, 0], [0.5, 0.68, 0.3]));
  attachPart(spine, box, secondary, 'spine', [0, 0.24, 0.17], [0.58, 0.12, 0.065]);
  attachPart(spine, limb, skin, 'spine', [0, 1.18, 0], [0.48, 0.18, 0.48]);

  const head = addJoint(spine, 'head', [0, 1.28, 0]);
  joints.head = head;
  pickables.push(attachPart(head, sphere, skin, 'head', [0, 0.2, 0], [0.3, 0.39, 0.29]));
  attachPart(head, sphere, hair, 'head', [0, 0.31, -0.05], [0.33, 0.38, 0.3]);
  attachPart(head, sphere, skin, 'head', [0, 0.14, 0.105], [0.29, 0.31, 0.27]);
  attachPart(head, sphere, white, 'head', [-0.105, 0.24, 0.368], [0.048, 0.031, 0.023]);
  attachPart(head, sphere, white, 'head', [0.105, 0.24, 0.368], [0.048, 0.031, 0.023]);
  attachPart(head, sphere, dark, 'head', [-0.105, 0.24, 0.391], [0.021, 0.021, 0.016]);
  attachPart(head, sphere, dark, 'head', [0.105, 0.24, 0.391], [0.021, 0.021, 0.016]);
  attachPart(head, box, dark, 'head', [-0.105, 0.305, 0.366], [0.1, 0.018, 0.018]);
  attachPart(head, box, dark, 'head', [0.105, 0.305, 0.366], [0.1, 0.018, 0.018]);
  attachPart(head, sphere, skin, 'head', [0, 0.16, 0.387], [0.035, 0.06, 0.045]);
  attachPart(head, box, secondary, 'head', [0, 0.065, 0.376], [0.1, 0.025, 0.016]);

  function createArm(side) {
    const suffix = side === 'left' ? 'L' : 'R';
    const direction = side === 'left' ? -1 : 1;
    const upper = addJoint(spine, `upperArm${suffix}`, [direction * 0.5, 0.94, 0]);
    const lower = addJoint(upper, `lowerArm${suffix}`, [0, -0.76, 0]);
    const hand = addJoint(lower, `hand${suffix}`, [0, -0.67, 0]);
    joints[`upperArm${suffix}`] = upper;
    joints[`lowerArm${suffix}`] = lower;
    joints[`hand${suffix}`] = hand;
    pickables.push(attachPart(upper, limb, skin, `upperArm${suffix}`, [0, -0.38, 0], [1, 0.76, 1]));
    pickables.push(attachPart(lower, slimLimb, skin, `lowerArm${suffix}`, [0, -0.335, 0], [1, 0.67, 1]));
    pickables.push(attachPart(hand, sphere, skin, `hand${suffix}`, [0, -0.1, 0.03], [0.15, 0.22, 0.11]));
  }

  function createLeg(side) {
    const suffix = side === 'left' ? 'L' : 'R';
    const direction = side === 'left' ? -1 : 1;
    const upper = addJoint(hips, `upperLeg${suffix}`, [direction * 0.27, -0.18, 0]);
    const lower = addJoint(upper, `lowerLeg${suffix}`, [0, -1.03, 0]);
    const foot = addJoint(lower, `foot${suffix}`, [0, -0.94, 0]);
    joints[`upperLeg${suffix}`] = upper;
    joints[`lowerLeg${suffix}`] = lower;
    joints[`foot${suffix}`] = foot;
    pickables.push(attachPart(upper, limb, primary, `upperLeg${suffix}`, [0, -0.515, 0], [1.35, 1.03, 1.3]));
    pickables.push(attachPart(lower, slimLimb, skin, `lowerLeg${suffix}`, [0, -0.47, 0], [1.08, 0.94, 1.08]));
    pickables.push(attachPart(foot, box, dark, `foot${suffix}`, [0, -0.09, 0.15], [0.28, 0.18, 0.55]));
  }

  createArm('left');
  createArm('right');
  createLeg('left');
  createLeg('right');
  [skin, hair, primary, secondary, dark, white].forEach((material) => material.dispose());
  return { root, joints, pickables, mode: 'procedural' };
}

function findRigJoints(scene) {
  const joints = {};
  scene.traverse((object) => {
    if (!object.isBone) return;
    const key = canonicalTeacherBone(object.name);
    if (key && !joints[key]) joints[key] = object;
  });
  return joints;
}

function jointFromSkinnedIntersection(hit) {
  const mesh = hit?.object;
  const geometry = mesh?.geometry;
  const faceIndex = Number(hit?.faceIndex);
  const skinIndex = geometry?.getAttribute?.('skinIndex');
  const skinWeight = geometry?.getAttribute?.('skinWeight');
  if (!mesh?.isSkinnedMesh || !mesh.skeleton || !Number.isInteger(faceIndex) || !skinIndex || !skinWeight) return '';
  let best = { weight: -1, key: '' };
  for (let corner = 0; corner < 3; corner += 1) {
    const rawIndex = faceIndex * 3 + corner;
    const vertexIndex = geometry.index ? geometry.index.getX(rawIndex) : rawIndex;
    const indexes = [skinIndex.getX(vertexIndex), skinIndex.getY(vertexIndex), skinIndex.getZ(vertexIndex), skinIndex.getW(vertexIndex)];
    const weights = [skinWeight.getX(vertexIndex), skinWeight.getY(vertexIndex), skinWeight.getZ(vertexIndex), skinWeight.getW(vertexIndex)];
    for (let influence = 0; influence < 4; influence += 1) {
      const bone = mesh.skeleton.bones[Math.round(indexes[influence])];
      const key = canonicalTeacherBone(bone?.name);
      if (key && Number(weights[influence]) > best.weight) best = { weight: Number(weights[influence]), key };
    }
  }
  return best.key;
}

function disposeObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose?.();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach((material) => {
      Object.values(material).forEach((value) => {
        if (value?.isTexture) value.dispose();
      });
      material.dispose?.();
    });
  });
}

async function loadRiggedTeacher(modelUrl) {
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const gltf = await new GLTFLoader().loadAsync(modelUrl);
  const model = gltf.scene;
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3());
  const centre = bounds.getCenter(new THREE.Vector3());
  if (!Number.isFinite(size.y) || size.y <= 0) throw new Error('The rigged model has no measurable body.');
  const scale = 4.05 / size.y;
  const container = new THREE.Group();
  model.position.set(-centre.x, -centre.y, -centre.z);
  model.scale.setScalar(scale);
  container.add(model);
  const joints = findRigJoints(model);
  if (Object.keys(joints).length < 6) {
    disposeObject(model);
    throw new Error('The GLB skeleton does not expose enough recognisable human joints.');
  }
  const pickables = [];
  const hitRadius = size.y * 0.055;
  const hitMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.001, depthWrite: false });
  Object.entries(joints).forEach(([jointKey, joint]) => {
    const hit = new THREE.Mesh(new THREE.SphereGeometry(hitRadius, 10, 8), hitMaterial.clone());
    hit.userData.jointKey = jointKey;
    joint.add(hit);
    pickables.push(hit);
  });
  model.traverse((object) => {
    if (!object.isSkinnedMesh) return;
    object.computeBoundingSphere?.();
    pickables.push(object);
  });
  hitMaterial.dispose();
  return { root: container, joints, pickables, mode: 'rigged', source: model };
}

function captureRotations(joints) {
  return Object.fromEntries(Object.entries(joints).map(([key, joint]) => [key, {
    x: joint.rotation.x,
    y: joint.rotation.y,
    z: joint.rotation.z,
  }]));
}

function targetRotations(joints, baseRotations, poseId) {
  const pose = teacherRigPose(poseId);
  return Object.fromEntries(Object.keys(joints).map((key) => {
    const base = baseRotations[key] || { x: 0, y: 0, z: 0 };
    const change = pose.joints[key] || {};
    return [key, {
      x: base.x + Number(change.x || 0),
      y: base.y + Number(change.y || 0),
      z: base.z + Number(change.z || 0),
    }];
  }));
}

function easeInOut(value) {
  const t = Math.max(0, Math.min(1, value));
  return t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2;
}

export async function createTeacherThreeRuntime({ canvas, container, teacher, performanceTier = 'balanced', onStatus, onJointChange }) {
  const qualityTier = ['lite', 'balanced', 'full'].includes(performanceTier) ? performanceTier : 'balanced';
  const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: qualityTier === 'full',
      powerPreference: qualityTier === 'lite' ? 'low-power' : 'high-performance',
    });
  } catch (error) {
    throw new Error(`3D rendering is unavailable on this device: ${error.message}`);
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(39, 1, 0.1, 100);
  camera.position.set(0, 0.15, 7.7);
  camera.lookAt(0, 0, 0);
  const rigRoot = new THREE.Group();
  scene.add(rigRoot);

  scene.add(new THREE.HemisphereLight(0xded8ff, 0x17162b, 2.1));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.7);
  keyLight.position.set(3.2, 5.4, 5.8);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x8167ff, 2.2);
  rimLight.position.set(-4, 2.5, -2);
  scene.add(rimLight);
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(2.2, 48),
    new THREE.MeshBasicMaterial({ color: 0x6650ae, transparent: true, opacity: 0.13, depthWrite: false }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -2.02;
  scene.add(floor);

  renderer.setClearColor(0x000000, 0);
  const maximumPixelRatio = qualityTier === 'lite' ? 1 : qualityTier === 'balanced' ? 1.3 : 1.6;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maximumPixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  let rig;
  let disposed = false;
  let active = true;
  let frameId = 0;
  let lastFrame = 0;
  let poseTween = null;
  let jumpStartedAt = 0;
  let currentRootPose = { y: 0, x: 0, z: 0 };
  let movement = { x: 0, depth: 1 };
  let baseRotations = {};
  let pointerDrag = null;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  function resize() {
    if (disposed) return;
    const bounds = container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(bounds.width));
    const height = Math.max(1, Math.floor(bounds.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function setPose(poseId, motionKey = 0) {
    if (!rig) return;
    const now = performance.now();
    const pose = teacherRigPose(poseId);
    poseTween = {
      startedAt: now,
      duration: reduceMotion ? 1 : 380,
      from: captureRotations(rig.joints),
      to: targetRotations(rig.joints, baseRotations, poseId),
      rootFrom: { ...currentRootPose },
      rootTo: { ...pose.root },
    };
    if (poseId === 'jump' && motionKey >= 0 && !reduceMotion) jumpStartedAt = now;
  }

  function setTransform(next = {}) {
    movement = {
      x: Number(next.x || 0),
      depth: Math.max(0.7, Math.min(1.45, Number(next.depth) || 1)),
    };
  }

  function updatePose(now) {
    if (poseTween) {
      const progress = (now - poseTween.startedAt) / poseTween.duration;
      const eased = easeInOut(progress);
      Object.entries(rig.joints).forEach(([key, joint]) => {
        const from = poseTween.from[key];
        const to = poseTween.to[key];
        if (!from || !to) return;
        joint.rotation.set(
          THREE.MathUtils.lerp(from.x, to.x, eased),
          THREE.MathUtils.lerp(from.y, to.y, eased),
          THREE.MathUtils.lerp(from.z, to.z, eased),
        );
      });
      currentRootPose = {
        y: THREE.MathUtils.lerp(poseTween.rootFrom.y, poseTween.rootTo.y, eased),
        x: THREE.MathUtils.lerp(poseTween.rootFrom.x, poseTween.rootTo.x, eased),
        z: THREE.MathUtils.lerp(poseTween.rootFrom.z, poseTween.rootTo.z, eased),
      };
      if (progress >= 1) poseTween = null;
    }
    const jumpProgress = jumpStartedAt ? (now - jumpStartedAt) / 760 : 2;
    const jumpHeight = jumpProgress >= 0 && jumpProgress <= 1 ? Math.sin(jumpProgress * Math.PI) * 1.05 : 0;
    if (jumpProgress > 1) jumpStartedAt = 0;
    const width = Math.max(1, container.getBoundingClientRect().width);
    rigRoot.position.set(
      (movement.x / width) * 5,
      currentRootPose.y + jumpHeight,
      (movement.depth - 1) * 3.8,
    );
    rigRoot.rotation.set(currentRootPose.x, 0, currentRootPose.z);
  }

  function render(now) {
    if (disposed) return;
    frameId = window.requestAnimationFrame(render);
    if (!active) return;
    const minimumFrameMs = qualityTier === 'lite' ? 40 : qualityTier === 'balanced' ? 24 : 16;
    if (now - lastFrame < minimumFrameMs) return;
    lastFrame = now;
    if (rig) updatePose(now);
    renderer.render(scene, camera);
  }

  function pickJoint(event) {
    if (!rig) return null;
    const bounds = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(rig.pickables, true)
      .map((hit) => hit.object.userData.jointKey || jointFromSkinnedIntersection(hit))
      .find(Boolean) || null;
  }

  function pointerDown(event) {
    if (event.button !== 0) return;
    const jointKey = pickJoint(event);
    const joint = rig?.joints[jointKey];
    if (!joint) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    poseTween = null;
    pointerDrag = {
      pointerId: event.pointerId,
      jointKey,
      x: event.clientX,
      y: event.clientY,
      start: { x: joint.rotation.x, y: joint.rotation.y, z: joint.rotation.z },
    };
    canvas.classList.add('is-moving-joint');
    onJointChange?.(TEACHER_JOINT_LABELS[jointKey] || 'Joint');
  }

  function pointerMove(event) {
    if (!pointerDrag || pointerDrag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    const rotation = draggedJointRotation(
      pointerDrag.jointKey,
      pointerDrag.start,
      (event.clientX - pointerDrag.x) / Math.max(1, bounds.width),
      (event.clientY - pointerDrag.y) / Math.max(1, bounds.height),
    );
    rig.joints[pointerDrag.jointKey].rotation.set(rotation.x, rotation.y, rotation.z);
  }

  function pointerEnd(event) {
    if (!pointerDrag || pointerDrag.pointerId !== event.pointerId) return;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    pointerDrag = null;
    canvas.classList.remove('is-moving-joint');
    onJointChange?.('');
  }

  function contextLost(event) {
    event.preventDefault();
    active = false;
    onStatus?.({ mode: 'fallback', message: 'Portrait fallback' });
  }

  canvas.addEventListener('pointerdown', pointerDown);
  canvas.addEventListener('pointermove', pointerMove);
  canvas.addEventListener('pointerup', pointerEnd);
  canvas.addEventListener('pointercancel', pointerEnd);
  canvas.addEventListener('webglcontextlost', contextLost);
  const resizeObserver = window.ResizeObserver ? new window.ResizeObserver(resize) : null;
  if (resizeObserver) resizeObserver.observe(container);
  else window.addEventListener('resize', resize);
  const intersectionObserver = window.IntersectionObserver
    ? new window.IntersectionObserver((entries) => {
      active = entries.some((entry) => entry.isIntersecting);
    }, { rootMargin: '120px' })
    : null;
  intersectionObserver?.observe(container);
  resize();
  frameId = window.requestAnimationFrame(render);

  try {
    if (teacher.modelUrl && qualityTier !== 'lite') {
      try {
        rig = await loadRiggedTeacher(teacher.modelUrl);
        onStatus?.({ mode: 'rigged', message: 'Rigged 3D model' });
      } catch (error) {
        console.warn('Custom rigged teacher could not load; using procedural skeleton:', error);
        rig = createProceduralTeacher(teacher, qualityTier);
        onStatus?.({ mode: 'procedural', message: 'Built-in 3D skeleton' });
      }
    } else {
      rig = createProceduralTeacher(teacher, qualityTier);
      onStatus?.({
        mode: 'procedural',
        message: teacher.modelUrl && qualityTier === 'lite' ? 'Lightweight 3D skeleton' : 'Built-in 3D skeleton',
      });
    }
    if (disposed) {
      disposeObject(rig.root);
      return null;
    }
    rigRoot.add(rig.root);
    baseRotations = captureRotations(rig.joints);
    setPose('ready');
  } catch (error) {
    renderer.dispose();
    renderer.forceContextLoss();
    throw error;
  }

  return {
    setPose,
    setTransform,
    dispose() {
      disposed = true;
      window.cancelAnimationFrame(frameId);
      resizeObserver?.disconnect();
      if (!resizeObserver) window.removeEventListener('resize', resize);
      intersectionObserver?.disconnect();
      canvas.removeEventListener('pointerdown', pointerDown);
      canvas.removeEventListener('pointermove', pointerMove);
      canvas.removeEventListener('pointerup', pointerEnd);
      canvas.removeEventListener('pointercancel', pointerEnd);
      canvas.removeEventListener('webglcontextlost', contextLost);
      disposeObject(scene);
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
