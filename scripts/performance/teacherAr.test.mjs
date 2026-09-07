import assert from 'node:assert/strict';
import test from 'node:test';
import {
  detectTeacherArCapabilities,
  pointFromStageTap,
  sanitizeArPoint,
  sanitizeTeacherArSettings,
  teacherArRoute,
} from '../../src/engine/teacherArEngine.js';

test('AR settings are bounded and untrusted values are removed', () => {
  assert.deepEqual(sanitizeTeacherArSettings({
    style: 'unknown',
    scale: 50,
    opacity: -2,
    pianoSide: 'ceiling',
    speaking: 1,
    cameraFrame: 'must-not-survive',
  }), {
    style: 'regular',
    scale: 1.65,
    opacity: 0.35,
    pianoSide: 'right',
    speaking: true,
  });
});
test('camera simulator taps become safe normalized coordinates', () => {
  const bounds = { left: 100, top: 50, width: 400, height: 200 };
  assert.deepEqual(pointFromStageTap({ clientX: 300, clientY: 150 }, bounds), { x: 0.5, y: 0.5 });
  assert.deepEqual(sanitizeArPoint({ x: -10, y: 10 }), { x: 0.08, y: 0.9 });
});

test('capability detection prefers spatial AR and never requests permission', async () => {
  let checkedMode = '';
  const capabilities = await detectTeacherArCapabilities({
    secureContext: true,
    windowRef: { isSecureContext: true },
    navigatorRef: {
      xr: { isSessionSupported: async (mode) => { checkedMode = mode; return true; } },
      mediaDevices: { getUserMedia() { throw new Error('must not be called'); } },
    },
  });
  assert.equal(checkedMode, 'immersive-ar');
  assert.equal(capabilities.spatialAr, true);
  assert.equal(capabilities.cameraPreview, true);
  assert.equal(capabilities.recommendedMode, 'spatial');
});

test('AR route carries only bounded display configuration', () => {
  assert.equal(
    teacherArRoute({ style: 'bikini', channel: 'lesson-1', simulate: true }),
    'teacher-ar?style=bikini&channel=lesson-1&simulate=1',
  );
});
