import assert from 'node:assert/strict';
import test from 'node:test';

const storage = new Map();

function installDevice({ width, height, coarse, platform, cores, memory }) {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      screen: { width, height },
      innerWidth: width,
      innerHeight: height,
      devicePixelRatio: coarse ? 3 : 1,
      matchMedia: () => ({ matches: coarse }),
      localStorage: {
        getItem: (key) => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, value),
      },
    },
  });
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      platform,
      hardwareConcurrency: cores,
      deviceMemory: memory,
    },
  });
}

installDevice({
  width: 390,
  height: 844,
  coarse: true,
  platform: 'iPhone',
  cores: 6,
  memory: undefined,
});

const performanceModule = await import('../../src/engine/devicePerformance.js');

test('a phone starts conservatively and can never be promoted to the desktop tier', () => {
  storage.clear();
  installDevice({ width: 390, height: 844, coarse: true, platform: 'iPhone', cores: 6 });
  assert.equal(performanceModule.detectDeviceClass(), 'phone');
  assert.equal(performanceModule.getInitialPerformanceTier(), 'lite');
  assert.equal(performanceModule.capTierForDevice('full'), 'balanced');
});

test('a tablet is capped at balanced while a computer can use full', () => {
  storage.clear();
  installDevice({ width: 820, height: 1180, coarse: true, platform: 'iPad', cores: 8 });
  assert.equal(performanceModule.detectDeviceClass(), 'tablet');
  assert.equal(performanceModule.capTierForDevice('full'), 'balanced');

  installDevice({ width: 1920, height: 1080, coarse: false, platform: 'Win32', cores: 20, memory: 32 });
  assert.equal(performanceModule.detectDeviceClass(), 'desktop');
  assert.equal(performanceModule.getInitialPerformanceTier(), 'full');
  assert.equal(performanceModule.capTierForDevice('full'), 'full');
});

test('a saved phone profile is reused and remains device-capped', () => {
  storage.clear();
  installDevice({ width: 390, height: 844, coarse: true, platform: 'iPhone', cores: 6 });
  performanceModule.savePerformanceProfile('full', { calibration: { averageFps: 60 } });
  const saved = performanceModule.readSavedPerformanceProfile();
  assert.equal(saved.deviceClass, 'phone');
  assert.equal(saved.tier, 'balanced');
  assert.equal(saved.measurements.calibration.averageFps, 60);
});
