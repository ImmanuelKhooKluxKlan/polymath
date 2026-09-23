import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isRetryableRouteLoadError,
  loadRouteWithRetry,
  routeLoadingDetail,
  routeLoadingProgress,
} from '../../src/utils/routeLoading.js';

test('route loading progress advances slowly and waits below completion', () => {
  const samples = [0, 5000, 15000, 30000, 60000, 180000].map(routeLoadingProgress);
  assert.deepEqual([...samples].sort((left, right) => left - right), samples);
  assert.equal(samples[0], 7);
  assert.ok(samples[2] < 60);
  assert.equal(samples.at(-1), 92);
  assert.match(routeLoadingDetail(9000), /slower connection/i);
  assert.match(routeLoadingDetail(35000), /still loading/i);
});

test('route loader retries transient chunk failures before succeeding', async () => {
  let attempts = 0;
  const waited = [];
  const loaded = await loadRouteWithRetry(async () => {
    attempts += 1;
    if (attempts < 3) throw new TypeError('Failed to fetch dynamically imported module');
    return { default: 'Loaded page' };
  }, {
    retryDelaysMs: [10, 20, 30],
    waitFor: async (milliseconds) => waited.push(milliseconds),
  });

  assert.equal(attempts, 3);
  assert.deepEqual(waited, [10, 20]);
  assert.equal(loaded.default, 'Loaded page');
});

test('route loader does not hide genuine rendering or programming errors', async () => {
  let attempts = 0;
  await assert.rejects(
    loadRouteWithRetry(async () => {
      attempts += 1;
      throw new Error('Component crashed while rendering');
    }, {
      retryDelaysMs: [0, 0],
      waitFor: async () => {},
    }),
    /Component crashed/,
  );
  assert.equal(attempts, 1);
  assert.equal(isRetryableRouteLoadError(new Error('Component crashed')), false);
});
