const assert = require('node:assert/strict');
const test = require('node:test');
const { createTaskPool, positiveInteger } = require('./taskPool');

test('task pool limits concurrent work and drains every queued task', async () => {
  const pool = createTaskPool(2);
  const releases = [];
  let active = 0;
  let peak = 0;

  const work = Array.from({ length: 5 }, (_, index) => pool.run(async () => {
    active += 1;
    peak = Math.max(peak, active);
    await new Promise((resolve) => releases.push(resolve));
    active -= 1;
    return index;
  }));

  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(pool.snapshot(), { active: 2, pending: 3, concurrency: 2 });
  assert.equal(peak, 2);

  while (releases.length || pool.snapshot().pending) {
    releases.splice(0).forEach((release) => release());
    await new Promise((resolve) => setImmediate(resolve));
  }

  assert.deepEqual(await Promise.all(work), [0, 1, 2, 3, 4]);
  assert.equal(peak, 2);
  assert.deepEqual(pool.snapshot(), { active: 0, pending: 0, concurrency: 2 });
});

test('task pool clamps unsafe concurrency values', () => {
  assert.equal(positiveInteger(0, 3), 1);
  assert.equal(positiveInteger('not-a-number', 3), 3);
  assert.equal(positiveInteger(100, 1, 8), 8);
});
