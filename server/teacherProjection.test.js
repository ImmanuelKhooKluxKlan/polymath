'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { createTeacherProjectionStore, normalizeState } = require('./teacherProjection');

test('normalizes projection state to the small display-only contract', () => {
  assert.deepEqual(normalizeState({
    mode: 'hologram', style: 'bikini', speaking: 1, visible: true,
    caption: 'x'.repeat(400), cameraFrame: 'must-not-survive',
  }), {
    mode: 'hologram', style: 'bikini', speaking: true, visible: true,
    caption: 'x'.repeat(280), version: 1,
  });
});

test('requires the owner to update and an unguessable token to read', async () => {
  const store = createTeacherProjectionStore();
  const created = await store.create('owner-1', { mode: 'wall' });
  assert.equal(await store.read(created.id, 'wrong'), null);
  assert.equal(await store.update('owner-2', created.id, { speaking: true }), null);
  const updated = await store.update('owner-1', created.id, { mode: 'hologram', speaking: true });
  assert.equal(updated.version, 2);
  assert.equal((await store.read(created.id, created.token)).state.mode, 'hologram');
});

test('expires sessions and lets only the owner close them', async () => {
  let timestamp = 1_000_000;
  const store = createTeacherProjectionStore({ now: () => timestamp, ttlMs: 60_000 });
  const first = await store.create('owner-1');
  assert.equal(await store.close('owner-2', first.id), false);
  assert.ok(await store.read(first.id, first.token));
  timestamp += 60_001;
  assert.equal(await store.read(first.id, first.token), null);
  const second = await store.create('owner-1');
  assert.equal(await store.close('owner-1', second.id), true);
});
