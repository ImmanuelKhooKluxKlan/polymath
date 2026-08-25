const assert = require('node:assert/strict');
const test = require('node:test');
const { StateConflictError, mergeDocuments } = require('./stateStore');

test('three-way merge preserves independent records from regional writers', () => {
  const base = { users: [{ id: 'u1', name: 'One', mcoins: 0 }], sessions: [], settings: { tax: 25 } };
  const current = structuredClone(base);
  current.users.push({ id: 'u2', name: 'America', mcoins: 2 });
  const proposed = structuredClone(base);
  proposed.users.push({ id: 'u3', name: 'Singapore', mcoins: 3 });

  const merged = mergeDocuments(base, current, proposed);
  assert.deepEqual(merged.users.map((user) => user.id), ['u1', 'u2', 'u3']);
});

test('three-way merge combines independent fields on the same record', () => {
  const base = { users: [{ id: 'u1', name: 'Original', mcoins: 0 }], sessions: [] };
  const current = structuredClone(base);
  current.users[0].name = 'Updated name';
  const proposed = structuredClone(base);
  proposed.users[0].mcoins = 10;

  const merged = mergeDocuments(base, current, proposed);
  assert.equal(merged.users[0].name, 'Updated name');
  assert.equal(merged.users[0].mcoins, 10);
});

test('three-way merge rejects conflicting balance updates instead of losing money', () => {
  const base = { users: [{ id: 'u1', mcoins: 10 }], sessions: [] };
  const current = { users: [{ id: 'u1', mcoins: 8 }], sessions: [] };
  const proposed = { users: [{ id: 'u1', mcoins: 15 }], sessions: [] };

  assert.throws(
    () => mergeDocuments(base, current, proposed),
    (error) => error instanceof StateConflictError && error.status === 409,
  );
});

test('session rows use token hashes as stable merge identifiers', () => {
  const base = { users: [], sessions: [] };
  const current = { users: [], sessions: [{ tokenHash: 'america', userId: 'u1' }] };
  const proposed = { users: [], sessions: [{ tokenHash: 'singapore', userId: 'u2' }] };

  const merged = mergeDocuments(base, current, proposed);
  assert.deepEqual(merged.sessions.map((session) => session.tokenHash), ['america', 'singapore']);
});
