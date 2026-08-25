const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-admin-bootstrap-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'owner@example.test';
process.env.ADMIN_PASSWORD = 'temporary-admin-password';
process.env.MUSCRIPTOR_ENABLED = 'false';

const { bootstrapAdminAccounts, readDb } = require('./server');

test('ADMIN_PASSWORD creates only missing admins and never overwrites an account', async (context) => {
  context.after(() => fs.rmSync(testDataDir, { recursive: true, force: true }));

  assert.deepEqual(await bootstrapAdminAccounts(), { created: 1 });
  const firstDb = await readDb();
  const admin = firstDb.users.find((user) => user.email === 'owner@example.test');
  assert.ok(admin);
  assert.equal(admin.mustChangePassword, true);
  assert.notEqual(admin.passwordHash, process.env.ADMIN_PASSWORD);
  assert.equal(admin.passwordHash.length, 128);

  const originalHash = admin.passwordHash;
  process.env.ADMIN_PASSWORD = 'a-different-temporary-password';
  assert.deepEqual(await bootstrapAdminAccounts(), { created: 0 });
  const unchangedAdmin = (await readDb()).users.find((user) => user.email === 'owner@example.test');
  assert.equal(unchangedAdmin.passwordHash, originalHash);
});
