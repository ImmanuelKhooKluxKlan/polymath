const test = require('node:test');
const assert = require('node:assert/strict');
const {
  SiteConfigurationError,
  adminSiteConfiguration,
  defaultSiteConfiguration,
  normalizeSiteConfiguration,
  publicSiteConfiguration,
  updateSiteConfiguration,
} = require('./siteConfiguration');

function database() {
  return {
    siteConfiguration: defaultSiteConfiguration(),
    siteConfigurationEvents: [],
  };
}

test('site configuration exposes editable labels without exposing administrator metadata', () => {
  const db = database();
  db.siteConfiguration.updatedBy = 'secret-admin-id';
  const publicResult = publicSiteConfiguration(db);
  assert.equal(publicResult.brand.name, 'Polymath');
  assert.equal(publicResult.navigation.find((item) => item.id === 'your-songs').access, 'signed-in');
  assert.equal(Object.hasOwn(publicResult, 'updatedBy'), false);
  assert.equal(Object.hasOwn(publicResult.navigation[0], 'path'), false);
});

test('administrator can rename, regroup, reorder, and hide registered navigation pages', () => {
  const db = database();
  const input = structuredClone(db.siteConfiguration);
  input.brand = { name: 'Polymath <script>', suffix: 'Studio' };
  input.announcement = { enabled: true, text: 'New lessons are live!', tone: 'success' };
  input.navigation = input.navigation.map((item) => (
    item.id === 'create-music'
      ? { ...item, label: 'Song Lab', group: 'more', order: 95, visible: false }
      : item
  ));

  const result = updateSiteConfiguration(db, input, 'admin-1', new Date('2026-09-07T01:02:03.000Z'));
  assert.equal(result.configuration.revision, 2);
  assert.equal(result.configuration.brand.name, 'Polymath script');
  assert.deepEqual(
    result.configuration.navigation.find((item) => item.id === 'create-music'),
    { id: 'create-music', label: 'Song Lab', group: 'more', order: 95, visible: false },
  );
  assert.equal(db.siteConfigurationEvents.length, 1);
  assert.equal(db.siteConfigurationEvents[0].actorId, 'admin-1');
  assert.ok(db.siteConfigurationEvents[0].changes.some((change) => change.field === 'create-music label'));
});

test('site configuration rejects stale revisions and structural navigation tampering', () => {
  const db = database();
  const stale = structuredClone(db.siteConfiguration);
  stale.revision = 0;
  assert.throws(
    () => updateSiteConfiguration(db, stale, 'admin-1'),
    (error) => error instanceof SiteConfigurationError && error.status === 409,
  );

  const unknown = structuredClone(db.siteConfiguration);
  unknown.navigation.push({ id: 'admin-database', label: 'Public admin', group: 'primary', order: 1, visible: true });
  assert.throws(() => updateSiteConfiguration(db, unknown, 'admin-1'), /Unknown navigation page/);

  const duplicate = structuredClone(db.siteConfiguration);
  duplicate.navigation.push({ ...duplicate.navigation[0] });
  assert.throws(() => updateSiteConfiguration(db, duplicate, 'admin-1'), /appear only once/);

  const missing = structuredClone(db.siteConfiguration);
  missing.navigation = missing.navigation.filter((item) => item.id !== 'studio');
  assert.throws(() => updateSiteConfiguration(db, missing, 'admin-1'), /missing the permanent studio page/);
});

test('site configuration limits the main menu and requires announcement text', () => {
  const db = database();
  const crowded = structuredClone(db.siteConfiguration);
  crowded.navigation = crowded.navigation.map((item, index) => ({
    ...item,
    group: index < 6 ? 'primary' : item.group,
  }));
  assert.throws(() => updateSiteConfiguration(db, crowded, 'admin-1'), /at most five visible pages/);

  const blankAnnouncement = structuredClone(db.siteConfiguration);
  blankAnnouncement.announcement = { enabled: true, text: '   ', tone: 'info' };
  assert.throws(() => updateSiteConfiguration(db, blankAnnouncement, 'admin-1'), /Write an announcement/);
});

test('normalization restores every immutable route and admin history is newest first', () => {
  const normalized = normalizeSiteConfiguration({
    navigation: [{ id: 'studio', label: '', group: 'broken', order: -10, visible: false }],
  });
  assert.equal(normalized.navigation.length, 9);
  assert.equal(normalized.navigation.find((item) => item.id === 'studio').label, 'Piano');
  assert.equal(normalized.navigation.find((item) => item.id === 'studio').group, 'primary');

  const db = database();
  const first = structuredClone(db.siteConfiguration);
  first.brand.name = 'First';
  updateSiteConfiguration(db, first, 'admin-1', new Date('2026-09-07T01:00:00.000Z'));
  const second = structuredClone(db.siteConfiguration);
  second.brand.name = 'Second';
  updateSiteConfiguration(db, second, 'admin-2', new Date('2026-09-07T02:00:00.000Z'));
  const adminResult = adminSiteConfiguration(db);
  assert.equal(adminResult.history[0].actorId, 'admin-2');
  assert.equal(adminResult.configuration.brand.name, 'Second');
});
