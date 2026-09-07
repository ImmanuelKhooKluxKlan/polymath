import assert from 'node:assert/strict';
import test from 'node:test';
import {
  readCampaignAttribution,
  rememberCampaignAttribution,
  sanitizeProductProperties,
  uploadSizeBucket,
} from '../../src/services/productAnalytics.js';

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

test('browser product events discard private and unexpected fields', () => {
  assert.deepEqual(sanitizeProductProperties({
    score: 92,
    freePreview: true,
    email: 'private@example.com',
    filename: 'private-song.mp3',
    songTitle: 'Private Song',
  }), {
    score: 92,
    freePreview: true,
  });
});

test('upload size buckets provide useful metrics without recording exact files', () => {
  assert.equal(uploadSizeBucket(500_000), '<1MB');
  assert.equal(uploadSizeBucket(5 * 1024 * 1024), '1-10MB');
  assert.equal(uploadSizeBucket(25 * 1024 * 1024), '10-100MB');
  assert.equal(uploadSizeBucket(700 * 1024 * 1024), '500MB+');
});

test('campaign attribution keeps the first valid creator touch for 30 days', (context) => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    localStorage: memoryStorage(),
    sessionStorage: memoryStorage(),
    navigator: {},
  };
  context.after(() => { globalThis.window = previousWindow; });
  const first = rememberCampaignAttribution({
    id: 'artist_campaign_12345678',
    slug: 'artist-first-song',
    referralCode: 'FIRST20',
  }, 'FIRST20', 1_000);
  const second = rememberCampaignAttribution({
    id: 'artist_campaign_87654321',
    slug: 'artist-second-song',
    referralCode: 'SECOND20',
  }, 'SECOND20', 2_000);
  assert.equal(first.campaignSlug, 'artist-first-song');
  assert.deepEqual(second, first);
  assert.deepEqual(readCampaignAttribution(1_000 + (31 * 24 * 60 * 60 * 1000)), {});
});
