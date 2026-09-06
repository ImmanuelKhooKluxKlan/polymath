import assert from 'node:assert/strict';
import test from 'node:test';
import {
  sanitizeProductProperties,
  uploadSizeBucket,
} from '../../src/services/productAnalytics.js';

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
