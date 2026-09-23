const assert = require('node:assert/strict');
const test = require('node:test');

const {
  DEFAULT_ORIGINS,
  allowedOrigins,
  mergeReadCorsRule,
} = require('./configureInstrumentCors');

test('default instrument origins include both public hostnames', () => {
  const origins = allowedOrigins();
  assert.deepEqual(origins, DEFAULT_ORIGINS);
  assert.ok(origins.includes('https://polymathmusician67.com'));
  assert.ok(origins.includes('https://www.polymathmusician67.com'));
});

test('read CORS merge preserves unrelated rules and existing origins', () => {
  const uploadRule = {
    allowed: { methods: ['PUT'], origins: ['https://uploader.example'] },
  };
  const readRule = {
    allowed: {
      methods: ['GET'],
      headers: ['Range'],
      origins: ['https://polymathmusician67.com'],
    },
    exposeHeaders: ['ETag'],
    maxAgeSeconds: 60,
  };
  const merged = mergeReadCorsRule(
    [uploadRule, readRule],
    ['https://www.polymathmusician67.com'],
  );
  assert.deepEqual(merged[0], uploadRule);
  assert.deepEqual(merged[1].allowed.methods, ['GET', 'HEAD']);
  assert.deepEqual(merged[1].allowed.headers, ['Range']);
  assert.deepEqual(merged[1].allowed.origins, [
    'https://polymathmusician67.com',
    'https://www.polymathmusician67.com',
  ]);
  assert.ok(merged[1].exposeHeaders.includes('Content-Length'));
  assert.equal(merged[1].maxAgeSeconds, 86400);
});
