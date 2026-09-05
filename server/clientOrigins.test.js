const test = require('node:test');
const assert = require('node:assert/strict');
const {
  buildClientOrigins,
  clientOriginAllowed,
  normalizeOrigin,
} = require('./clientOrigins');

test('production accepts both public Polymath hostnames', () => {
  const origins = buildClientOrigins({
    NODE_ENV: 'production',
    CLIENT_ORIGIN: 'https://polymathmusician67.com',
  });

  assert.equal(clientOriginAllowed('https://polymathmusician67.com', origins), true);
  assert.equal(clientOriginAllowed('https://www.polymathmusician67.com', origins), true);
});

test('configured preview origins are normalized without accepting lookalike hosts', () => {
  const origins = buildClientOrigins({
    NODE_ENV: 'production',
    CLIENT_ORIGIN: 'https://polymathmusician67.com/',
    CLIENT_ORIGINS: ' https://polymath-musician.pages.dev/ ',
  });

  assert.equal(normalizeOrigin('https://polymathmusician67.com///'), 'https://polymathmusician67.com');
  assert.equal(clientOriginAllowed('https://polymath-musician.pages.dev', origins), true);
  assert.equal(clientOriginAllowed('https://polymathmusician67.com.attacker.example', origins), false);
});
