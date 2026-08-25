const assert = require('node:assert/strict');
const test = require('node:test');
const { applyMissingEnvironment, parseSecret } = require('./startAwsRuntime');

test('AWS runtime secret parsing accepts JSON objects and rejects arrays', () => {
  assert.deepEqual(parseSecret({ SecretString: '{"API_KEY":"secret"}' }), { API_KEY: 'secret' });
  assert.throws(() => parseSecret({ SecretString: '[]' }), /JSON object/);
});

test('AWS runtime secrets never overwrite explicit task environment values', () => {
  const target = { CLIENT_ORIGIN: 'https://polymathmusician67.com' };
  applyMissingEnvironment({ CLIENT_ORIGIN: 'http://localhost:5173', API_KEY: 'secret' }, target);
  assert.equal(target.CLIENT_ORIGIN, 'https://polymathmusician67.com');
  assert.equal(target.API_KEY, 'secret');
});
