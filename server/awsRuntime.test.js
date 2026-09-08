const assert = require('node:assert/strict');
const test = require('node:test');
const {
  applyManagedDatabaseCredentials,
  applyMissingEnvironment,
  parseSecret,
  withoutManagedDatabaseEnvironment,
} = require('./startAwsRuntime');
const { createAwsRdsPasswordProvider } = require('./awsSecrets');

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

test('generic runtime secrets cannot override the managed RDS connection', () => {
  const target = {
    CLIENT_ORIGIN: 'https://polymathmusician67.com',
    DATABASE_URL: 'postgres://stale:stale@old.example.invalid/polymath',
    PGUSER: 'stale-user',
    PGPASSWORD: 'stale-password',
  };
  const runtime = withoutManagedDatabaseEnvironment({
    API_KEY: 'runtime-secret',
    DATABASE_URL: 'postgres://also-stale.example.invalid/polymath',
    PGUSER: 'also-stale-user',
    PGPASSWORD: 'also-stale-password',
  });

  applyMissingEnvironment(runtime, target);
  applyManagedDatabaseCredentials({ username: 'polymath_admin', password: 'current-password' }, target);

  assert.equal(target.API_KEY, 'runtime-secret');
  assert.equal(target.PGUSER, 'polymath_admin');
  assert.equal(target.PGPASSWORD, 'current-password');
  assert.equal(target.DATABASE_URL, undefined);
});

test('RDS password provider reads the current secret for every new connection', async () => {
  let reads = 0;
  const client = {
    async send() {
      reads += 1;
      return {
        SecretString: JSON.stringify({
          username: 'polymath_admin',
          password: reads === 1 ? 'first-password' : 'rotated-password',
        }),
      };
    },
  };
  const provider = createAwsRdsPasswordProvider({
    client,
    secretId: 'arn:aws:secretsmanager:example',
    expectedUsername: 'polymath_admin',
  });

  assert.equal(await provider(), 'first-password');
  assert.equal(await provider(), 'rotated-password');
  assert.equal(reads, 2);
});
