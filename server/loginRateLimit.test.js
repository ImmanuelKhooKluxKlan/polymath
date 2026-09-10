const test = require('node:test');
const assert = require('node:assert/strict');

const { createLoginRateLimiter } = require('./loginRateLimit');

function request(ip) {
  return {
    ip,
    get(name) {
      return name === 'cf-connecting-ip' ? ip : '';
    },
  };
}

test('blocks repeated failures for one account and network without storing identifiers', () => {
  const limiter = createLoginRateLimiter({
    LOGIN_RATE_WINDOW_MS: 60_000,
    LOGIN_RATE_PAIR_FAILURES: 3,
    LOGIN_RATE_ACCOUNT_FAILURES: 6,
  });
  const client = request('203.0.113.10');

  assert.equal(limiter.inspect(client, 'user@example.test', 1_000).allowed, true);
  limiter.recordFailure(client, 'user@example.test', 1_000);
  limiter.recordFailure(client, 'user@example.test', 2_000);
  const blocked = limiter.recordFailure(client, 'user@example.test', 3_000);

  assert.equal(blocked.allowed, false);
  assert.equal(blocked.retryAfterSeconds, 58);
});

test('account-wide ceiling still applies when an attacker rotates networks', () => {
  const limiter = createLoginRateLimiter({
    LOGIN_RATE_WINDOW_MS: 60_000,
    LOGIN_RATE_PAIR_FAILURES: 3,
    LOGIN_RATE_ACCOUNT_FAILURES: 4,
  });
  const identifier = '+6581234567';

  limiter.recordFailure(request('203.0.113.1'), '+65 8123 4567', 1_000);
  limiter.recordFailure(request('203.0.113.2'), '+65-8123-4567', 2_000);
  limiter.recordFailure(request('203.0.113.3'), identifier, 3_000);
  const blocked = limiter.recordFailure(request('203.0.113.4'), identifier, 4_000);

  assert.equal(blocked.allowed, false);
});

test('successful authentication resets the active failure window', () => {
  const limiter = createLoginRateLimiter({
    LOGIN_RATE_WINDOW_MS: 60_000,
    LOGIN_RATE_PAIR_FAILURES: 3,
  });
  const client = request('203.0.113.20');

  limiter.recordFailure(client, 'user@example.test', 1_000);
  limiter.recordFailure(client, 'user@example.test', 2_000);
  limiter.reset(client, 'user@example.test');

  assert.equal(limiter.inspect(client, 'user@example.test', 3_000).allowed, true);
});
