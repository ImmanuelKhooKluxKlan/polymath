const crypto = require('node:crypto');

function boundedInteger(value, minimum, maximum, fallback) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function digest(value) {
  return crypto.createHash('sha256').update(String(value || '')).digest('hex');
}

function networkHint(request) {
  const cloudflare = request?.get?.('cf-connecting-ip');
  const forwarded = request?.get?.('x-forwarded-for');
  const firstForwarded = String(forwarded || '').split(',')[0].trim();
  return String(
    cloudflare
    || firstForwarded
    || request?.ip
    || request?.socket?.remoteAddress
    || 'unknown',
  ).trim().slice(0, 96);
}

function normalizedIdentifier(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (/^\+?[\d\s().-]+$/.test(normalized)) {
    return normalized.replace(/[\s().-]/g, '');
  }
  return normalized;
}

function createLoginRateLimiter(environment = process.env) {
  const windowMs = boundedInteger(
    environment.LOGIN_RATE_WINDOW_MS,
    60_000,
    60 * 60_000,
    15 * 60_000,
  );
  const pairLimit = boundedInteger(
    environment.LOGIN_RATE_PAIR_FAILURES,
    3,
    30,
    8,
  );
  const accountLimit = boundedInteger(
    environment.LOGIN_RATE_ACCOUNT_FAILURES,
    pairLimit,
    100,
    24,
  );
  const maximumKeys = boundedInteger(
    environment.LOGIN_RATE_MAX_KEYS,
    1_000,
    100_000,
    20_000,
  );
  const buckets = new Map();

  function keys(request, identifier) {
    const account = digest(normalizedIdentifier(identifier));
    const network = digest(networkHint(request));
    return {
      pair: `pair:${network}:${account}`,
      account: `account:${account}`,
    };
  }

  function recentFailures(key, now) {
    const cutoff = now - windowMs;
    const recent = (buckets.get(key) || []).filter((timestamp) => timestamp > cutoff);
    if (recent.length) buckets.set(key, recent);
    else buckets.delete(key);
    return recent;
  }

  function keepBounded(now) {
    if (buckets.size <= maximumKeys) return;
    for (const key of buckets.keys()) {
      recentFailures(key, now);
      if (buckets.size <= maximumKeys) return;
    }
    while (buckets.size > maximumKeys) {
      const oldestKey = buckets.keys().next().value;
      if (oldestKey === undefined) break;
      buckets.delete(oldestKey);
    }
  }

  function inspect(request, identifier, now = Date.now()) {
    const loginKeys = keys(request, identifier);
    const pair = recentFailures(loginKeys.pair, now);
    const account = recentFailures(loginKeys.account, now);
    const blocked = pair.length >= pairLimit || account.length >= accountLimit;
    const oldestBlockingFailure = pair.length >= pairLimit ? pair[0] : account[0];
    return {
      allowed: !blocked,
      retryAfterSeconds: blocked
        ? Math.max(1, Math.ceil((oldestBlockingFailure + windowMs - now) / 1_000))
        : 0,
    };
  }

  function recordFailure(request, identifier, now = Date.now()) {
    const loginKeys = keys(request, identifier);
    for (const key of [loginKeys.pair, loginKeys.account]) {
      const failures = recentFailures(key, now);
      failures.push(now);
      buckets.set(key, failures);
    }
    keepBounded(now);
    return inspect(request, identifier, now);
  }

  function reset(request, identifier) {
    const loginKeys = keys(request, identifier);
    buckets.delete(loginKeys.pair);
    buckets.delete(loginKeys.account);
  }

  return {
    inspect,
    recordFailure,
    reset,
  };
}

module.exports = {
  createLoginRateLimiter,
};
