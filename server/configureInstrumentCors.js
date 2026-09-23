const API_BASE = 'https://api.cloudflare.com/client/v4';

const DEFAULT_ORIGINS = Object.freeze([
  'https://polymathmusician67.com',
  'https://www.polymathmusician67.com',
  'https://polymath-musician.pages.dev',
  'https://scaling-preview.polymath-musician.pages.dev',
  'http://localhost:5173',
  'http://localhost:5174',
  'http://127.0.0.1:5173',
  'http://127.0.0.1:5174',
]);

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function allowedOrigins(value = '') {
  const configured = String(value)
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
  return unique(configured.length ? configured : DEFAULT_ORIGINS);
}

function mergeReadCorsRule(rules = [], origins = DEFAULT_ORIGINS) {
  const nextRules = Array.isArray(rules) ? rules.map((rule) => ({ ...rule })) : [];
  const index = nextRules.findIndex((rule) => (
    Array.isArray(rule?.allowed?.methods)
    && rule.allowed.methods.includes('GET')
  ));
  const current = index >= 0 ? nextRules[index] : {};
  const replacement = {
    ...current,
    allowed: {
      ...(current.allowed || {}),
      origins: unique([...(current.allowed?.origins || []), ...origins]),
      methods: unique([...(current.allowed?.methods || []), 'GET', 'HEAD']),
      headers: current.allowed?.headers?.length ? current.allowed.headers : ['*'],
    },
    exposeHeaders: unique([
      ...(current.exposeHeaders || []),
      'ETag',
      'Content-Length',
      'Content-Range',
      'Accept-Ranges',
    ]),
    maxAgeSeconds: Math.max(86400, Number(current.maxAgeSeconds) || 0),
  };

  if (index >= 0) nextRules[index] = replacement;
  else nextRules.push(replacement);
  return nextRules;
}

async function cloudflareRequest(path, token, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.success === false) {
    const message = payload.errors?.map((error) => error.message).join('; ')
      || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

async function main() {
  const token = String(process.env.CLOUDFLARE_API_TOKEN || '').trim();
  const accountId = String(process.env.CLOUDFLARE_ACCOUNT_ID || '').trim();
  const bucket = String(process.env.INSTRUMENT_R2_BUCKET || '').trim();
  if (!token || !accountId || !bucket) {
    throw new Error(
      'CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, and INSTRUMENT_R2_BUCKET are required.',
    );
  }

  const path = `/accounts/${accountId}/r2/buckets/${bucket}/cors`;
  const current = await cloudflareRequest(path, token);
  const origins = allowedOrigins(process.env.INSTRUMENT_ALLOWED_ORIGINS);
  const rules = mergeReadCorsRule(current.result?.rules, origins);
  await cloudflareRequest(path, token, {
    method: 'PUT',
    body: JSON.stringify({ rules }),
  });
  console.log(`Instrument CORS ready for ${origins.length} approved origins.`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Instrument CORS configuration failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = {
  DEFAULT_ORIGINS,
  allowedOrigins,
  mergeReadCorsRule,
};
