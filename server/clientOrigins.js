const PRODUCTION_CLIENT_ORIGINS = Object.freeze([
  'https://polymathmusician67.com',
  'https://www.polymathmusician67.com',
]);

function normalizeOrigin(origin) {
  return String(origin || '').trim().replace(/\/+$/, '');
}

function buildClientOrigins(env = process.env) {
  const isProduction = String(env.NODE_ENV || '').trim().toLowerCase() === 'production';
  const origins = new Set(
    [env.CLIENT_ORIGIN || 'http://localhost:5173', ...String(env.CLIENT_ORIGINS || '').split(',')]
      .map(normalizeOrigin)
      .filter(Boolean),
  );

  if (isProduction) {
    for (const origin of PRODUCTION_CLIENT_ORIGINS) origins.add(origin);
  } else {
    origins.add('http://localhost:5173');
    origins.add('http://127.0.0.1:5173');
    origins.add('http://localhost:5174');
    origins.add('http://127.0.0.1:5174');
  }

  return origins;
}

function clientOriginAllowed(origin, allowedOrigins) {
  if (!origin) return true;
  const normalizedOrigin = normalizeOrigin(origin);
  return allowedOrigins.has(normalizedOrigin)
    || /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(normalizedOrigin);
}

module.exports = {
  PRODUCTION_CLIENT_ORIGINS,
  buildClientOrigins,
  clientOriginAllowed,
  normalizeOrigin,
};
