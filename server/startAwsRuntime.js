const { SecretsManagerClient } = require('@aws-sdk/client-secrets-manager');
const {
  databaseCredentials,
  parseSecret,
  readSecret,
} = require('./awsSecrets');

const MANAGED_DATABASE_ENVIRONMENT_KEYS = new Set([
  'DATABASE_URL',
  'PGDATABASE',
  'PGHOST',
  'PGPASSWORD',
  'PGPORT',
  'PGUSER',
]);

function applyMissingEnvironment(values, target = process.env) {
  for (const [key, value] of Object.entries(values)) {
    if (!/^[A-Z][A-Z0-9_]*$/.test(key)) continue;
    if (target[key] === undefined || target[key] === '') target[key] = String(value ?? '');
  }
  return target;
}

function withoutManagedDatabaseEnvironment(values = {}) {
  return Object.fromEntries(
    Object.entries(values).filter(([key]) => !MANAGED_DATABASE_ENVIRONMENT_KEYS.has(key)),
  );
}

function applyManagedDatabaseCredentials(secret, target = process.env) {
  const credentials = databaseCredentials(secret);
  // DATABASE_URL embeds a password and takes precedence over separate PG fields.
  // In AWS, the RDS-managed secret must be the only credential source.
  delete target.DATABASE_URL;
  target.PGUSER = credentials.username;
  target.PGPASSWORD = credentials.password;
  return target;
}

async function loadAwsEnvironment() {
  const client = new SecretsManagerClient({
    region: process.env.AWS_SECRET_REGION || process.env.AWS_REGION || 'us-east-2',
  });
  const databaseSecretId = process.env.AWS_RDS_SECRET_ARN;
  const runtime = await readSecret(client, process.env.AWS_RUNTIME_SECRET_ARN);
  applyMissingEnvironment(
    databaseSecretId ? withoutManagedDatabaseEnvironment(runtime) : runtime,
  );

  if (databaseSecretId) {
    const database = await readSecret(client, databaseSecretId);
    applyManagedDatabaseCredentials(database);
  }
}

async function start() {
  await loadAwsEnvironment();

  const { startServer } = require('./server');
  await startServer();
}

if (require.main === module) {
  start().catch((error) => {
    console.error('Polymath AWS runtime failed to start:', error.message || error);
    process.exitCode = 1;
  });
}

module.exports = {
  applyManagedDatabaseCredentials,
  applyMissingEnvironment,
  loadAwsEnvironment,
  parseSecret,
  readSecret,
  start,
  withoutManagedDatabaseEnvironment,
};
