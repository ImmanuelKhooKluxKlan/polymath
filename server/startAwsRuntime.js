const {
  GetSecretValueCommand,
  SecretsManagerClient,
} = require('@aws-sdk/client-secrets-manager');

function parseSecret(response) {
  const text = response.SecretString || Buffer.from(response.SecretBinary || '', 'base64').toString('utf8');
  const parsed = JSON.parse(text || '{}');
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('AWS runtime secret must contain a JSON object.');
  }
  return parsed;
}

function applyMissingEnvironment(values, target = process.env) {
  for (const [key, value] of Object.entries(values)) {
    if (!/^[A-Z][A-Z0-9_]*$/.test(key)) continue;
    if (target[key] === undefined || target[key] === '') target[key] = String(value ?? '');
  }
  return target;
}

async function readSecret(client, secretId) {
  if (!secretId) return {};
  return parseSecret(await client.send(new GetSecretValueCommand({ SecretId: secretId })));
}

async function loadAwsEnvironment() {
  const client = new SecretsManagerClient({
    region: process.env.AWS_SECRET_REGION || process.env.AWS_REGION || 'us-east-2',
  });
  const runtime = await readSecret(client, process.env.AWS_RUNTIME_SECRET_ARN);
  applyMissingEnvironment(runtime);

  const database = await readSecret(client, process.env.AWS_RDS_SECRET_ARN);
  applyMissingEnvironment({
    PGUSER: database.username,
    PGPASSWORD: database.password,
  });
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

module.exports = { applyMissingEnvironment, loadAwsEnvironment, parseSecret, readSecret, start };
