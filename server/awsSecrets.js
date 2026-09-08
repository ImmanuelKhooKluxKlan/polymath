'use strict';

const {
  GetSecretValueCommand,
  SecretsManagerClient,
} = require('@aws-sdk/client-secrets-manager');

function parseSecret(response = {}) {
  const text = response.SecretString
    || Buffer.from(response.SecretBinary || '', 'base64').toString('utf8');
  const parsed = JSON.parse(text || '{}');
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('AWS runtime secret must contain a JSON object.');
  }
  return parsed;
}

async function readSecret(client, secretId) {
  if (!secretId) return {};
  return parseSecret(await client.send(new GetSecretValueCommand({ SecretId: secretId })));
}

function databaseCredentials(secret = {}) {
  const username = String(secret.username || '').trim();
  const password = String(secret.password || '');
  if (!username || !password) {
    throw new Error('AWS RDS secret is missing its username or password.');
  }
  return { username, password };
}

function createAwsRdsCredentialProvider({
  client,
  expectedUsername,
  region,
  secretId,
} = {}) {
  const resolvedSecretId = String(secretId || '').trim();
  if (!resolvedSecretId) return null;

  const secretsClient = client || new SecretsManagerClient({ region });
  const requiredUsername = String(expectedUsername || '').trim();
  let activeRead = null;

  return async function currentRdsCredentials() {
    if (!activeRead) {
      activeRead = readSecret(secretsClient, resolvedSecretId)
        .then(databaseCredentials);
    }
    const request = activeRead;

    try {
      const credentials = await request;
      if (requiredUsername && credentials.username !== requiredUsername) {
        throw new Error('AWS RDS secret username does not match the configured database user.');
      }
      return credentials;
    } finally {
      // Share simultaneous reads, but fetch again for the next connection so
      // an RDS-managed password rotation does not require an application restart.
      if (activeRead === request) activeRead = null;
    }
  };
}

function createAwsRdsPasswordProvider(options = {}) {
  const credentialProvider = createAwsRdsCredentialProvider(options);
  if (!credentialProvider) return null;
  return async () => (await credentialProvider()).password;
}

module.exports = {
  createAwsRdsCredentialProvider,
  createAwsRdsPasswordProvider,
  databaseCredentials,
  parseSecret,
  readSecret,
};
