const fs = require('fs');
const path = require('path');
const { isDeepStrictEqual } = require('util');
const { GetObjectCommand, S3Client } = require('@aws-sdk/client-s3');
const { Pool } = require('pg');

require('dotenv').config({ path: path.join(__dirname, '.env') });

const VERIFY_ONLY = process.argv.includes('--verify-only');
const REPLACE = process.argv.includes('--replace');
const DATA_DIR = path.resolve(process.env.POLYMATH_DATA_DIR || path.join(__dirname, 'data'));
const SOURCE_PATH = path.resolve(process.env.POLYMATH_STATE_SOURCE || path.join(DATA_DIR, 'database.json'));
const SOURCE_OBJECT_KEY = String(process.env.POLYMATH_STATE_SOURCE_OBJECT_KEY || '').trim();
const STATE_KEY = String(process.env.DATABASE_STATE_KEY || 'primary').trim();
const DATABASE_URL = String(process.env.DATABASE_URL || '').trim();
const DATABASE_HOST = String(process.env.PGHOST || '').trim();

function collectionCounts(document) {
  return Object.fromEntries(
    Object.entries(document)
      .filter(([, value]) => Array.isArray(value))
      .map(([key, value]) => [key, value.length]),
  );
}

async function loadSource() {
  if (!SOURCE_OBJECT_KEY) {
    if (!fs.existsSync(SOURCE_PATH)) throw new Error('The source database.json file does not exist.');
    return JSON.parse(fs.readFileSync(SOURCE_PATH, 'utf8'));
  }

  const bucket = String(process.env.ARTIFACT_S3_BUCKET || '').trim();
  if (!bucket) throw new Error('ARTIFACT_S3_BUCKET is required for an object source.');
  const endpoint = String(process.env.ARTIFACT_S3_ENDPOINT || '').trim() || undefined;
  const client = new S3Client({
    region: String(process.env.ARTIFACT_S3_REGION || 'auto').trim(),
    endpoint,
    forcePathStyle: String(process.env.ARTIFACT_S3_FORCE_PATH_STYLE || 'false').toLowerCase() === 'true',
    ...(process.env.ARTIFACT_S3_ACCESS_KEY_ID && process.env.ARTIFACT_S3_SECRET_ACCESS_KEY ? {
      credentials: {
        accessKeyId: process.env.ARTIFACT_S3_ACCESS_KEY_ID,
        secretAccessKey: process.env.ARTIFACT_S3_SECRET_ACCESS_KEY,
      },
    } : {}),
  });
  const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: SOURCE_OBJECT_KEY }));
  const bytes = await response.Body.transformToByteArray();
  return JSON.parse(Buffer.from(bytes).toString('utf8'));
}

async function main() {
  if (!DATABASE_URL && !DATABASE_HOST) {
    throw new Error('DATABASE_URL or PGHOST is required.');
  }
  if (VERIFY_ONLY && REPLACE) throw new Error('--verify-only and --replace cannot be combined.');

  const source = await loadSource();
  const sslEnabled = String(process.env.DATABASE_SSL || 'true').toLowerCase() !== 'false';
  const rejectUnauthorized = String(process.env.DATABASE_SSL_REJECT_UNAUTHORIZED || 'true').toLowerCase() !== 'false';
  const sslCaPath = String(process.env.DATABASE_SSL_CA_PATH || '').trim();
  const pool = new Pool({
    ...(DATABASE_URL ? { connectionString: DATABASE_URL } : {
      host: DATABASE_HOST,
      port: Math.max(1, Number(process.env.PGPORT || 5432)),
      user: String(process.env.PGUSER || '').trim(),
      password: String(process.env.PGPASSWORD || ''),
      database: String(process.env.PGDATABASE || 'polymath').trim(),
    }),
    max: 2,
    ssl: sslEnabled ? {
      rejectUnauthorized,
      ...(sslCaPath ? { ca: fs.readFileSync(sslCaPath, 'utf8') } : {}),
    } : false,
    application_name: 'polymath-state-migration',
  });

  try {
    if (!VERIFY_ONLY) {
      const client = await pool.connect();
      try {
        await client.query('BEGIN');
        await client.query(`
          CREATE TABLE IF NOT EXISTS polymath_state (
            state_key text PRIMARY KEY,
            revision bigint NOT NULL DEFAULT 0,
            document jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
          )
        `);
        const current = await client.query(
          'SELECT revision, document FROM polymath_state WHERE state_key = $1 FOR UPDATE',
          [STATE_KEY],
        );

        if (current.rowCount === 0) {
          await client.query(
            'INSERT INTO polymath_state (state_key, document) VALUES ($1, $2::jsonb)',
            [STATE_KEY, JSON.stringify(source)],
          );
          console.log('PostgreSQL state migration: inserted missing primary state.');
        } else if (REPLACE) {
          await client.query(
            `UPDATE polymath_state
             SET revision = revision + 1, document = $2::jsonb, updated_at = now()
             WHERE state_key = $1`,
            [STATE_KEY, JSON.stringify(source)],
          );
          console.log('PostgreSQL state migration: replaced existing state by explicit request.');
        } else if (!isDeepStrictEqual(current.rows[0].document, source)) {
          throw new Error('PostgreSQL already contains different state; refusing to overwrite without --replace.');
        } else {
          console.log('PostgreSQL state migration: existing state is already identical.');
        }
        await client.query('COMMIT');
      } catch (error) {
        await client.query('ROLLBACK').catch(() => {});
        throw error;
      } finally {
        client.release();
      }
    }

    const verified = await pool.query(
      'SELECT revision, document FROM polymath_state WHERE state_key = $1',
      [STATE_KEY],
    );
    if (verified.rowCount !== 1) throw new Error('PostgreSQL state row is missing.');
    if (!isDeepStrictEqual(verified.rows[0].document, source)) {
      throw new Error('PostgreSQL state verification failed: document mismatch.');
    }

    const counts = collectionCounts(source);
    console.log(
      `PostgreSQL state verification passed: revision ${verified.rows[0].revision}, `
      + `${Object.keys(counts).length} collections, ${counts.users || 0} users, `
      + `${counts.listings || 0} listings.`,
    );
  } finally {
    await pool.end();
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message || error);
    process.exitCode = 1;
  });
}

module.exports = { collectionCounts, loadSource, main };
