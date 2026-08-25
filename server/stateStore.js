const fs = require('fs');
const { isDeepStrictEqual } = require('util');
const { Pool } = require('pg');

const STATE_REVISION = Symbol('polymathStateRevision');
const STATE_BASELINE = Symbol('polymathStateBaseline');

class StateConflictError extends Error {
  constructor(path) {
    super(`Concurrent database change conflicts at ${path}. Retry the request.`);
    this.name = 'StateConflictError';
    this.status = 409;
  }
}

function clone(value) {
  return value === undefined ? undefined : structuredClone(value);
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function mergeValue(base, current, proposed, location) {
  if (isDeepStrictEqual(proposed, base)) return clone(current);
  if (isDeepStrictEqual(current, base)) return clone(proposed);
  if (isDeepStrictEqual(current, proposed)) return clone(current);

  if (isPlainObject(base) && isPlainObject(current) && isPlainObject(proposed)) {
    const merged = {};
    const keys = new Set([...Object.keys(base), ...Object.keys(current), ...Object.keys(proposed)]);
    for (const key of keys) {
      const value = mergeValue(base[key], current[key], proposed[key], `${location}.${key}`);
      if (value !== undefined) merged[key] = value;
    }
    return merged;
  }

  throw new StateConflictError(location);
}

function recordKey(collection, record) {
  if (!record || typeof record !== 'object') return '';
  if (collection === 'sessions') return String(record.tokenHash || record.token || '');
  return String(record.id || '');
}

function mergeCollection(collection, baseRows, currentRows, proposedRows) {
  const allRows = [...baseRows, ...currentRows, ...proposedRows];
  if (allRows.some((row) => !recordKey(collection, row))) {
    return mergeValue(baseRows, currentRows, proposedRows, collection);
  }

  const index = (rows) => new Map(rows.map((row) => [recordKey(collection, row), row]));
  const base = index(baseRows);
  const current = index(currentRows);
  const proposed = index(proposedRows);
  const merged = [];
  const emitted = new Set();

  for (const row of currentRows) {
    const key = recordKey(collection, row);
    const value = mergeValue(base.get(key), current.get(key), proposed.get(key), `${collection}[${key}]`);
    emitted.add(key);
    if (value !== undefined) merged.push(value);
  }

  for (const row of proposedRows) {
    const key = recordKey(collection, row);
    if (emitted.has(key)) continue;
    const value = mergeValue(base.get(key), current.get(key), proposed.get(key), `${collection}[${key}]`);
    if (value !== undefined) merged.push(value);
  }

  return merged;
}

function mergeDocuments(base, current, proposed) {
  const merged = {};
  const keys = new Set([...Object.keys(base), ...Object.keys(current), ...Object.keys(proposed)]);
  for (const key of keys) {
    const values = [base[key], current[key], proposed[key]];
    merged[key] = values.every(Array.isArray)
      ? mergeCollection(key, ...values)
      : mergeValue(...values, key);
  }
  return merged;
}

function attachMetadata(document, revision, baseline = document) {
  Object.defineProperty(document, STATE_REVISION, { value: Number(revision), writable: true });
  Object.defineProperty(document, STATE_BASELINE, { value: clone(baseline), writable: true });
  return document;
}

class StateStore {
  constructor({ databaseUrl, filePath, stateKey = 'primary' }) {
    this.databaseUrl = String(databaseUrl || '').trim();
    this.filePath = filePath;
    this.stateKey = stateKey;
    this.pool = null;
    this.initialized = false;
  }

  get provider() {
    return this.databaseUrl ? 'postgresql' : 'atomic-json';
  }

  async initialize(seedDocument) {
    if (this.initialized) return;
    if (!this.databaseUrl) {
      this.initialized = true;
      return;
    }

    const sslEnabled = String(process.env.DATABASE_SSL || 'true').toLowerCase() !== 'false';
    const rejectUnauthorized = String(process.env.DATABASE_SSL_REJECT_UNAUTHORIZED || 'true').toLowerCase() !== 'false';
    this.pool = new Pool({
      connectionString: this.databaseUrl,
      max: Math.max(2, Math.min(30, Number(process.env.DATABASE_POOL_MAX || 10))),
      ssl: sslEnabled ? { rejectUnauthorized } : false,
      application_name: process.env.APP_REGION ? `polymath-${process.env.APP_REGION}` : 'polymath',
    });

    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS polymath_state (
        state_key text PRIMARY KEY,
        revision bigint NOT NULL DEFAULT 0,
        document jsonb NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now()
      )
    `);
    await this.pool.query(
      `INSERT INTO polymath_state (state_key, document)
       VALUES ($1, $2::jsonb)
       ON CONFLICT (state_key) DO NOTHING`,
      [this.stateKey, JSON.stringify(seedDocument)],
    );
    this.initialized = true;
  }

  async read(seedDocument) {
    await this.initialize(seedDocument);
    if (!this.pool) {
      const document = JSON.parse(fs.readFileSync(this.filePath, 'utf8'));
      return attachMetadata(document, 0);
    }

    const result = await this.pool.query(
      'SELECT revision, document FROM polymath_state WHERE state_key = $1',
      [this.stateKey],
    );
    if (result.rowCount !== 1) throw new Error('PostgreSQL state row is missing.');
    return attachMetadata(result.rows[0].document, result.rows[0].revision);
  }

  async write(document) {
    if (!this.initialized) throw new Error('State store must be initialized before writing.');
    if (!this.pool) {
      const temp = `${this.filePath}.tmp`;
      fs.writeFileSync(temp, JSON.stringify(document, null, 2));
      fs.renameSync(temp, this.filePath);
      document[STATE_BASELINE] = clone(document);
      return;
    }

    const proposed = clone(document);
    const expectedRevision = Number(document[STATE_REVISION] || 0);
    const baseline = clone(document[STATE_BASELINE] || proposed);
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      const result = await client.query(
        'SELECT revision, document FROM polymath_state WHERE state_key = $1 FOR UPDATE',
        [this.stateKey],
      );
      if (result.rowCount !== 1) throw new Error('PostgreSQL state row is missing.');

      const currentRevision = Number(result.rows[0].revision);
      const current = result.rows[0].document;
      const committed = currentRevision === expectedRevision
        ? proposed
        : mergeDocuments(baseline, current, proposed);
      const nextRevision = currentRevision + 1;

      await client.query(
        `UPDATE polymath_state
         SET revision = $2, document = $3::jsonb, updated_at = now()
         WHERE state_key = $1`,
        [this.stateKey, nextRevision, JSON.stringify(committed)],
      );
      await client.query('COMMIT');
      document[STATE_REVISION] = nextRevision;
      document[STATE_BASELINE] = clone(committed);
    } catch (error) {
      await client.query('ROLLBACK').catch(() => {});
      throw error;
    } finally {
      client.release();
    }
  }

  async close() {
    if (this.pool) await this.pool.end();
  }
}

function createStateStore(options) {
  return new StateStore(options);
}

module.exports = {
  StateConflictError,
  createStateStore,
  mergeDocuments,
};
