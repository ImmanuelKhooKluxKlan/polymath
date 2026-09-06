const fs = require('fs');
const { isDeepStrictEqual } = require('util');
const { Pool } = require('pg');
const { summarizeProductEvents, summaryFromCounts } = require('./productAnalytics');

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
  constructor({
    databaseUrl,
    databaseHost,
    databasePort,
    databaseUser,
    databasePassword,
    databaseName,
    filePath,
    stateKey = 'primary',
  }) {
    this.databaseUrl = String(databaseUrl || '').trim();
    this.databaseHost = String(databaseHost || '').trim();
    this.databaseConfig = this.databaseUrl
      ? { connectionString: this.databaseUrl }
      : this.databaseHost
        ? {
            host: this.databaseHost,
            port: Math.max(1, Number(databasePort || 5432)),
            user: String(databaseUser || '').trim(),
            password: String(databasePassword || ''),
            database: String(databaseName || 'polymath').trim(),
          }
        : null;
    this.filePath = filePath;
    this.stateKey = stateKey;
    this.pool = null;
    this.initialized = false;
    this.memoryProductEvents = [];
  }

  get provider() {
    return this.databaseConfig ? 'postgresql' : 'atomic-json';
  }

  async initialize(seedDocument) {
    if (this.initialized) return;
    if (!this.databaseConfig) {
      this.initialized = true;
      return;
    }

    const sslEnabled = String(process.env.DATABASE_SSL || 'true').toLowerCase() !== 'false';
    const rejectUnauthorized = String(process.env.DATABASE_SSL_REJECT_UNAUTHORIZED || 'true').toLowerCase() !== 'false';
    const sslCaPath = String(process.env.DATABASE_SSL_CA_PATH || '').trim();
    this.pool = new Pool({
      ...this.databaseConfig,
      max: Math.max(2, Math.min(30, Number(process.env.DATABASE_POOL_MAX || 10))),
      ssl: sslEnabled ? {
        rejectUnauthorized,
        ...(sslCaPath ? { ca: fs.readFileSync(sslCaPath, 'utf8') } : {}),
      } : false,
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
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS polymath_product_events (
        event_id text PRIMARY KEY,
        event_name text NOT NULL,
        occurred_at timestamptz NOT NULL,
        received_at timestamptz NOT NULL DEFAULT now(),
        user_id text,
        anonymous_id text,
        session_id text,
        path text NOT NULL DEFAULT '',
        release text NOT NULL DEFAULT '',
        properties jsonb NOT NULL DEFAULT '{}'::jsonb
      )
    `);
    await this.pool.query(`
      CREATE INDEX IF NOT EXISTS polymath_product_events_occurred_idx
      ON polymath_product_events (occurred_at DESC)
    `);
    await this.pool.query(`
      CREATE INDEX IF NOT EXISTS polymath_product_events_name_occurred_idx
      ON polymath_product_events (event_name, occurred_at DESC)
    `);
    await this.pool.query(`
      DELETE FROM polymath_product_events
      WHERE received_at < now() - interval '180 days'
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

  async recordProductEvents(events) {
    if (!this.initialized) throw new Error('State store must be initialized before recording product events.');
    if (!Array.isArray(events) || !events.length) return 0;
    const receivedAt = new Date().toISOString();
    if (!this.pool) {
      const known = new Set(this.memoryProductEvents.map((event) => event.eventId));
      let inserted = 0;
      for (const event of events) {
        if (known.has(event.eventId)) continue;
        this.memoryProductEvents.push({ ...event, receivedAt });
        known.add(event.eventId);
        inserted += 1;
      }
      if (this.memoryProductEvents.length > 10000) {
        this.memoryProductEvents.splice(0, this.memoryProductEvents.length - 10000);
      }
      return inserted;
    }

    const columnsPerEvent = 10;
    const values = [];
    const placeholders = events.map((event, rowIndex) => {
      const offset = rowIndex * columnsPerEvent;
      values.push(
        event.eventId,
        event.eventName,
        event.occurredAt,
        receivedAt,
        event.userId || null,
        event.anonymousId || null,
        event.sessionId || null,
        event.path || '',
        event.release || '',
        JSON.stringify(event.properties || {}),
      );
      return `(${Array.from({ length: columnsPerEvent }, (_, index) => `$${offset + index + 1}`).join(',')})`;
    });
    const result = await this.pool.query(
      `INSERT INTO polymath_product_events
       (event_id, event_name, occurred_at, received_at, user_id, anonymous_id, session_id, path, release, properties)
       VALUES ${placeholders.join(',')}
       ON CONFLICT (event_id) DO NOTHING`,
      values,
    );
    return result.rowCount;
  }

  async productEventSummary(days = 30) {
    if (!this.initialized) throw new Error('State store must be initialized before reading product events.');
    const windowDays = Math.max(1, Math.min(180, Math.floor(Number(days) || 30)));
    const cutoff = new Date(Date.now() - (windowDays * 24 * 60 * 60 * 1000)).toISOString();
    if (!this.pool) {
      return summarizeProductEvents(
        this.memoryProductEvents.filter((event) => String(event.occurredAt) >= cutoff),
        windowDays,
      );
    }

    const [countsResult, dailyResult, returnResult, feedbackResult] = await Promise.all([
      this.pool.query(`
        SELECT
          event_name,
          count(*)::integer AS events,
          count(DISTINCT coalesce(nullif(user_id, ''), nullif(anonymous_id, ''), session_id))::integer AS actors,
          avg(CASE WHEN jsonb_typeof(properties->'score') = 'number' THEN (properties->>'score')::numeric END) AS average_score,
          avg(CASE WHEN jsonb_typeof(properties->'durationSeconds') = 'number' THEN (properties->>'durationSeconds')::numeric END) AS average_duration_seconds
        FROM polymath_product_events
        WHERE occurred_at >= $1::timestamptz
        GROUP BY event_name
      `, [cutoff]),
      this.pool.query(`
        SELECT occurred_at::date::text AS day, count(*)::integer AS event_count
        FROM polymath_product_events
        WHERE occurred_at >= $1::timestamptz
        GROUP BY occurred_at::date
        ORDER BY occurred_at::date
      `, [cutoff]),
      this.pool.query(`
        WITH signed_days AS (
          SELECT user_id, count(DISTINCT occurred_at::date)::integer AS active_days
          FROM polymath_product_events
          WHERE occurred_at >= $1::timestamptz AND nullif(user_id, '') IS NOT NULL
          GROUP BY user_id
        )
        SELECT
          count(*)::integer AS signed_actors,
          count(*) FILTER (WHERE active_days >= 2)::integer AS returning_actors
        FROM signed_days
      `, [cutoff]),
      this.pool.query(`
        SELECT properties->>'feedback' AS feedback,
               count(DISTINCT coalesce(nullif(user_id, ''), nullif(anonymous_id, ''), session_id))::integer AS actors
        FROM polymath_product_events
        WHERE occurred_at >= $1::timestamptz
          AND event_name = 'transcription_feedback'
          AND properties->>'feedback' IN ('accurate', 'needs-work')
        GROUP BY properties->>'feedback'
      `, [cutoff]),
    ]);
    const counts = countsResult.rows.map((row) => ({
      eventName: row.event_name,
      events: row.events,
      actors: row.actors,
      averageScore: row.average_score,
      averageDurationSeconds: row.average_duration_seconds,
    }));
    const daily = dailyResult.rows.map((row) => ({ day: row.day, eventCount: Number(row.event_count || 0) }));
    return summaryFromCounts({
      counts,
      daily,
      feedback: feedbackResult.rows,
      days: windowDays,
      signedActors: Number(returnResult.rows[0]?.signed_actors || 0),
      returningActors: Number(returnResult.rows[0]?.returning_actors || 0),
    });
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
