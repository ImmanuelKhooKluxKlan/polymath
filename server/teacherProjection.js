'use strict';

const crypto = require('crypto');
const { Pool } = require('pg');

const DEFAULT_TTL_MS = 2 * 60 * 60 * 1000;

function tokenHash(token) {
  return crypto.createHash('sha256').update(String(token || '')).digest('hex');
}

function safeEqual(left, right) {
  const leftBuffer = Buffer.from(String(left || ''), 'utf8');
  const rightBuffer = Buffer.from(String(right || ''), 'utf8');
  return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

function normalizeState(input = {}, previous = {}) {
  const mode = input.mode === 'hologram' ? 'hologram' : 'wall';
  const style = input.style === 'bikini' ? 'bikini' : 'regular';
  return {
    mode,
    style,
    speaking: Boolean(input.speaking),
    visible: input.visible !== false,
    caption: String(input.caption || '').trim().slice(0, 280),
    version: Number(previous.version || 0) + 1,
  };
}

function createTeacherProjectionStore(options = {}) {
  const databaseUrl = String(options.databaseUrl || '').trim();
  const databaseHost = String(options.databaseHost || '').trim();
  const databaseConfig = databaseUrl
    ? { connectionString: databaseUrl }
    : databaseHost
      ? {
          host: databaseHost,
          port: Math.max(1, Number(options.databasePort || 5432)),
          user: String(options.databaseUser || '').trim(),
          password: String(options.databasePassword || ''),
          database: String(options.databaseName || 'polymath').trim(),
        }
      : null;
  if (options.pool || databaseConfig) {
    return createPostgresProjectionStore({ ...options, databaseConfig });
  }

  const sessions = new Map();
  const now = options.now || Date.now;
  const ttlMs = Math.max(60_000, Number(options.ttlMs) || DEFAULT_TTL_MS);

  function cleanup() {
    const timestamp = now();
    for (const [id, session] of sessions) {
      if (session.expiresAt <= timestamp) sessions.delete(id);
    }
  }

  async function create(ownerId, initialState = {}) {
    cleanup();
    const id = crypto.randomUUID();
    const token = crypto.randomBytes(32).toString('base64url');
    const createdAt = now();
    const session = {
      id,
      ownerId: String(ownerId),
      tokenHash: tokenHash(token),
      createdAt,
      expiresAt: createdAt + ttlMs,
      state: normalizeState(initialState),
    };
    sessions.set(id, session);
    return {
      id,
      token,
      createdAt: new Date(createdAt).toISOString(),
      expiresAt: new Date(session.expiresAt).toISOString(),
      state: { ...session.state },
    };
  }

  async function update(ownerId, id, input) {
    cleanup();
    const session = sessions.get(String(id));
    if (!session || session.ownerId !== String(ownerId)) return null;
    session.state = normalizeState(input, session.state);
    return { ...session.state };
  }

  async function read(id, token) {
    cleanup();
    const session = sessions.get(String(id));
    if (!session || !safeEqual(session.tokenHash, tokenHash(token))) return null;
    return {
      state: { ...session.state },
      expiresAt: new Date(session.expiresAt).toISOString(),
    };
  }

  async function close(ownerId, id) {
    cleanup();
    const session = sessions.get(String(id));
    if (!session || session.ownerId !== String(ownerId)) return false;
    sessions.delete(String(id));
    return true;
  }

  return Object.freeze({ provider: 'memory', create, update, read, close, cleanup });
}

function createPostgresProjectionStore(options = {}) {
  const pool = options.pool || new Pool({ ...options.databaseConfig, max: 4 });
  const ttlMs = Math.max(60_000, Number(options.ttlMs) || DEFAULT_TTL_MS);
  let initializePromise = null;

  function initialize() {
    if (!initializePromise) {
      initializePromise = pool.query(`
        CREATE TABLE IF NOT EXISTS teacher_projection_sessions (
          id UUID PRIMARY KEY,
          owner_id TEXT NOT NULL,
          token_hash TEXT NOT NULL,
          state JSONB NOT NULL,
          version INTEGER NOT NULL DEFAULT 1,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          expires_at TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS teacher_projection_sessions_expires_idx
          ON teacher_projection_sessions (expires_at);
      `);
    }
    return initializePromise;
  }

  async function cleanup() {
    await initialize();
    await pool.query('DELETE FROM teacher_projection_sessions WHERE expires_at <= NOW()');
  }

  async function create(ownerId, initialState = {}) {
    await initialize();
    await pool.query('DELETE FROM teacher_projection_sessions WHERE expires_at <= NOW()');
    const id = crypto.randomUUID();
    const token = crypto.randomBytes(32).toString('base64url');
    const state = normalizeState(initialState);
    const expiresAt = new Date(Date.now() + ttlMs);
    const storedState = { ...state };
    delete storedState.version;
    const result = await pool.query(`
      INSERT INTO teacher_projection_sessions
        (id, owner_id, token_hash, state, version, expires_at)
      VALUES ($1, $2, $3, $4::jsonb, 1, $5)
      RETURNING created_at
    `, [id, String(ownerId), tokenHash(token), JSON.stringify(storedState), expiresAt]);
    return {
      id,
      token,
      createdAt: new Date(result.rows[0].created_at).toISOString(),
      expiresAt: expiresAt.toISOString(),
      state,
    };
  }

  async function update(ownerId, id, input = {}) {
    await initialize();
    const normalized = normalizeState(input);
    delete normalized.version;
    const result = await pool.query(`
      UPDATE teacher_projection_sessions
      SET state = $3::jsonb, version = version + 1
      WHERE id = $1 AND owner_id = $2 AND expires_at > NOW()
      RETURNING state, version
    `, [String(id), String(ownerId), JSON.stringify(normalized)]);
    if (!result.rowCount) return null;
    return { ...result.rows[0].state, version: Number(result.rows[0].version) };
  }

  async function read(id, token) {
    await initialize();
    const result = await pool.query(`
      SELECT state, version, expires_at
      FROM teacher_projection_sessions
      WHERE id = $1 AND token_hash = $2 AND expires_at > NOW()
    `, [String(id), tokenHash(token)]);
    if (!result.rowCount) return null;
    return {
      state: { ...result.rows[0].state, version: Number(result.rows[0].version) },
      expiresAt: new Date(result.rows[0].expires_at).toISOString(),
    };
  }

  async function close(ownerId, id) {
    await initialize();
    const result = await pool.query(
      'DELETE FROM teacher_projection_sessions WHERE id = $1 AND owner_id = $2',
      [String(id), String(ownerId)],
    );
    return result.rowCount > 0;
  }

  return Object.freeze({ provider: 'postgresql', create, update, read, close, cleanup });
}

module.exports = {
  createTeacherProjectionStore,
  normalizeState,
  tokenHash,
};
