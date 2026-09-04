'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { MUSIC_KNOWLEDGE_VERSION, retrieveMusicKnowledge } = require('./musicKnowledge');

test('retrieves exact music fundamentals without flooding unrelated chat', () => {
  const guitar = retrieveMusicKnowledge('What are the strings in standard guitar tuning?');
  assert.equal(guitar.version, MUSIC_KNOWLEDGE_VERSION);
  assert.equal(guitar.entries[0].id, 'guitar-standard-tuning');
  assert.match(guitar.entries[0].fact, /E2-A2-D3-G3-B3-E4/);

  const piano = retrieveMusicKnowledge('Why does the piano sustain pedal make this muddy?');
  assert.equal(piano.entries.some((entry) => entry.id === 'piano-pedals'), true);
  assert.equal(retrieveMusicKnowledge('What movie should I watch tonight?').entries.length, 0);
});

test('bounds retrieved reference anchors', () => {
  const result = retrieveMusicKnowledge('pitch interval scale chord rhythm piano pedal guitar violin acoustics practice', 3);
  assert.equal(result.entries.length, 3);
  assert.equal(new Set(result.entries.map((entry) => entry.id)).size, 3);
});
