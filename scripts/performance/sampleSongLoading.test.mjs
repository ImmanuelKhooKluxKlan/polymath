import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(
  new URL('../../src/data/sampleSongs.js', import.meta.url),
  'utf8',
);

test('the MIDI decoder stays outside the initial application bundle', () => {
  assert.doesNotMatch(source, /^import\s+.*['"]@tonejs\/midi['"];?$/m);
  assert.match(source, /await import\(['"]@tonejs\/midi['"]\)/);
});
