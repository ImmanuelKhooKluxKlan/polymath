import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(
  new URL('../../src/components/PianoKeyboard.jsx', import.meta.url),
  'utf8',
);

test('white and black piano key templates remain accessible buttons', () => {
  const keyTemplates = [...source.matchAll(/<button[\s\S]*?<\/button>/g)]
    .map((match) => match[0])
    .filter((markup) => markup.includes('piano-key'));

  assert.equal(keyTemplates.length, 2);
  keyTemplates.forEach((markup) => {
    assert.match(markup, /type="button"/);
    assert.match(markup, /aria-label={`Piano key \${noteToDisplayName\(key\.midi, true\)}`}/);
  });
  assert.doesNotMatch(source, /className="black-layer"\s+aria-hidden="true"/);
});
