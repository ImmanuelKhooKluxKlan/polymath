const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { createArtifactStore } = require('./artifactStore');

test('local artifact store writes and materializes safe keys', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-artifacts-'));
  const store = createArtifactStore({ localRoot: root });
  try {
    const key = await store.putBuffer('scores/example.json', Buffer.from('{"ok":true}'), 'application/json');
    assert.equal(key, 'scores/example.json');
    assert.equal(store.provider, 'local-disk');
    assert.equal(await store.materialize(key, path.join(root, 'unused')), path.join(root, 'scores', 'example.json'));
    assert.equal(fs.readFileSync(path.join(root, 'scores', 'example.json'), 'utf8'), '{"ok":true}');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('local artifact store blocks traversal outside its root', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-artifacts-'));
  const store = createArtifactStore({ localRoot: root });
  try {
    assert.throws(() => store.localPath('../secret.txt'), /Invalid artifact path/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
