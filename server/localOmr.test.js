const assert = require('node:assert/strict');
const test = require('node:test');

const { localOmrAvailability, parseWorkerError } = require('./localOmr');

test('local OMR is enabled by default and configurable without an API key', () => {
  const capability = localOmrAvailability({ OMR_RENDER_DPI: '360', OMR_MAX_PAGES: '24' });
  assert.equal(capability.enabled, true);
  assert.equal(capability.provider, 'Polymath Local OMR');
  assert.equal(capability.renderDpi, 360);
  assert.equal(capability.maxPages, 24);
});

test('local OMR can be disabled explicitly', () => {
  assert.equal(localOmrAvailability({ OMR_ENABLED: 'false' }).enabled, false);
});

test('worker errors are converted to a concise user-safe message', () => {
  assert.equal(
    parseWorkerError({ stderr: 'diagnostic\n{"ok":false,"error":"No five-line staffs were detected."}' }),
    'No five-line staffs were detected.',
  );
});
