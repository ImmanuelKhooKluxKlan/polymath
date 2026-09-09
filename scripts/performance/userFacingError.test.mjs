import test from 'node:test';
import assert from 'node:assert/strict';
import { userFacingError } from '../../src/utils/userFacingError.js';

test('customer errors hide infrastructure details', () => {
  assert.equal(
    userFacingError(new Error('RunPod worker returned an invalid JSON blueprint.'), 'Please try again.'),
    'Please try again.',
  );
  assert.equal(
    userFacingError(Object.assign(new Error('Upstream failure'), { status: 503 }), 'Temporarily unavailable.'),
    'Temporarily unavailable.',
  );
});

test('customer errors preserve useful action a person can take', () => {
  assert.equal(
    userFacingError(new Error('Not enough Mcoins.'), 'Please try again.'),
    'Not enough Mcoins.',
  );
  assert.equal(
    userFacingError(new Error('Please sign in first.'), 'Please try again.'),
    'Please sign in first.',
  );
});
