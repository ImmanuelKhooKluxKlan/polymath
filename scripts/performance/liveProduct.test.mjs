import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyLiveNavigationPolicy,
  resolveLivePage,
} from '../../src/config/liveProduct.js';
import { normalizePublicSiteConfiguration } from '../../src/config/siteConfiguration.js';

test('retired public sections redirect while Learn opens the teacher marketplace', () => {
  assert.equal(resolveLivePage('band'), 'studio');
  assert.equal(resolveLivePage('create-music'), 'studio');
  assert.equal(resolveLivePage('teacher-ar'), 'studio');
  assert.equal(resolveLivePage('teacher-projection'), 'studio');
  assert.equal(resolveLivePage('learn'), 'find-teacher');
  assert.equal(resolveLivePage('guitar'), 'guitar');
});

test('live navigation policy cannot be bypassed by an older saved configuration', () => {
  const navigation = applyLiveNavigationPolicy([
    { id: 'band', label: 'Band', visible: true },
    { id: 'create-music', label: 'Create Music', visible: true },
    { id: 'find-teacher', label: 'Find Teacher', visible: false },
  ]);
  assert.equal(navigation.find((item) => item.id === 'band').visible, false);
  assert.equal(navigation.find((item) => item.id === 'create-music').visible, false);
  assert.deepEqual(
    navigation.find((item) => item.id === 'find-teacher'),
    { id: 'find-teacher', label: 'Learn', visible: true },
  );
});

test('normalized public configuration exposes Learn and keeps side projects hidden', () => {
  const configuration = normalizePublicSiteConfiguration({
    navigation: [
      { id: 'band', label: 'Band', visible: true },
      { id: 'create-music', label: 'Create Music', visible: true },
      { id: 'find-teacher', label: 'Find Teacher', visible: true },
    ],
  });
  assert.equal(configuration.navigation.find((item) => item.id === 'band').visible, false);
  assert.equal(configuration.navigation.find((item) => item.id === 'create-music').visible, false);
  assert.equal(configuration.navigation.find((item) => item.id === 'find-teacher').label, 'Learn');
});
