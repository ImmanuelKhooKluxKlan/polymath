import assert from 'node:assert/strict';
import test from 'node:test';

import {
  normalizeSpeakerOutputMode,
  resolveSpeakerOutputProfile,
  speakerRegisterGain,
  tonePresetForSpeaker,
} from '../../src/engine/speakerOutputProfile.js';

test('automatic output keeps full-range desktop sound and protects small devices', () => {
  assert.equal(resolveSpeakerOutputProfile('auto', {
    deviceClass: 'desktop', performanceTier: 'full',
  }), 'full-range');
  assert.equal(resolveSpeakerOutputProfile('auto', {
    deviceClass: 'phone', performanceTier: 'balanced',
  }), 'small-speaker');
  assert.equal(resolveSpeakerOutputProfile('auto', {
    deviceClass: 'desktop', performanceTier: 'lite',
  }), 'small-speaker');
  assert.equal(resolveSpeakerOutputProfile('full', {
    deviceClass: 'phone', performanceTier: 'lite',
  }), 'full-range');
  assert.equal(normalizeSpeakerOutputMode('unknown'), 'auto');
});

test('small-speaker register curve preserves body and progressively tames treble', () => {
  assert.equal(speakerRegisterGain(84, 'full-range'), 1);
  assert.ok(speakerRegisterGain(48, 'small-speaker') > 1);
  assert.ok(speakerRegisterGain(60, 'small-speaker') < 1);
  assert.ok(speakerRegisterGain(72, 'small-speaker') < speakerRegisterGain(60, 'small-speaker'));
  assert.ok(speakerRegisterGain(84, 'small-speaker') < speakerRegisterGain(72, 'small-speaker'));
});

test('small-speaker EQ moves energy from screech and sub-bass into audible body', () => {
  const preset = {
    highPassFrequency: 25,
    lowShelfFrequency: 135,
    lowShelfGain: 0.85,
    presenceGain: 0.68,
    airGain: 0.28,
    dryGain: 0.8,
    wetGain: 0.2,
    resonanceGain: 0.064,
  };
  assert.equal(tonePresetForSpeaker(preset, 'full-range'), preset);
  const compact = tonePresetForSpeaker(preset, 'small-speaker');
  assert.ok(compact.highPassFrequency > preset.highPassFrequency);
  assert.ok(compact.lowShelfFrequency > preset.lowShelfFrequency);
  assert.ok(compact.lowShelfGain > preset.lowShelfGain);
  assert.ok(compact.presenceGain < preset.presenceGain);
  assert.ok(compact.airGain < preset.airGain);
  assert.ok(compact.wetGain < preset.wetGain);
});
