import assert from 'node:assert/strict';
import test from 'node:test';

import {
  inferPortableSpeakerHint,
  normalizeSpeakerOutputMode,
  resolveSpeakerOutputProfile,
  speakerMixBus,
  speakerPerformanceGain,
  speakerPolyphonyHeadroom,
  speakerRegisterGain,
  speakerSampleGainCompensation,
  speakerVoiceProfile,
  tonePresetForSpeaker,
} from '../../src/engine/speakerOutputProfile.js';

test('automatic output keeps full-range desktop sound and protects small devices', () => {
  assert.equal(resolveSpeakerOutputProfile('auto', {
    deviceClass: 'desktop', performanceTier: 'full',
  }), 'full-range');
  assert.equal(resolveSpeakerOutputProfile('auto', {
    deviceClass: 'desktop', performanceTier: 'full', portableSpeakerHint: true,
  }), 'small-speaker');
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

test('portable speaker hint catches phones, batteries, and compact laptop displays', () => {
  assert.equal(inferPortableSpeakerHint({ deviceClass: 'phone' }), true);
  assert.equal(inferPortableSpeakerHint({ deviceClass: 'desktop', hasBattery: true }), true);
  assert.equal(inferPortableSpeakerHint({
    deviceClass: 'desktop', screenWidth: 1536, screenHeight: 864,
  }), true);
  assert.equal(inferPortableSpeakerHint({
    deviceClass: 'desktop', screenWidth: 2560, screenHeight: 1440,
  }), false);
});

test('small-speaker register curve preserves body and progressively tames treble', () => {
  assert.equal(speakerRegisterGain(84, 'full-range'), 1);
  assert.ok(speakerRegisterGain(24, 'small-speaker') > speakerRegisterGain(48, 'small-speaker'));
  assert.ok(speakerRegisterGain(48, 'small-speaker') > 1);
  assert.ok(speakerRegisterGain(60, 'small-speaker') < 1);
  assert.ok(speakerRegisterGain(72, 'small-speaker') < speakerRegisterGain(60, 'small-speaker'));
  assert.ok(speakerRegisterGain(84, 'small-speaker') < speakerRegisterGain(72, 'small-speaker'));
});

test('small-speaker mix closes extreme arrangement gain gaps without changing full-range gain', () => {
  assert.equal(speakerPerformanceGain(0.25, 48, 'full-range'), 0.25);
  const compactBass = speakerPerformanceGain(0.25, 48, 'small-speaker');
  const compactTreble = speakerPerformanceGain(1.5, 76, 'small-speaker');
  assert.ok(compactBass >= 0.74);
  assert.ok(compactTreble <= 0.98);
  assert.ok(compactTreble / compactBass < 1.6);
});

test('compact bass uses harmonics instead of wasteful raw sub-bass gain', () => {
  assert.equal(speakerSampleGainCompensation(21, 'full-range'), 3.2);
  assert.ok(speakerSampleGainCompensation(21, 'small-speaker') <= 1.72);
  assert.ok(speakerSampleGainCompensation(21, 'small-speaker') > 1);
  assert.equal(speakerSampleGainCompensation(48, 'small-speaker'), 1);
  const lowBass = speakerVoiceProfile(21, 'small-speaker');
  assert.ok(lowBass.harmonicDrive > 1.8);
  assert.ok(lowBass.bodyFrequency >= 110 && lowBass.bodyFrequency <= 240);
  assert.equal(speakerVoiceProfile(60, 'small-speaker').harmonicDrive, 1);
});

test('compact chord headroom scales smoothly while full-range output remains untouched', () => {
  assert.equal(speakerPolyphonyHeadroom(36, 'full-range'), 1);
  assert.equal(speakerPolyphonyHeadroom(3, 'small-speaker'), 1);
  assert.ok(speakerPolyphonyHeadroom(10, 'small-speaker') < 0.85);
  assert.ok(speakerPolyphonyHeadroom(24, 'small-speaker') < speakerPolyphonyHeadroom(10, 'small-speaker'));
  assert.ok(speakerPolyphonyHeadroom(64, 'small-speaker') >= 0.5);
});

test('compact output separates musical roles before dynamics processing', () => {
  assert.equal(speakerMixBus(84, 'accompaniment', 'small-speaker'), 'accompaniment');
  assert.equal(speakerMixBus(48, 'melody', 'small-speaker'), 'melody');
  assert.equal(speakerMixBus(48, '', 'small-speaker', 'manual'), 'direct');
  assert.equal(speakerMixBus(72, '', 'small-speaker', 'manual'), 'direct');
  assert.equal(speakerMixBus(48, '', 'small-speaker'), 'accompaniment');
  assert.equal(speakerMixBus(72, '', 'small-speaker'), 'melody');
  assert.equal(speakerMixBus(48, 'melody', 'full-range'), 'direct');
});

test('small-speaker voices recover bass harmonics and de-harsh the upper register', () => {
  const bass = speakerVoiceProfile(33, 'small-speaker');
  const treble = speakerVoiceProfile(84, 'small-speaker');
  assert.equal(bass.bodyType, 'peaking');
  assert.ok(bass.bodyFrequency >= 110 && bass.bodyFrequency <= 240);
  assert.ok(bass.bodyGainOffset > 2);
  assert.ok(treble.hammerGainOffset < bass.hammerGainOffset);
  assert.ok(treble.airGainOffset < bass.airGainOffset);
  assert.equal(speakerVoiceProfile(33, 'full-range').compact, false);
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
  assert.ok(compact.panWidth < 0.1);
  assert.equal(compact.monoOutput, true);
  assert.ok(compact.melodyBusGain < compact.accompanimentBusGain);
  assert.ok(compact.melodyBusRatio > compact.accompanimentBusRatio);
  assert.ok(compact.masterLevel <= 0.84);
  assert.ok(compact.limiterThreshold <= -8);
  assert.equal(compact.compactPeakProtection, true);
});
