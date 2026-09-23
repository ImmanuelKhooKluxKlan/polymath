import assert from 'node:assert/strict';
import test from 'node:test';

import {
  inferPortableSpeakerHint,
  normalizeSpeakerOutputMode,
  productionPerformanceGain,
  resolveSpeakerOutputProfile,
  speakerMixBus,
  speakerPerformanceGain,
  speakerPolyphonyHeadroom,
  speakerRegisterGain,
  speakerSampleGainCompensation,
  speakerVirtualBassProfile,
  speakerVoiceProfile,
  tonePresetForSpeaker,
} from '../../src/engine/speakerOutputProfile.js';

test('automatic output adapts phones and portable computers without changing desktops', () => {
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
  }), 'full-range');
  assert.equal(resolveSpeakerOutputProfile('small', {
    deviceClass: 'phone', performanceTier: 'lite',
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

test('melody emphasis never makes the left-hand accompaniment quieter', () => {
  const melody = productionPerformanceGain(1.12, {
    midi: 72, role: 'melody', source: 'autoplay',
  });
  const accompaniment = productionPerformanceGain(0.88, {
    midi: 43, role: 'accompaniment', source: 'autoplay',
  });
  assert.ok(Math.abs(melody - (10 ** (1.5 / 20))) < 1e-9);
  assert.equal(accompaniment, 1);
  assert.ok(melody > accompaniment);
  assert.ok(melody / accompaniment < 1.2);
  assert.equal(productionPerformanceGain(1.5, {
    midi: 72, role: 'melody', source: 'autoplay',
  }), 1.24);
  assert.equal(productionPerformanceGain(1.5, {
    midi: 84, role: 'melody', source: 'manual',
  }), 1);
});

test('unlabelled scores receive a smooth upper-register lift without weakening bass', () => {
  const bass = productionPerformanceGain(1, { midi: 43, source: 'autoplay' });
  const middle = productionPerformanceGain(1, { midi: 60, source: 'autoplay' });
  const upper = productionPerformanceGain(1, { midi: 84, source: 'autoplay' });
  assert.equal(bass, 1);
  assert.ok(middle > bass);
  assert.ok(upper > middle);
  assert.ok(upper < 1.1);
});

test('small-speaker register curve preserves body and progressively tames treble', () => {
  assert.equal(speakerRegisterGain(84, 'full-range'), 1);
  assert.ok(speakerRegisterGain(24, 'small-speaker') > speakerRegisterGain(48, 'small-speaker'));
  assert.ok(speakerRegisterGain(48, 'small-speaker') > 1);
  assert.equal(speakerRegisterGain(60, 'small-speaker'), 1);
  assert.equal(speakerRegisterGain(72, 'small-speaker'), 1);
  assert.ok(speakerRegisterGain(84, 'small-speaker') < speakerRegisterGain(72, 'small-speaker'));
  assert.ok(speakerRegisterGain(108, 'small-speaker') >= 0.9);
});

test('small-speaker mix closes extreme arrangement gain gaps without changing full-range gain', () => {
  assert.equal(speakerPerformanceGain(0.25, 48, 'full-range'), 0.25);
  const compactBass = speakerPerformanceGain(0.25, 48, 'small-speaker');
  const compactTreble = speakerPerformanceGain(1.5, 76, 'small-speaker');
  assert.ok(compactBass >= 0.9);
  assert.ok(compactTreble <= 1.2);
  assert.ok(compactTreble / compactBass < 1.6);
});

test('compact bass keeps the accepted sample balance and adds quiet audible partials', () => {
  assert.equal(speakerSampleGainCompensation(21, 'full-range'), 3.2);
  assert.equal(speakerSampleGainCompensation(21, 'small-speaker'), 3.2);
  assert.equal(speakerSampleGainCompensation(48, 'small-speaker'), 1);
  const lowBass = speakerVoiceProfile(21, 'small-speaker');
  assert.equal(lowBass.harmonicDrive, 1);
  assert.equal(lowBass.phaseSafeMono, false);
  assert.equal(lowBass.usePerKeyCalibration, false);
  assert.ok(lowBass.bodyFrequency >= 130 && lowBass.bodyFrequency <= 260);
  assert.equal(speakerVoiceProfile(60, 'small-speaker').harmonicDrive, 1);

  const virtualBass = speakerVirtualBassProfile(21, 'small-speaker');
  const fundamental = 440 * (2 ** ((21 - 69) / 12));
  assert.equal(virtualBass.enabled, true);
  assert.equal(virtualBass.harmonics.length, 3);
  assert.ok(virtualBass.harmonics.every((harmonic) => harmonic * fundamental >= 120));
  assert.ok(virtualBass.gain > 0 && virtualBass.gain < 0.06);
  assert.equal(speakerVirtualBassProfile(46, 'small-speaker').enabled, false);
  assert.equal(speakerVirtualBassProfile(21, 'full-range').enabled, false);
});

test('compact chord headroom scales smoothly while full-range output remains untouched', () => {
  assert.equal(speakerPolyphonyHeadroom(36, 'full-range'), 1);
  assert.equal(speakerPolyphonyHeadroom(8, 'small-speaker'), 1);
  assert.ok(speakerPolyphonyHeadroom(10, 'small-speaker') > 0.98);
  assert.ok(speakerPolyphonyHeadroom(24, 'small-speaker') < speakerPolyphonyHeadroom(10, 'small-speaker'));
  assert.ok(speakerPolyphonyHeadroom(64, 'small-speaker') >= 0.8);
});

test('all notes share one continuous piano bus on compact output', () => {
  assert.equal(speakerMixBus(84, 'accompaniment', 'small-speaker'), 'direct');
  assert.equal(speakerMixBus(48, 'melody', 'small-speaker'), 'direct');
  assert.equal(speakerMixBus(48, '', 'small-speaker', 'manual'), 'direct');
  assert.equal(speakerMixBus(72, '', 'small-speaker', 'manual'), 'direct');
  assert.equal(speakerMixBus(48, '', 'small-speaker'), 'direct');
  assert.equal(speakerMixBus(72, '', 'small-speaker'), 'direct');
  assert.equal(speakerMixBus(48, 'melody', 'full-range'), 'direct');
});

test('small-speaker voices recover bass harmonics and de-harsh the upper register', () => {
  const bass = speakerVoiceProfile(33, 'small-speaker');
  const treble = speakerVoiceProfile(84, 'small-speaker');
  assert.equal(bass.bodyType, 'peaking');
  assert.ok(bass.bodyFrequency >= 130 && bass.bodyFrequency <= 260);
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
  assert.equal(compact.highPassFrequency, preset.highPassFrequency);
  assert.ok(compact.lowShelfFrequency > preset.lowShelfFrequency);
  assert.ok(compact.lowShelfGain > preset.lowShelfGain);
  assert.ok(compact.presenceGain < preset.presenceGain);
  assert.ok(compact.airGain < preset.airGain);
  assert.ok(compact.wetGain < preset.wetGain);
  assert.ok(compact.panWidth < 0.15);
  assert.equal(compact.monoOutput, false);
  assert.equal(compact.melodyBusGain, compact.accompanimentBusGain);
  assert.ok(compact.masterLevel <= 0.84);
  assert.ok(compact.limiterThreshold <= -4.5);
  assert.equal(compact.compactPeakProtection, false);
});
