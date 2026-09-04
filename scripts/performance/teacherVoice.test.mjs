import assert from 'node:assert/strict';
import test from 'node:test';
import {
  selectTeacherVoice,
  speechReadyText,
  speechSegments,
  teacherVoiceProfile,
  teacherVoiceScore,
} from '../../src/engine/teacherVoice.js';

const voices = [
  { name: 'Zarvox', voiceURI: 'novelty', lang: 'en-US', localService: true },
  { name: 'Microsoft Aria Online (Natural)', voiceURI: 'aria-natural', lang: 'en-US' },
  { name: 'Microsoft Guy Online (Natural)', voiceURI: 'guy-natural', lang: 'en-US' },
  { name: 'Samantha', voiceURI: 'samantha', lang: 'en-US', localService: true },
  { name: 'Amelie', voiceURI: 'amelie', lang: 'fr-FR', localService: true },
];

test('selects a natural voice matching the teacher and language', () => {
  assert.equal(selectTeacherVoice(voices, { id: 'nova' }, 'en-US').voiceURI, 'aria-natural');
  assert.equal(selectTeacherVoice(voices, { id: 'mace' }, 'en-US').voiceURI, 'guy-natural');
  assert.equal(teacherVoiceScore(voices[0], { id: 'nova' }, 'en-US') < 0, true);
});

test('honours an explicit device voice and keeps teacher prosody', () => {
  assert.equal(selectTeacherVoice(voices, { id: 'nova' }, 'en-US', 'samantha').voiceURI, 'samantha');
  assert.deepEqual(teacherVoiceProfile({ id: 'anakin' }), {
    voiceType: 'masculine', rate: 0.96, pitch: 0.92,
  });
});

test('converts notation and markdown to speech-friendly text', () => {
  assert.equal(speechReadyText('Play **C#4**, then [G4](https://example.test).'), 'Play C sharp 4, then G4.');
  const segments = speechSegments(
    'First sentence. This second sentence contains enough carefully selected words to exceed the short spoken segment limit without cutting any word in half.',
    80,
  );
  assert.equal(segments.length > 1, true);
  assert.equal(segments.every((segment) => segment.length <= 80), true);
});
