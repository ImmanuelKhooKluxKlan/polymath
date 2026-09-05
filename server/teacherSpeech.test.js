'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  createTeacherSpeechService,
  speechRegion,
  speechText,
  teacherGreeting,
  teacherSpeechProfile,
} = require('./teacherSpeech');

test('assigns every built-in teacher a distinct original generative voice', () => {
  const ids = ['aria', 'nova', 'anakin', 'taylor', 'mace'];
  const profiles = ids.map((id) => teacherSpeechProfile({ id }));
  assert.equal(new Set(profiles.map((profile) => profile.voiceId)).size, ids.length);
  assert.equal(profiles.find((profile) => profile.id === 'nova').voiceId, 'Salli');
  assert.equal(profiles.every((profile) => profile.engine === 'generative'), true);
});

test('gives opted-in adult Padme sessions a warm greeting without impersonating a real person', () => {
  assert.equal(
    teacherGreeting({ teacher: { id: 'nova' }, studentName: 'Maya Student', conversationMode: 'adult-companion' }),
    "Oh, hi, sweetheart. I'm Padme. Come sit with me. What kind of mood are you in today?",
  );
  assert.match(
    teacherGreeting({ teacher: { id: 'mace' }, studentName: 'Maya Student' }),
    /level your wrists/i,
  );
});

test('normalizes lesson text and selects a supported nearby speech region', () => {
  assert.equal(speechText('Play **C#4**, then `Gb3`.'), 'Play C sharp 4, then G flat 3.');
  assert.equal(speechRegion({ APP_REGION: 'ap-southeast-1' }), 'ap-southeast-1');
  assert.equal(speechRegion({ APP_REGION: 'us-east-2' }), 'us-east-1');
});

test('synthesizes and caches generative speech for repeated playback', async () => {
  const requests = [];
  const client = {
    async send(command) {
      requests.push(command.input);
      return {
        AudioStream: Uint8Array.from([73, 68, 51, 4]),
        ContentType: 'audio/mpeg',
        RequestCharacters: command.input.Text.length,
      };
    },
  };
  const service = createTeacherSpeechService({ NODE_ENV: 'test' }, { client, region: 'ap-southeast-1' });
  const first = await service.synthesize({ teacher: { id: 'nova' }, text: 'Oh, hello there.' });
  const replay = await service.synthesize({ teacher: { id: 'nova' }, text: 'Oh, hello there.' });

  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], {
    Engine: 'generative',
    LanguageCode: 'en-US',
    OutputFormat: 'mp3',
    SampleRate: '24000',
    Text: 'Oh, hello there.',
    TextType: 'text',
    VoiceId: 'Salli',
  });
  assert.equal(first.cached, false);
  assert.equal(replay.cached, true);
  assert.deepEqual([...replay.audio], [73, 68, 51, 4]);
  assert.deepEqual(service.capabilities({ id: 'nova' }).profile, {
    characterId: 'nova',
    quality: 'generative',
    character: 'Youthful adult, light, and playful',
  });
});

test('falls back to neural speech when the generative engine is temporarily rejected', async () => {
  const requests = [];
  const client = {
    async send(command) {
      requests.push(command.input);
      if (requests.length === 1) {
        const error = new Error('engine unavailable');
        error.name = 'EngineNotSupportedException';
        throw error;
      }
      return { AudioStream: Uint8Array.from([1]), ContentType: 'audio/mpeg' };
    },
  };
  const service = createTeacherSpeechService({}, { client, region: 'us-east-1' });
  await service.synthesize({ teacher: { id: 'mace' }, text: 'Again, slowly.' });
  assert.deepEqual(requests.map((request) => request.Engine), ['generative', 'neural']);
  assert.deepEqual(requests.map((request) => request.VoiceId), ['Stephen', 'Stephen']);
});
