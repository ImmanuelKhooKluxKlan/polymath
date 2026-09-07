'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  createTeacherAssistant,
  parseSceneJson,
  validateImageDataUrl,
} = require('./teacherAssistant');

test('reports unavailable remote capabilities without pretending they work', () => {
  const assistant = createTeacherAssistant({}, {});
  assert.deepEqual(assistant.capabilities(), {
    conversation: false,
    coaching: true,
    browserSpeechInput: true,
    browserSpeechOutput: true,
    localKeyboardVision: true,
    generalSceneVision: false,
    scenePrivacy: 'Snapshots are sent only when the learner presses Look. They are not retained by Polymath.',
  });
});

test('grounds chat in measured observations and explicit scene evidence', async () => {
  let submitted;
  const assistant = createTeacherAssistant({}, {
    chatClient: {
      async chat(messages) {
        submitted = messages;
        return { choices: [{ message: { content: 'Hold C4 longer, then repeat it.' } }] };
      },
    },
  });
  const result = await assistant.chat({
    messages: [{ role: 'user', content: 'How was that?' }],
    lessonContext: { title: 'Test song', currentTime: 12.5 },
    observations: [{ type: 'hold-short', note: 'C4', earlyBySeconds: 0.31 }],
  });
  assert.equal(result.reply, 'Hold C4 longer, then repeat it.');
  assert.match(submitted[0].content, /Never invent/);
  assert.match(submitted[0].content, /hold-short/);
  assert.equal(submitted.at(-1).content, 'How was that?');
});

test('validates camera image data and normalizes scene JSON', () => {
  const valid = 'data:image/jpeg;base64,AQIDBA==';
  assert.equal(validateImageDataUrl(valid), valid);
  assert.throws(() => validateImageDataUrl('data:text/plain;base64,AQID'), /JPEG/);
  assert.deepEqual(parseSceneJson('```json\n{"summary":"A blue mug","objects":[{"name":"mug","attributes":"blue","confidence":0.93}],"pianoVisible":false,"uncertainty":""}\n```'), {
    summary: 'A blue mug',
    objects: [{ name: 'mug', attributes: 'blue', confidence: 0.93 }],
    pianoVisible: false,
    uncertainty: '',
  });
});

test('uses an OpenAI-compatible vision endpoint only for explicit snapshots', async () => {
  let request;
  const assistant = createTeacherAssistant({
    TEACHER_VISION_BASE_URL: 'https://vision.example/v1',
    TEACHER_VISION_API_KEY: 'test-key',
    TEACHER_VISION_MODEL: 'test-vlm',
  }, {
    async fetch(url, options) {
      request = { url, ...options };
      return {
        ok: true,
        async json() {
          return { choices: [{ message: { content: '{"summary":"A toy","objects":[{"name":"toy","attributes":"red","confidence":0.8}],"pianoVisible":false,"uncertainty":""}' } }] };
        },
      };
    },
  });
  const scene = await assistant.analyzeScene({
    imageDataUrl: 'data:image/png;base64,AQIDBA==',
    prompt: 'What did I buy?',
  });
  assert.equal(scene.objects[0].name, 'toy');
  assert.equal(request.url, 'https://vision.example/v1/chat/completions');
  const body = JSON.parse(request.body);
  assert.equal(body.messages[0].content[1].type, 'image_url');
});
