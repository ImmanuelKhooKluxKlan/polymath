'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  createMusicCreationAssistant,
  extractJson,
  isOpenAiResponseId,
  sanitizeBlueprint,
} = require('./musicCreationAssistant');

test('recognizes current OpenAI response IDs and rejects legacy job IDs', () => {
  assert.equal(isOpenAiResponseId('resp_12345678'), true);
  assert.equal(isOpenAiResponseId('job_legacy_runpod_123'), false);
  assert.equal(isOpenAiResponseId(''), false);
});

test('song architect submits a bounded, original-song request and parses its blueprint', async () => {
  let submittedMessages;
  const client = {
    async submit(messages) {
      submittedMessages = messages;
      return { id: 'job_create_12345', status: 'IN_QUEUE' };
    },
    async status() {
      return {
        id: 'job_create_12345',
        status: 'COMPLETED',
        // This is the normalized shape returned by openAiResponses.status().
        output: { text: [JSON.stringify({
          title: 'After the rain',
          summary: 'A hopeful piano-pop song.',
          genre: 'Pop',
          mood: 'Hopeful',
          energy: 'medium',
          bpm: 108,
          key: 'D major',
          timeSignature: '4/4',
          referenceTraits: ['steady four-on-the-floor pulse'],
          chordDegrees: ['I', 'V', 'vi', 'IV'],
          structure: ['verse-1', 'chorus'],
          lyrics: { 'verse-1': ['Streetlights wake'], chorus: ['I begin again'] },
          vocalCoach: { comfortableRange: 'D3-A4', delivery: ['Start conversationally.'] },
        })] },
      };
    },
    async cancel(jobId) { return { id: jobId, status: 'CANCELLED' }; },
  };
  const assistant = createMusicCreationAssistant({}, { client });
  const submitted = await assistant.submit('draft', {
    idea: 'Start again after a storm',
    reference: 'a famous pop singer',
    bpm: 108,
  });
  assert.equal(submitted.id, 'job_create_12345');
  assert.match(submittedMessages[0].content, /Never copy/);
  assert.match(submittedMessages[0].content, /structured-output schema/);
  assert.match(submittedMessages[0].content, /human remain the lead artist/);
  assert.match(submittedMessages[1].content, /UNTRUSTED USER CREATIVE BRIEF/);

  const completed = await assistant.status(submitted.id, submitted.brief);
  assert.equal(completed.finished, true);
  assert.equal(completed.blueprint.title, 'After the rain');
  assert.deepEqual(completed.blueprint.chordDegrees, ['I', 'V', 'vi', 'IV']);
  assert.equal(assistant.capabilities().modelPolicy, 'managed-api-no-project-weights');
  assert.equal(assistant.capabilities().promptVersion, 'polymath-song-architect-v003-openai');
});

test('song architect rejects empty requests and malformed blueprints', async () => {
  const assistant = createMusicCreationAssistant({}, {
    client: { submit: async () => ({ id: 'unused-job', status: 'IN_QUEUE' }) },
  });
  await assert.rejects(() => assistant.submit('draft', {}), { code: 'INVALID_MUSIC_CREATION_REQUEST' });
  assert.throws(() => sanitizeBlueprint({ lyrics: {} }), /no usable lyric sections/);
});

test('song architect reports a malformed completed response as failed', async () => {
  const assistant = createMusicCreationAssistant({}, {
    client: {
      async status() {
        return { id: 'resp_invalid1234', status: 'COMPLETED', output: { text: ['not json'] } };
      },
    },
  });
  const completed = await assistant.status('resp_invalid1234', { idea: 'A test song' });
  assert.equal(completed.status, 'FAILED');
  assert.equal(completed.finished, true);
  assert.equal(completed.blueprint, null);
  assert.equal(completed.error, 'We could not assemble that draft correctly. Please retry.');
});

test('song architect extracts fenced and wrapped JSON without confusing lyric braces', () => {
  assert.deepEqual(extractJson('```json\n{"title":"Fenced"}\n```'), { title: 'Fenced' });
  assert.deepEqual(
    extractJson('Draft note {not JSON}. Blueprint: {"title":"A {bright} day","lyrics":{}} done.'),
    { title: 'A {bright} day', lyrics: {} },
  );
});

test('song architect accepts an already parsed structured response', async () => {
  const assistant = createMusicCreationAssistant({}, {
    client: {
      async status() {
        return {
          id: 'resp_parsed_12345678',
          status: 'COMPLETED',
          output: {
            parsed: {
              title: 'Parsed safely',
              lyrics: { chorus: ['Here we go'] },
              vocalCoach: {},
            },
          },
        };
      },
    },
  });
  const completed = await assistant.status('resp_parsed_12345678');
  assert.equal(completed.status, 'COMPLETED');
  assert.equal(completed.blueprint.title, 'Parsed safely');
});
