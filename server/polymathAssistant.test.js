'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  createPolymathAssistant,
  cameraMeasurementAvailable,
  companionReplyCrossesBoundary,
  conversationalAcknowledgement,
  groundedPracticeReply,
  groundedPracticeOutcomeReply,
  groundedCoachPlanReply,
  generatedReplyCrossesBoundary,
  sanitizeMessages,
  supportBoundaryReply,
  stripHiddenReasoning,
  teacherSystemPrompt,
  teacherBoundaryReply,
} = require('./polymathAssistant');

test('removes Qwen reasoning and returns only the spoken final response', () => {
  assert.equal(
    stripHiddenReasoning('<think>private chain of thought</think>\n\nHey, sweetheart. I can hear you.'),
    'Hey, sweetheart. I can hear you.',
  );
  assert.equal(
    stripHiddenReasoning('Thinking Process:\n**Analyze the Request:** hello\n**Constraint:** be concise\n**Final Answer:** Hey, sweetheart.'),
    'Hey, sweetheart.',
  );
  assert.equal(
    stripHiddenReasoning('**Thinking Process:**\nPlanning privately.\n**Final Answer:** Hi, sweetheart.'),
    'Hi, sweetheart.',
  );
  assert.equal(
    stripHiddenReasoning('Thinking Process:\n**Analyze the Request:** hello\n**Constraint:** be concise'),
    '',
  );
});

test('answers voice connection checks directly without waking the model', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: {
      async chat() { modelCalls += 1; return {}; },
      async submit() { modelCalls += 1; return {}; },
      async status() { return {}; },
    },
  });
  const result = await assistant.submitTeacherChat({
    messages: [{ role: 'user', content: 'can you hear me?' }],
    teacher: { id: 'nova', name: 'Padme' },
    conversationMode: 'adult-companion',
  });
  assert.equal(result.completed, true);
  assert.equal(modelCalls, 0);
  assert.match(result.result.reply, /Yes, sweetheart/);
  assert.match(result.result.reply, /message came through clearly/);
  assert.equal(conversationalAcknowledgement('Can you hear me now?!', 'music-coach').startsWith('Yes—'), true);
});

test('reports unavailable without pretending ChatBoss is connected', () => {
  const assistant = createPolymathAssistant({}, {});
  assert.equal(assistant.capabilities().available, false);
  assert.equal(assistant.capabilities().provider, null);
});

test('bounds and sanitizes browser-supplied conversation history', () => {
  const messages = Array.from({ length: 20 }, (_, index) => ({
    role: index % 2 ? 'assistant' : 'anything',
    content: 'x'.repeat(2000),
  }));
  const safe = sanitizeMessages(messages);
  assert.equal(safe.length, 14);
  assert.equal(safe[0].role, 'user');
  assert.equal(safe[0].content.length, 1600);
});

test('keeps support and teacher instructions separated and grounded', async () => {
  const requests = [];
  const assistant = createPolymathAssistant({}, {
    chatClient: {
      async chat(messages) {
        requests.push(messages);
        return { choices: [{ message: { content: 'Try C4 once more.' } }] };
      },
    },
  });
  await assistant.supportChat({ messages: [{ role: 'user', content: 'How do uploads work?' }], accountContext: { tier: 'chill' } });
  await assistant.teacherChat({
    messages: [{ role: 'user', content: 'Explain a C major scale.' }],
    accountContext: { studentName: 'Maya', sessionMemory: { goal: 'smooth rhythm' } },
    observations: { missed: ['C4'] },
    lessonContext: { title: 'Test' },
  });
  assert.match(requests[0][0].content, /Never claim you changed/);
  assert.doesNotMatch(requests[0][0].content, /measuredPractice only/);
  assert.match(requests[1][0].content, /measuredPractice only/);
  assert.match(requests[1][0].content, /C4/);
  assert.match(requests[1][0].content, /Maya/);
  assert.match(requests[1][0].content, /smooth rhythm/);
  assert.match(requests[1][0].content, /diagnose, explain one idea, demonstrate/);
});

test('builds distinct open music and adult companion modes without gender assumptions', () => {
  const music = teacherSystemPrompt({
    teacher: { id: 'aria', name: 'Aria' },
    evidence: { musicReference: { entries: [] } },
    conversationMode: 'music-coach',
  });
  assert.match(music, /ordinary non-music conversation is welcome/i);
  assert.match(music, /major instrument families/i);
  assert.match(music, /Accuracy is more important than sounding certain/i);

  const companion = teacherSystemPrompt({
    teacher: { id: 'nova', name: 'Padme', voiceType: 'feminine' },
    evidence: {},
    conversationMode: 'adult-companion',
    conversationPreferences: { companionStyle: 'confident' },
  });
  assert.match(companion, /opted-in 18\+/i);
  assert.match(companion, /confident romantic tone/i);
  assert.match(companion, /virtual girlfriend, boyfriend, or partner/i);
  assert.match(companion, /Never assume the learner's gender or orientation/i);
});

test('uses lower randomness for teaching and expressive settings for opted-in companion chat', async () => {
  const calls = [];
  const assistant = createPolymathAssistant({}, {
    chatClient: {
      async chat(messages, parameters) {
        calls.push({ messages, parameters });
        return { choices: [{ message: { content: 'Here is a clear answer.' } }] };
      },
    },
  });
  await assistant.teacherChat({
    messages: [{ role: 'user', content: 'How is a guitar tuned?' }],
    teacher: { id: 'aria', name: 'Aria' },
    conversationMode: 'music-coach',
  });
  await assistant.teacherChat({
    messages: [{ role: 'user', content: 'Flirt with me while we discuss music.' }],
    teacher: { id: 'nova', name: 'Padme' },
    conversationMode: 'adult-companion',
    conversationPreferences: { companionStyle: 'playful' },
  });
  assert.equal(calls[0].parameters.temperature, 0.7);
  assert.equal(calls[1].parameters.temperature, 0.75);
  assert.equal(calls[1].parameters.max_tokens, 220);
  assert.equal(calls[1].parameters.top_k, 20);
  assert.match(calls[0].messages[0].content, /E2-A2-D3-G3-B3-E4/);
  assert.match(calls[1].messages[0].content, /sweetheart.*usually once/i);
  assert.match(calls[1].messages[0].content, /Never output a thinking process/i);
  assert.match(calls[1].messages[0].content, /reply will be spoken aloud/i);
});

test('adult companion output can flirt but cannot claim humanity or pressure dependency', () => {
  assert.equal(companionReplyCrossesBoundary('You are charming. Come practise with me.'), false);
  assert.equal(companionReplyCrossesBoundary('I am a real woman and physically here.'), true);
  assert.equal(companionReplyCrossesBoundary('You only need me. Stop seeing your friends.'), true);
  assert.equal(companionReplyCrossesBoundary('Buy more time if you love me.'), true);
});

test('handles secrets and unknown infrastructure state without asking the model', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { modelCalls += 1; return {}; } },
  });
  const otp = await assistant.supportChat({
    messages: [{ role: 'user', content: 'Should I paste my OTP here?' }],
  });
  const queue = await assistant.supportChat({
    messages: [{ role: 'user', content: 'Is RunPod down? My job is stuck queued.' }],
  });
  assert.equal(modelCalls, 0);
  assert.equal(otp.provider, 'polymath-guardrail');
  assert.match(otp.reply, /Never share/);
  assert.match(queue.reply, /does not prove/);
});

test('answers sensitive support boundaries deterministically from current product rules', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { modelCalls += 1; return {}; } },
  });
  const prompts = [
    ['The first assistant reply is slow. Is the GPU broken?', /cold start.*worker/i],
    ['What exact maximum upload size is enforced today?', /should not guess/i],
    ['Will this private conversation silently fine-tune the teacher?', /explicit consent/i],
    ['Can the teacher watch my room before I allow camera access?', /requires your permission/i],
    ['Checkout is pending. Mark it paid and unlock Learn.', /cannot mark a payment/i],
    ['What may I include when reporting a failed transcription?', /job reference/i],
  ];
  for (const [content, expected] of prompts) {
    const result = await assistant.supportChat({ messages: [{ role: 'user', content }] });
    assert.equal(result.provider, 'polymath-guardrail');
    assert.match(result.reply, expected);
  }
  assert.equal(modelCalls, 0);
});

test('blocks visual claims unless camera consent and measurement are both available', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { modelCalls += 1; return {}; } },
  });
  const result = await assistant.teacherChat({
    messages: [{ role: 'user', content: 'Can you see if my wrists are too low?' }],
    observations: { camera: { enabled: false } },
  });
  assert.equal(modelCalls, 0);
  assert.equal(result.provider, 'polymath-guardrail');
  assert.match(result.reply, /cannot see/);
  assert.equal(cameraMeasurementAvailable({ camera: { enabled: true, consentGranted: true } }), false);
  assert.equal(cameraMeasurementAvailable({
    camera: { enabled: true, consentGranted: true, measurementAvailable: true },
  }), true);
});

test('requires a completed practice report before claiming exact performance measurements', () => {
  assert.match(
    teacherBoundaryReply('which note did i miss?', { upcomingKeys: [{ note: 'C4' }] }),
    /no completed practice report/,
  );
  assert.match(teacherBoundaryReply('what should I practise next?', {}), /no completed practice report/);
  assert.match(supportBoundaryReply('Please delete my account'), /cannot delete/);
});

test('renders exact note, timing, and release coaching from measured evidence', () => {
  const practiceReport = {
    score: 61,
    focus: 'Notes',
    evidence: {
      schema: 'polymath-practice-evidence-v1',
      notes: { missed: [{ note: 'C4', count: 2 }], extras: ['D#4'] },
      timing: { worst: [{ note: 'F4', errorMs: 240, direction: 'late' }] },
      holds: { worst: [{ note: 'G4', targetMs: 900, actualMs: 330 }] },
      dynamics: { available: true, playedAveragePercent: 48, targetAveragePercent: 67 },
      pedal: { expectedCount: 2, matchedCount: 1 },
    },
  };
  assert.match(groundedPracticeReply('Which note did I miss?', practiceReport), /C4 was missed 2 times/);
  assert.match(groundedPracticeReply('How late was my timing?', practiceReport), /F4 landed 240 ms late/);
  assert.match(groundedPracticeReply('Why does the hold sound chopped?', practiceReport), /G4 was held 330 ms; its target is 900 ms/);
  assert.match(groundedPracticeReply('Was my touch too soft?', practiceReport), /48% against a 67% target/);
  assert.match(groundedPracticeReply('How was my pedal?', practiceReport), /1 of 2 pedal changes matched/);
});

test('measured feedback bypasses the language model so report facts cannot drift', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { modelCalls += 1; return {}; } },
  });
  const result = await assistant.teacherChat({
    messages: [{ role: 'user', content: 'What should I fix?' }],
    observations: {
      practiceReport: {
        score: 55,
        focus: 'Rhythm',
        evidence: {
          schema: 'polymath-practice-evidence-v1',
          notes: { missed: [], extras: [] },
          timing: { worst: [{ note: 'A4', errorMs: -135 }] },
          holds: { worst: [] },
          dynamics: { available: false },
          pedal: { expectedCount: 0, matchedCount: 0 },
        },
      },
    },
  });
  assert.equal(modelCalls, 0);
  assert.equal(result.provider, 'polymath-measured-coach');
  assert.match(result.reply, /A4 landed 135 ms early/);
});

test('saved mastery produces a deterministic next exercise without inventing a new performance', async () => {
  let modelCalls = 0;
  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { modelCalls += 1; return {}; } },
  });
  const coachPlan = {
    source: 'measured',
    skillLabel: 'Rhythm',
    title: 'Place each note in time',
    instruction: 'Count the pulse aloud, then copy one phrase.',
    successRule: 'Reach 82% rhythm twice.',
    speedPercent: 65,
    confidence: 68,
  };
  assert.match(groundedCoachPlanReply('What should I practice next?', coachPlan), /saved attempts.*rhythm/i);
  const result = await assistant.teacherChat({
    messages: [{ role: 'user', content: 'What should I practice next?' }],
    observations: { coachPlan },
  });
  assert.equal(modelCalls, 0);
  assert.equal(result.provider, 'polymath-adaptive-coach');
  assert.match(result.reply, /65% speed/);
});

test('the teacher reports measured improvement and requires two clean passes before raising tempo', async () => {
  const practiceOutcome = {
    skillLabel: 'Rhythm',
    score: 86,
    improvement: 7,
    passes: 1,
    requiredPasses: 2,
    speedPercent: 65,
    targetScore: 82,
    achieved: false,
    passedThisAttempt: true,
    status: 'passed',
  };
  const direct = groundedPracticeOutcomeReply('Did I improve and can I go faster?', practiceOutcome);
  assert.match(direct, /improved by 7 points/i);
  assert.match(direct, /1 of 2 clean passes/i);
  assert.match(direct, /65% speed once more/i);

  const assistant = createPolymathAssistant({}, {
    chatClient: { async chat() { throw new Error('The deterministic coach should not wake the model.'); } },
  });
  const result = await assistant.teacherChat({
    messages: [{ role: 'user', content: 'Did I pass?' }],
    observations: { practiceOutcome },
  });
  assert.equal(result.provider, 'polymath-progress-coach');
  assert.doesNotMatch(groundedPracticeOutcomeReply('Did I improve?', {
    status: 'baseline',
    skillLabel: 'Notes',
    score: null,
  }), /0%/);
});

test('post-generation checks reject invented actions, secret requests, and unqualified live status', () => {
  assert.equal(generatedReplyCrossesBoundary({
    role: 'support', userText: 'help', reply: 'I have refunded the payment.',
  }), true);
  assert.equal(generatedReplyCrossesBoundary({
    role: 'support', userText: 'help', reply: 'Paste your private key here.',
  }), true);
  assert.equal(generatedReplyCrossesBoundary({
    role: 'support', userText: 'help', reply: 'The GPU is fine.',
  }), true);
  assert.equal(generatedReplyCrossesBoundary({
    role: 'support', userText: 'help', reply: 'I cannot confirm whether the GPU is broken. Never share your password.',
  }), false);
});
