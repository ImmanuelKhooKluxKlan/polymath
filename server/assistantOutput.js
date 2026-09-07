'use strict';

function stripHiddenReasoning(value) {
  let text = String(value || '').trim();
  if (!text) return '';

  // Qwen normally wraps reasoning in <think>, but older RunPod request paths
  // can emit the scratchpad as ordinary content. Never expose either form.
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  if (/<think>/i.test(text)) text = text.slice(0, text.search(/<think>/i)).trim();

  const planningMarker = /(?:^|\n|\s)(?:#{1,6}\s*)?(?:\*\*)?(?:thinking process|analysis|reasoning|analy[sz]e the (?:input|request)|context|persona|constraint|determine the content|drafting options?|selecting the best fit)\s*:(?:\*\*)?/gi;
  const planningMatches = [...text.matchAll(planningMarker)];
  const looksLikeScratchpad = /^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:thinking process|analysis|reasoning)\s*:(?:\*\*)?/i.test(text)
    || planningMatches.length >= 2;

  if (looksLikeScratchpad) {
    const finalMarker = /(?:^|\n|\s)(?:#{1,6}\s*)?(?:\*\*)?(?:final answer|final response|response to (?:the )?user|spoken reply)\s*:(?:\*\*)?\s*/gi;
    const finals = [...text.matchAll(finalMarker)];
    if (!finals.length) return '';
    const last = finals.at(-1);
    text = text.slice(last.index + last[0].length).trim();
  }

  return text
    .replace(/^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:final answer|final response|response to (?:the )?user|spoken reply)\s*:(?:\*\*)?\s*/i, '')
    .replace(/^['"]|['"]$/g, '')
    .trim();
}

module.exports = { stripHiddenReasoning };
