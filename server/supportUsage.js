'use strict';

const SUPPORT_DAILY_QUESTION_LIMIT = 7;

function validDate(value) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isFinite(date.getTime()) ? date : new Date();
}

function utcDayKey(value = new Date()) {
  return validDate(value).toISOString().slice(0, 10);
}

function nextUtcReset(value = new Date()) {
  const now = validDate(value);
  return new Date(Date.UTC(
    now.getUTCFullYear(),
    now.getUTCMonth(),
    now.getUTCDate() + 1,
  )).toISOString();
}

function storedUsage(user, dateKey) {
  const raw = user?.supportQuestionUsage;
  if (!raw || raw.utcDate !== dateKey) return 0;
  const used = Math.floor(Number(raw.usedQuestions));
  return Number.isFinite(used) ? Math.max(0, used) : 0;
}

function supportQuestionAllowance(user, { now = new Date(), unlimited = false } = {}) {
  const utcDate = utcDayKey(now);
  if (unlimited) {
    return {
      unlimited: true,
      dailyLimit: null,
      usedQuestions: 0,
      remainingQuestions: null,
      utcDate,
      resetsAt: nextUtcReset(now),
    };
  }
  const usedQuestions = Math.min(SUPPORT_DAILY_QUESTION_LIMIT, storedUsage(user, utcDate));
  return {
    unlimited: false,
    dailyLimit: SUPPORT_DAILY_QUESTION_LIMIT,
    usedQuestions,
    remainingQuestions: Math.max(0, SUPPORT_DAILY_QUESTION_LIMIT - usedQuestions),
    utcDate,
    resetsAt: nextUtcReset(now),
  };
}

function reserveSupportQuestion(user, options = {}) {
  const allowance = supportQuestionAllowance(user, options);
  if (allowance.unlimited) return { allowed: true, reserved: false, allowance };
  if (allowance.remainingQuestions <= 0) return { allowed: false, reserved: false, allowance };
  user.supportQuestionUsage = {
    utcDate: allowance.utcDate,
    usedQuestions: allowance.usedQuestions + 1,
  };
  return {
    allowed: true,
    reserved: true,
    utcDate: allowance.utcDate,
    allowance: supportQuestionAllowance(user, options),
  };
}

function refundSupportQuestion(user, reservation) {
  if (!reservation?.reserved || !reservation.utcDate) return false;
  const usage = user?.supportQuestionUsage;
  if (!usage || usage.utcDate !== reservation.utcDate) return false;
  usage.usedQuestions = Math.max(0, Math.floor(Number(usage.usedQuestions) || 0) - 1);
  return true;
}

module.exports = {
  SUPPORT_DAILY_QUESTION_LIMIT,
  nextUtcReset,
  refundSupportQuestion,
  reserveSupportQuestion,
  supportQuestionAllowance,
  utcDayKey,
};
