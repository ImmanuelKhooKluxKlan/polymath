'use strict';

const DEFAULT_TEST_BUDGET_USD = 500;

const STATUS_LABELS = Object.freeze({
  setup: 'Set up',
  quality: 'Fix proof first',
  collect: 'Collect evidence',
  repair: 'Repair funnel',
  hold: 'Hold and learn',
  scale: 'Ready to scale',
});

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function count(value) {
  return Math.max(0, Math.round(finiteNumber(value)));
}

function percent(numerator, denominator) {
  return denominator > 0
    ? Math.round((finiteNumber(numerator) / finiteNumber(denominator)) * 1000) / 10
    : null;
}

function stageActors(campaign, id) {
  return count(campaign?.stages?.find((stage) => stage.id === id)?.actors);
}

function isAffiliateSubscriptionOffer(promotion) {
  return ['subscription_percent', 'subscription_fixed'].includes(promotion?.kind)
    && Boolean(promotion?.affiliateUserId)
    && promotion?.retired !== true;
}

function teacherPartnerExperiment(analytics, promotions) {
  const programs = promotions.filter(isAffiliateSubscriptionOffer);
  const referralByCode = new Map((analytics?.referrals || []).map((entry) => [
    String(entry.promotionCode || '').trim().toUpperCase(),
    entry,
  ]));
  const evidence = programs.map((program) => (
    referralByCode.get(String(program.code || '').trim().toUpperCase()) || {}
  ));
  const visits = evidence.reduce((total, item) => total + count(item.visitors), 0);
  const claims = evidence.reduce((total, item) => total + count(item.claims), 0);
  const checkouts = evidence.reduce((total, item) => total + count(item.checkouts), 0);
  const activations = evidence.reduce((total, item) => total + count(item.activations), 0);
  const checkoutRate = percent(checkouts, visits);
  const activationRate = percent(activations, visits);
  const activePrograms = programs.filter((program) => program.active !== false).length;

  let status = 'setup';
  let verdict = 'Create one measurable partner offer before recruiting anyone.';
  let nextAction = 'Choose one teacher or small piano creator, then create a unique 10% customer / 10% creator code.';
  if (programs.length > 0 && visits < 100) {
    status = 'collect';
    verdict = `${Math.max(0, 100 - visits)} more tracked visits are needed before judging the offer.`;
    nextAction = 'Give each selected partner one useful demonstration brief. Change neither the offer nor the landing page until the sample reaches 100 visits.';
  } else if (programs.length > 0 && (finiteNumber(checkoutRate) < 3 || finiteNumber(activationRate) < 1)) {
    status = 'repair';
    verdict = 'The current funnel has enough traffic to show a conversion problem.';
    nextAction = finiteNumber(checkoutRate) < 3
      ? 'Do not recruit more partners yet. Rewrite the proof and offer on the first screen, then run a fresh controlled test.'
      : 'Inspect payment friction and buyer trust between checkout and verified activation before sending more traffic.';
  } else if (visits >= 100 && activations >= 3 && finiteNumber(activationRate) >= 3) {
    status = 'scale';
    verdict = 'Verified paid activation has cleared the first scale gate.';
    nextAction = 'Clone the winning content format to five closely matched partners. Keep unique codes so weak partners can still be stopped individually.';
  } else if (programs.length > 0) {
    status = 'hold';
    verdict = 'The offer has signal, but not enough verified paid conversion to scale safely.';
    nextAction = 'Interview activated and abandoned buyers, change one variable, and collect another 100 visits.';
  }

  return {
    id: 'teacher-partners',
    title: 'Founding teacher partners',
    audience: 'Piano teachers and small piano-cover creators whose viewers ask for sheets or lessons.',
    status,
    statusLabel: STATUS_LABELS[status],
    verdict,
    nextAction,
    actionId: 'prepare-partner-offer',
    actionLabel: programs.length ? 'Prepare another partner offer' : 'Prepare the first offer',
    metrics: [
      { id: 'partners', label: 'Partner codes', value: programs.length, detail: `${activePrograms} active` },
      { id: 'visits', label: 'Tracked visits', value: visits, detail: visits < 100 ? `${100 - visits} to decision` : 'sample gate met' },
      { id: 'checkouts', label: 'Checkouts', value: checkouts, detail: checkoutRate === null ? 'not enough data' : `${checkoutRate}% of visits` },
      { id: 'activations', label: 'Paid activations', value: activations, detail: activationRate === null ? 'not enough data' : `${activationRate}% of visits` },
    ],
    scaleRule: 'Scale only after at least 100 tracked visits, 3 verified activations, and a 3% visit-to-activation rate.',
    stopRule: 'At 100 visits, repair or pause when checkout is below 3% or verified activation is below 1%.',
    brief: 'Show one real 15–30 second song problem, demonstrate the playable result, then invite viewers to try that exact outcome with the disclosed partner link.',
    safety: 'No upfront sponsorship and no reward for clicks. Commission is paid only after a verified subscription activation.',
    trustedActivations: activations,
    activationRatePercent: activationRate,
  };
}

function songProofExperiment(analytics, campaigns) {
  const candidates = campaigns.filter((campaign) => campaign?.status !== 'archived');
  const published = candidates.filter((campaign) => (
    campaign?.status === 'published' && campaign?.live !== false
  ));
  const measured = published.length ? published : candidates;
  const measuredIds = new Set(measured.map((campaign) => campaign.id));
  const evidence = (analytics?.campaigns || []).filter((campaign) => measuredIds.has(campaign.campaignId));
  const views = evidence.reduce((total, item) => total + stageActors(item, 'viewed'), 0);
  const loaded = evidence.reduce((total, item) => total + stageActors(item, 'loaded'), 0);
  const attempted = evidence.reduce((total, item) => total + stageActors(item, 'attempted'), 0);
  const completed = evidence.reduce((total, item) => total + stageActors(item, 'completed'), 0);
  const shared = evidence.reduce((total, item) => total + stageActors(item, 'shared'), 0);
  const activations = evidence.reduce((total, item) => total + stageActors(item, 'activated'), 0);
  const loadRate = percent(loaded, views);
  const attemptRate = percent(attempted, views);
  const completionRate = percent(completed, attempted);
  const shareRate = percent(shared, completed);
  const activationRate = percent(activations, views);
  const lowestQa = published.length
    ? Math.min(...published.map((campaign) => finiteNumber(campaign.qaScore ?? campaign.verification?.qaScore)))
    : null;

  let status = 'setup';
  let verdict = 'Build one rights-cleared, human-verified song proof instead of a generic advertisement.';
  let nextAction = 'Use an original, licensed, or public-domain 15–30 second excerpt and verify the playable result with a musician.';
  if (candidates.length > 0 && published.length === 0) {
    status = 'quality';
    verdict = 'A draft exists, but no challenge has cleared the publication gates.';
    nextAction = 'Finish rights evidence, human verification, and an 80/100 or better QA check before seeking traffic.';
  } else if (published.length > 0 && finiteNumber(lowestQa) < 80) {
    status = 'quality';
    verdict = 'A published proof has fallen below the product quality gate.';
    nextAction = 'Pause distribution and re-verify the musical excerpt before collecting more views.';
  } else if (published.length > 0 && views < 100) {
    status = 'collect';
    verdict = `${Math.max(0, 100 - views)} more tracked views are needed before judging the proof.`;
    nextAction = 'Share the same proof hook organically. Keep the excerpt, headline, and call to action unchanged until 100 views.';
  } else if (published.length > 0 && (
    finiteNumber(loadRate) < 70
    || finiteNumber(attemptRate) < 35
    || finiteNumber(completionRate) < 40
  )) {
    status = 'repair';
    verdict = 'The challenge has enough views to locate a product-experience leak.';
    if (finiteNumber(loadRate) < 70) nextAction = 'Fix loading speed or compatibility before promoting this challenge again.';
    else if (finiteNumber(attemptRate) < 35) nextAction = 'Make the first playable action and value proposition clearer; do not buy more impressions.';
    else nextAction = 'Shorten or simplify the excerpt so more starters reach a scored result.';
  } else if (published.length > 0
      && views >= 100
      && completed >= 20
      && finiteNumber(completionRate) >= 40
      && finiteNumber(shareRate) >= 10
      && activations >= 2
      && finiteNumber(activationRate) >= 2) {
    status = 'scale';
    verdict = 'The proof creates completed attempts, sharing, and verified paid activation.';
    nextAction = 'Amplify only this exact winning post to a tightly matched piano audience, with a small capped test and no creative changes.';
  } else if (published.length > 0) {
    status = 'hold';
    verdict = 'People can use the proof, but sharing or paid conversion has not cleared the scale gate.';
    nextAction = 'Ask completers why they did or did not share or subscribe. Change one element, then run a new 100-view test.';
  }

  return {
    id: 'song-proof',
    title: 'Playable song proof',
    audience: 'Curious players who want to hear and try a recognizable musical outcome before signing in.',
    status,
    statusLabel: STATUS_LABELS[status],
    verdict,
    nextAction,
    actionId: 'prepare-song-proof',
    actionLabel: candidates.length ? 'Prepare a controlled proof' : 'Build the first proof',
    metrics: [
      { id: 'proofs', label: 'Published proofs', value: published.length, detail: `${candidates.length - published.length} private draft${candidates.length - published.length === 1 ? '' : 's'}` },
      { id: 'views', label: 'Tracked views', value: views, detail: views < 100 ? `${100 - views} to decision` : 'sample gate met' },
      { id: 'attempts', label: 'Attempts', value: attempted, detail: attemptRate === null ? 'not enough data' : `${attemptRate}% of views` },
      { id: 'completions', label: 'Completed', value: completed, detail: completionRate === null ? 'not enough data' : `${completionRate}% of attempts` },
      { id: 'shares', label: 'Shares', value: shared, detail: shareRate === null ? 'not enough data' : `${shareRate}% of completions` },
      { id: 'activations', label: 'Paid activations', value: activations, detail: activationRate === null ? 'not enough data' : `${activationRate}% of views` },
    ],
    scaleRule: 'Scale only after 100 views, 20 completions, 40% attempt completion, 10% completion-to-share, and 2% verified activation.',
    stopRule: 'After 100 views, repair when fewer than 70% load, 35% attempt, or 40% of starters complete.',
    brief: '“Can Polymath hear this?” Show the original or licensed musical moment, its playable Polymath result, and one real attempt. The proof is the advertisement.',
    safety: 'Private uploads never become marketing. Publish only rights-cleared excerpts with artist approval and human musical QA.',
    trustedActivations: activations,
    activationRatePercent: activationRate,
  };
}

function budgetPlan(ceilingUsd, experiments) {
  const ceiling = Math.max(0, Math.min(1_000_000, finiteNumber(ceilingUsd, DEFAULT_TEST_BUDGET_USD)));
  const proofAndQa = Number((ceiling * 0.2).toFixed(2));
  const challengeReward = Number((ceiling * 0.2).toFixed(2));
  const amplification = Number((ceiling - proofAndQa - challengeReward).toFixed(2));
  const winner = experiments
    .filter((experiment) => experiment.status === 'scale')
    .sort((left, right) => (
      finiteNumber(right.trustedActivations) - finiteNumber(left.trustedActivations)
      || finiteNumber(right.activationRatePercent) - finiteNumber(left.activationRatePercent)
    ))[0];
  return {
    ceilingUsd: ceiling,
    automaticSpendUsd: 0,
    paidAmplificationEligible: Boolean(winner),
    winnerId: winner?.id || null,
    rule: winner
      ? `${winner.title} cleared its evidence gate. Any spend still requires administrator approval.`
      : 'Do not approve paid amplification until one experiment clears its scale gate.',
    trackingNotice: 'This is a planning ceiling, not an external accounting ledger. Promotion redemption caps remain the enforceable liability guardrail.',
    allocations: [
      { id: 'proof-qa', label: 'Proof asset and human QA', ceilingUsd: proofAndQa },
      { id: 'challenge', label: 'Capped challenge reward or teacher pilot support', ceilingUsd: challengeReward },
      { id: 'winner', label: 'Reserve for the single proven winner', ceilingUsd: amplification },
    ],
  };
}

function buildMarketingAutopilot({
  analytics = {},
  promotions = [],
  campaigns = [],
  budgetUsd = DEFAULT_TEST_BUDGET_USD,
} = {}) {
  const experiments = [
    teacherPartnerExperiment(analytics, Array.isArray(promotions) ? promotions : []),
    songProofExperiment(analytics, Array.isArray(campaigns) ? campaigns : []),
  ];
  return {
    generatedAt: new Date().toISOString(),
    principle: 'Two experiments. One variable at a time. No paid reach before verified product proof.',
    automationBoundary: 'Polymath diagnoses funnels and prepares launch kits. It never messages people, promises payouts, publishes music, or spends money without administrator action.',
    budget: budgetPlan(budgetUsd, experiments),
    experiments,
  };
}

module.exports = {
  DEFAULT_TEST_BUDGET_USD,
  buildMarketingAutopilot,
  songProofExperiment,
  teacherPartnerExperiment,
};
