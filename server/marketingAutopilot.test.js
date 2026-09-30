'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { buildMarketingAutopilot } = require('./marketingAutopilot');

function stages(values = {}) {
  return Object.entries(values).map(([id, actors]) => ({ id, actors }));
}

test('autopilot starts with exactly two no-spend experiments', () => {
  const result = buildMarketingAutopilot();
  assert.deepEqual(result.experiments.map((experiment) => experiment.id), [
    'teacher-partners',
    'song-proof',
  ]);
  assert.equal(result.budget.ceilingUsd, 500);
  assert.equal(result.budget.automaticSpendUsd, 0);
  assert.equal(result.budget.paidAmplificationEligible, false);
  assert.equal(result.experiments[0].status, 'setup');
  assert.equal(result.experiments[1].status, 'setup');
});

test('teacher campaign waits for a real sample then diagnoses weak conversion', () => {
  const promotion = {
    code: 'TEACH10', kind: 'subscription_percent', affiliateUserId: 'teacher_1', active: true,
  };
  const collecting = buildMarketingAutopilot({
    promotions: [promotion],
    analytics: { referrals: [{ promotionCode: 'TEACH10', visitors: 99, checkouts: 8, activations: 3 }] },
  });
  assert.equal(collecting.experiments[0].status, 'collect');

  const repair = buildMarketingAutopilot({
    promotions: [promotion],
    analytics: { referrals: [{ promotionCode: 'TEACH10', visitors: 100, checkouts: 2, activations: 0 }] },
  });
  assert.equal(repair.experiments[0].status, 'repair');
  assert.match(repair.experiments[0].nextAction, /Do not recruit more partners/);
});

test('verified teacher activation can unlock a scale recommendation without automatic spending', () => {
  const result = buildMarketingAutopilot({
    promotions: [{
      code: 'PIANO10', kind: 'subscription_percent', affiliateUserId: 'teacher_1', active: true,
    }],
    analytics: { referrals: [{ promotionCode: 'PIANO10', visitors: 120, checkouts: 12, activations: 5 }] },
  });
  assert.equal(result.experiments[0].status, 'scale');
  assert.equal(result.budget.paidAmplificationEligible, true);
  assert.equal(result.budget.winnerId, 'teacher-partners');
  assert.equal(result.budget.automaticSpendUsd, 0);
});

test('song proof requires publication quality and measurable use before scale', () => {
  const draft = buildMarketingAutopilot({
    campaigns: [{ id: 'campaign_1', status: 'draft', qaScore: 90 }],
  });
  assert.equal(draft.experiments[1].status, 'quality');

  const result = buildMarketingAutopilot({
    campaigns: [{ id: 'campaign_1', status: 'published', qaScore: 92 }],
    analytics: {
      campaigns: [{
        campaignId: 'campaign_1',
        stages: stages({ viewed: 120, loaded: 110, attempted: 70, completed: 40, shared: 8, activated: 4 }),
      }],
    },
  });
  assert.equal(result.experiments[1].status, 'scale');
  assert.equal(result.budget.winnerId, 'song-proof');
});

test('when both experiments qualify, the reserve follows more verified activations', () => {
  const result = buildMarketingAutopilot({
    promotions: [{
      code: 'TEACH10', kind: 'subscription_percent', affiliateUserId: 'teacher_1', active: true,
    }],
    campaigns: [{ id: 'campaign_1', status: 'published', qaScore: 94 }],
    analytics: {
      referrals: [{ promotionCode: 'TEACH10', visitors: 100, checkouts: 10, activations: 3 }],
      campaigns: [{
        campaignId: 'campaign_1',
        stages: stages({ viewed: 100, loaded: 95, attempted: 60, completed: 30, shared: 6, activated: 5 }),
      }],
    },
  });
  assert.deepEqual(result.experiments.map((experiment) => experiment.status), ['scale', 'scale']);
  assert.equal(result.budget.winnerId, 'song-proof');
});
