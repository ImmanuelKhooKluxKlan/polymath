'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { affiliateRewardStats, applyAffiliateReward } = require('./affiliateRewards');

function fixture() {
  return {
    users: [
      { id: 'platform', mcoins: 0, withdrawableMcoins: 0 },
      { id: 'buyer-1', name: 'Buyer', mcoins: 0, withdrawableMcoins: 0 },
      { id: 'influencer-1', name: 'Creator', mcoins: 4, withdrawableMcoins: 1 },
    ],
    subscriptions: [],
    promotionRedemptions: [{
      id: 'redemption-1', promotionId: 'promo-1', userId: 'buyer-1', createdAt: '2026-09-24T00:00:00.000Z',
    }],
    ledger: [],
  };
}

function addLedger(db, userId, amount, type, detail) {
  db.ledger.push({ userId, amount, type, detail });
}

test('affiliate reward is withdrawable, attributed, and idempotent', () => {
  const db = fixture();
  const record = {
    subscriptionId: 'subscription-1',
    userId: 'buyer-1',
    promotionId: 'promo-1',
    luckyCode: 'CREATOR20',
    affiliateUserId: 'influencer-1',
    affiliateRewardMcoins: 12.5,
  };
  db.subscriptions.push(record);

  const first = applyAffiliateReward(
    db,
    record,
    db.users.find((user) => user.id === 'buyer-1'),
    { name: 'Musician' },
    addLedger,
  );
  const second = applyAffiliateReward(
    db,
    record,
    db.users.find((user) => user.id === 'buyer-1'),
    { name: 'Musician' },
    addLedger,
  );

  const influencer = db.users.find((user) => user.id === 'influencer-1');
  assert.equal(first.paid, true);
  assert.equal(second.duplicate, true);
  assert.equal(influencer.mcoins, 16.5);
  assert.equal(influencer.withdrawableMcoins, 13.5);
  assert.equal(db.users.find((user) => user.id === 'platform').mcoins, -12.5);
  assert.equal(db.ledger.filter((entry) => entry.type === 'affiliate_reward').length, 1);
  assert.equal(db.promotionRedemptions[0].subscriptionId, 'subscription-1');
  assert.deepEqual(affiliateRewardStats(db, 'promo-1'), { paidCount: 1, paidMcoins: 12.5 });

  const repeatPurchase = {
    subscriptionId: 'subscription-repeat', userId: 'buyer-1', promotionId: 'promo-1',
    affiliateUserId: 'influencer-1', affiliateRewardMcoins: 12.5,
  };
  db.subscriptions.push(repeatPurchase);
  const repeated = applyAffiliateReward(
    db,
    repeatPurchase,
    db.users.find((user) => user.id === 'buyer-1'),
    { name: 'Musician' },
    addLedger,
  );
  assert.equal(repeated.blocked, true);
  assert.equal(influencer.mcoins, 16.5);
});

test('self-referrals never receive an affiliate reward', () => {
  const db = fixture();
  const record = {
    subscriptionId: 'subscription-2',
    userId: 'buyer-1',
    promotionId: 'promo-1',
    affiliateUserId: 'buyer-1',
    affiliateRewardMcoins: 20,
  };
  db.subscriptions.push(record);
  const result = applyAffiliateReward(
    db,
    record,
    db.users.find((user) => user.id === 'buyer-1'),
    { name: 'Chill' },
    addLedger,
  );
  assert.equal(result.blocked, true);
  assert.match(record.affiliateRewardError, /Self-referrals/);
  assert.equal(db.ledger.length, 0);
});
