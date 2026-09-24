'use strict';

function rewardAmount(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return 0;
  return Math.max(0, Math.min(1_000_000_000, Math.round(amount * 100) / 100));
}

function affiliateRewardStats(db, promotionId) {
  const rewarded = (db?.subscriptions || []).filter((record) => (
    record.promotionId === promotionId
    && record.affiliateRewardPaidAt
    && Number(record.affiliateRewardMcoins) > 0
  ));
  return {
    paidCount: rewarded.length,
    paidMcoins: Number(rewarded.reduce(
      (total, record) => total + rewardAmount(record.affiliateRewardMcoins),
      0,
    ).toFixed(2)),
  };
}

function applyAffiliateReward(db, record, buyer, product, addLedger) {
  if (!record || record.affiliateRewardPaidAt || record.affiliateRewardBlockedAt) {
    return { paid: false, duplicate: Boolean(record?.affiliateRewardPaidAt) };
  }
  const amountMcoins = rewardAmount(record.affiliateRewardMcoins);
  const affiliateUserId = String(record.affiliateUserId || '').trim();
  if (!affiliateUserId || amountMcoins <= 0) return { paid: false, duplicate: false };

  const now = new Date().toISOString();
  if (!buyer?.id || buyer.id === affiliateUserId) {
    record.affiliateRewardBlockedAt = now;
    record.affiliateRewardError = 'Self-referrals are not eligible for influencer rewards.';
    return { paid: false, blocked: true };
  }
  const previousConversion = (db.subscriptions || []).find((candidate) => (
    candidate !== record
    && candidate.userId === buyer.id
    && candidate.promotionId === record.promotionId
    && candidate.affiliateRewardPaidAt
  ));
  if (previousConversion) {
    record.affiliateRewardBlockedAt = now;
    record.affiliateRewardError = 'This customer conversion already paid an influencer reward.';
    return { paid: false, blocked: true };
  }

  const influencer = (db.users || []).find((candidate) => candidate.id === affiliateUserId);
  if (!influencer || influencer.id === 'platform') {
    record.affiliateRewardBlockedAt = now;
    record.affiliateRewardError = 'The linked influencer account is unavailable.';
    return { paid: false, blocked: true };
  }

  const platform = (db.users || []).find((candidate) => candidate.id === 'platform');
  influencer.mcoins = Number((Number(influencer.mcoins || 0) + amountMcoins).toFixed(2));
  influencer.withdrawableMcoins = Number((Number(influencer.withdrawableMcoins || 0) + amountMcoins).toFixed(2));
  if (platform) platform.mcoins = Number((Number(platform.mcoins || 0) - amountMcoins).toFixed(2));

  const detail = `${amountMcoins.toFixed(2)} Mcoins for ${product?.name || 'subscription'} referral ${record.luckyCode || ''}`.trim();
  addLedger(db, influencer.id, amountMcoins, 'affiliate_reward', detail);
  if (platform) addLedger(db, platform.id, -amountMcoins, 'affiliate_marketing_cost', detail);

  record.affiliateRewardPaidAt = now;
  record.affiliateRewardMcoins = amountMcoins;
  record.affiliateRewardError = null;
  const redemption = (db.promotionRedemptions || [])
    .filter((entry) => entry.promotionId === record.promotionId && entry.userId === buyer.id)
    .sort((left, right) => String(right.createdAt).localeCompare(String(left.createdAt)))[0];
  if (redemption) {
    redemption.affiliateUserId = influencer.id;
    redemption.affiliateRewardMcoins = amountMcoins;
    redemption.affiliateRewardPaidAt = now;
    redemption.subscriptionId = record.subscriptionId;
  }
  return { paid: true, amountMcoins, influencer };
}

module.exports = {
  affiliateRewardStats,
  applyAffiliateReward,
  rewardAmount,
};
