const test = require('node:test');
const assert = require('node:assert/strict');

process.env.ADMIN_EMAILS = 'admin@example.test';
const { subscriptionRules } = require('./server');

test('institution catalog uses exact monthly prices and a 20% annual discount', () => {
  const products = Object.values(subscriptionRules.products).filter((product) => product.audience === 'institution');
  assert.equal(products.length, 6);
  assert.deepEqual(Object.fromEntries(products.map((product) => [product.id, product.price])), {
    'polymath-institution-class-monthly': '300.00',
    'polymath-institution-class-yearly': '2880.00',
    'polymath-institution-cohort-monthly': '2250.00',
    'polymath-institution-cohort-yearly': '21600.00',
    'polymath-institution-school-monthly': '7500.00',
    'polymath-institution-school-yearly': '72000.00',
  });
  products.filter((product) => product.interval === 'YEAR').forEach((product) => {
    assert.equal(product.annualDiscountPercent, 20);
    assert.equal(Number(product.price), Number(product.annualListPrice) * 0.8);
  });
});

test('free, Chill, Musician, and administrator translation rules are exact', () => {
  const now = new Date('2026-02-28T12:00:00.000Z');
  const free = { email: 'free@example.test', pro: false, proStatus: 'INACTIVE' };
  const freeAllowance = subscriptionRules.translationAllowance(free, now);
  assert.equal(freeAllowance.plan, 'free');
  assert.equal(freeAllowance.limit, 0);
  assert.equal(freeAllowance.remaining, 0);
  assert.equal(freeAllowance.overageCostMcoins, 2);
  assert.equal(subscriptionRules.hasMusicianAccess(free), false);

  const chill = {
    email: 'chill@example.test',
    pro: true,
    proStatus: 'ACTIVE',
    subscriptionTier: 'chill',
    subscriptionInterval: 'MONTH',
    subscriptionStartedAt: '2026-01-31T12:00:00.000Z',
  };
  const chillAllowance = subscriptionRules.translationAllowance(chill, now);
  assert.equal(chillAllowance.plan, 'chill');
  assert.equal(chillAllowance.limit, 10);
  assert.equal(chillAllowance.remaining, 10);
  assert.equal(chillAllowance.overageCostMcoins, 0.5);
  assert.equal(chillAllowance.resetAt, '2026-03-31T12:00:00.000Z');
  assert.equal(subscriptionRules.hasMusicianAccess(chill), false);

  const musician = {
    ...chill,
    email: 'musician@example.test',
    subscriptionTier: 'musician',
  };
  const musicianAllowance = subscriptionRules.translationAllowance(musician, now);
  assert.equal(musicianAllowance.limit, 20);
  assert.equal(musicianAllowance.overageCostMcoins, 0.5);
  assert.equal(subscriptionRules.hasMusicianAccess(musician), true);

  const administrator = { email: 'admin@example.test', pro: false, proStatus: 'INACTIVE' };
  const adminAllowance = subscriptionRules.translationAllowance(administrator, now);
  assert.equal(adminAllowance.unlimited, true);
  assert.equal(adminAllowance.remaining, null);
  assert.equal(subscriptionRules.hasMusicianAccess(administrator), true);
});

test('Chill to Musician activation resets day one once and is webhook-idempotent', () => {
  const user = {
    id: 'user-1',
    email: 'member@example.test',
    pro: true,
    proStatus: 'ACTIVE',
    subscriptionTier: 'chill',
    subscriptionInterval: 'YEAR',
    subscriptionStartedAt: '2026-01-10T00:00:00.000Z',
    paypalSubscriptionId: 'old-chill',
    translationUsage: { period: 'old-period', includedUsed: 10 },
  };
  const record = {
    subscriptionId: 'new-musician',
    productId: 'polymath-musician-yearly',
    planId: 'plan-test',
    userId: user.id,
    status: 'APPROVAL_PENDING',
    createdAt: new Date().toISOString(),
  };
  const db = { users: [user], subscriptions: [record] };

  subscriptionRules.applySubscriptionStatus(db, record.subscriptionId, 'ACTIVE', user.id);
  assert.equal(user.subscriptionTier, 'musician');
  assert.equal(user.subscriptionInterval, 'YEAR');
  assert.equal(user.paypalSubscriptionId, 'new-musician');
  assert.equal(user.translationUsage.includedUsed, 0);
  assert.ok(user.subscriptionStartedAt);
  const firstStart = user.subscriptionStartedAt;

  user.translationUsage.includedUsed = 5;
  subscriptionRules.applySubscriptionStatus(db, record.subscriptionId, 'ACTIVE', user.id);
  assert.equal(user.subscriptionStartedAt, firstStart);
  assert.equal(user.translationUsage.includedUsed, 5);
});

test('institution Lucky codes grant Musician access and stop at the seat limit', () => {
  const institution = {
    id: 'institution-class', name: 'Test Class', plan: 'class', seatLimit: 1,
    accessCode: 'CLASS-ABC123', status: 'ACTIVE',
  };
  const db = {
    users: [], institutions: [institution], institutionMemberships: [],
    promotions: [], promotionRedemptions: [], ledger: [],
  };
  const student = { id: 'student-1', email: 'student@example.test', mcoins: 0, pro: false, proStatus: 'INACTIVE' };
  db.users.push(student);
  const result = subscriptionRules.resolveSignupLuckyCode(db, 'class-abc123');
  assert.equal(result.error, '');
  subscriptionRules.applySignupLuckyCode(db, student, result.claim);
  assert.equal(subscriptionRules.activeSubscriptionTier(student), 'musician');
  assert.equal(subscriptionRules.hasMusicianAccess(student), true);
  assert.equal(student.institutionRole, 'member');
  assert.equal(student.institutionAccessCode, '');
  assert.equal(subscriptionRules.institutionSeatCount(db, institution.id), 1);

  const full = subscriptionRules.resolveSignupLuckyCode(db, 'CLASS-ABC123');
  assert.match(full.error, /no seats remaining/i);
});

test('activating an institution subscription creates the owner code and cancellation removes member access', () => {
  const owner = { id: 'owner-1', name: 'Music School', email: 'owner@example.test', pro: false, proStatus: 'INACTIVE' };
  const record = {
    subscriptionId: 'institution-subscription-1', productId: 'polymath-institution-class-monthly',
    planId: 'paypal-plan', userId: owner.id, status: 'APPROVAL_PENDING', createdAt: new Date().toISOString(),
  };
  const db = { users: [owner], subscriptions: [record], institutions: [], institutionMemberships: [] };
  subscriptionRules.applySubscriptionStatus(db, record.subscriptionId, 'ACTIVE', owner.id);
  assert.equal(db.institutions.length, 1);
  assert.match(db.institutions[0].accessCode, /^CLASS-[A-F0-9]{6}$/);
  assert.equal(owner.institutionRole, 'owner');
  assert.equal(owner.institutionSeatLimit, 30);

  const member = { id: 'member-1', institutionId: db.institutions[0].id, institutionStatus: 'ACTIVE' };
  db.users.push(member);
  db.institutionMemberships.push({
    id: 'membership-1', institutionId: db.institutions[0].id, userId: member.id, role: 'member', status: 'ACTIVE',
  });
  subscriptionRules.applySubscriptionStatus(db, record.subscriptionId, 'CANCELLED', owner.id);
  assert.equal(member.institutionStatus, 'INACTIVE');
  assert.equal(subscriptionRules.activeSubscriptionTier(member), 'free');
});

test('promotion codes and Friend IDs can lock a signup subscription discount', () => {
  const subscriptionPromotion = {
    id: 'promo-jan', code: 'JAN50', name: 'January half price', kind: 'subscription_percent',
    value: 50, active: true, maxRedemptions: 0, perUserLimit: 1, minimumAccountAgeDays: 0,
    createdAt: '2026-01-01T00:00:00.000Z',
  };
  const friendPromotion = {
    id: 'promo-friend', code: 'FRIEND50', name: 'Friend half price', kind: 'friend_id_percent',
    value: 50, active: true, maxRedemptions: 0, perUserLimit: 1, minimumAccountAgeDays: 0,
    createdAt: '2026-01-02T00:00:00.000Z',
  };
  const friend = { id: 'friend-1', friendId: 'user_abc12' };
  const user = { id: 'new-1', mcoins: 0 };
  const db = {
    users: [friend, user], institutions: [], institutionMemberships: [],
    promotions: [subscriptionPromotion, friendPromotion], promotionRedemptions: [], ledger: [],
  };

  const codeResult = subscriptionRules.resolveSignupLuckyCode(db, 'jan50');
  assert.equal(codeResult.error, '');
  subscriptionRules.applySignupLuckyCode(db, user, codeResult.claim);
  assert.deepEqual(subscriptionRules.subscriptionPriceForUser(
    subscriptionRules.products['polymath-musician-monthly'], user,
  ), { price: '7.50', discountPercent: 50, luckyCode: 'JAN50' });

  const referred = { id: 'new-2', mcoins: 0 };
  db.users.push(referred);
  const friendResult = subscriptionRules.resolveSignupLuckyCode(db, 'USER_ABC12');
  assert.equal(friendResult.error, '');
  subscriptionRules.applySignupLuckyCode(db, referred, friendResult.claim);
  assert.equal(referred.luckyCodeClaim.friendId, 'user_abc12');
  assert.equal(referred.luckyCodeClaim.value, 50);
  assert.equal(db.promotionRedemptions.length, 2);
});

test('ready-sheet uploads include two monthly attempts, then cost 0.5 Mcoin, while paid plans are unlimited', () => {
  const january = new Date('2026-01-15T12:00:00.000Z');
  const february = new Date('2026-02-01T00:00:00.000Z');
  const member = {
    id: 'member-1',
    email: 'member@example.test',
    mcoins: 0,
    pro: false,
    proStatus: 'INACTIVE',
  };
  const db = { ledger: [] };
  assert.deepEqual(subscriptionRules.readySheetAllowance(member, january), {
    unlimited: false,
    limit: 2,
    used: 0,
    remaining: 2,
    resetAt: '2026-02-01T00:00:00.000Z',
    overageCostMcoins: 0.5,
  });
  assert.deepEqual(subscriptionRules.chargeReadySheetUpload(db, member, 'song.mid', january), {
    costMcoins: 0,
    paymentMethod: 'free_attempt',
  });
  assert.equal(subscriptionRules.readySheetAllowance(member, january).remaining, 1);
  assert.deepEqual(subscriptionRules.chargeReadySheetUpload(db, member, 'second.json', january), {
    costMcoins: 0,
    paymentMethod: 'free_attempt',
  });
  assert.equal(subscriptionRules.readySheetAllowance(member, january).remaining, 0);
  assert.equal(subscriptionRules.chargeReadySheetUpload(db, member, 'third.mid', january), null);

  member.mcoins = 1;
  assert.deepEqual(subscriptionRules.chargeReadySheetUpload(db, member, 'third.mid', january), {
    costMcoins: 0.5,
    paymentMethod: 'mcoins',
  });
  assert.equal(member.mcoins, 0.5);
  assert.equal(db.ledger.at(-1).amount, -0.5);
  assert.equal(subscriptionRules.readySheetAllowance(member, february).remaining, 2);

  const subscriber = {
    id: 'subscriber-1',
    email: 'subscriber@example.test',
    mcoins: 0,
    pro: true,
    proStatus: 'ACTIVE',
    subscriptionTier: 'chill',
  };
  assert.equal(subscriptionRules.readySheetAllowance(subscriber, january).unlimited, true);
  assert.deepEqual(subscriptionRules.chargeReadySheetUpload(db, subscriber, 'subscriber.mid', january), {
    costMcoins: 0,
    paymentMethod: 'unlimited',
  });
  assert.equal(subscriber.mcoins, 0);

  const administrator = {
    id: 'admin-1',
    email: 'admin@example.test',
    mcoins: 0,
    pro: false,
    proStatus: 'INACTIVE',
  };
  assert.equal(subscriptionRules.readySheetAllowance(administrator, january).unlimited, true);
  assert.deepEqual(subscriptionRules.chargeReadySheetUpload(db, administrator, 'admin.mid', january), {
    costMcoins: 0,
    paymentMethod: 'unlimited',
  });
  assert.equal(administrator.mcoins, 0);
});
