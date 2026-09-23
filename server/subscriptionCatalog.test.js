'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  activeEntitlements,
  applyCustomProductAccess,
  createPlan,
  ensureSubscriptionCatalog,
  listPublicCatalog,
  resolveProduct,
  updatePlan,
} = require('./subscriptionCatalog');

test('administrator-created plans remain drafts until checkout is configured', () => {
  const db = { subscriptionCategories: [], subscriptionPlans: [], subscriptionCatalogEvents: [] };
  ensureSubscriptionCatalog(db);
  const plan = createPlan(db, {
    categoryId: 'category-create-music',
    name: 'Creator Studio',
    price: '12.99',
    interval: 'MONTH',
    features: ['Cloud projects', 'AI songwriting guidance'],
    entitlements: ['create_music.projects', 'create_music.ai_guidance'],
  }, 'admin-1');
  assert.equal(plan.status, 'draft');
  assert.equal(listPublicCatalog(db, {}).products.length, 0);
  assert.throws(
    () => updatePlan(db, plan.id, { revision: 1, status: 'published' }, 'admin-1'),
    /PayPal plan ID/,
  );

  const published = updatePlan(db, plan.id, {
    revision: 1,
    paypalPlanId: 'P-TEST-CREATOR',
    status: 'published',
  }, 'admin-1');
  assert.equal(published.status, 'published');
  const product = resolveProduct(db, {}, plan.id);
  assert.equal(product.price, '12.99');
  assert.equal(product.checkoutConfigured, true);
  assert.equal(listPublicCatalog(db, {}).products.length, 0);
  assert.equal(listPublicCatalog(db, {}).categories.some((item) => item.slug === 'create-music'), false);
});

test('custom access is additive and sold pricing cannot mutate', () => {
  const db = { subscriptionCategories: [], subscriptionPlans: [], subscriptionCatalogEvents: [] };
  ensureSubscriptionCatalog(db);
  const plan = createPlan(db, {
    categoryId: 'category-create-music',
    name: 'Creator Studio',
    unitAmountCents: 1299,
    interval: 'MONTH',
    paypalPlanId: 'P-TEST-CREATOR',
    status: 'published',
    entitlements: ['create_music.projects', 'create_music.exports'],
  }, 'admin-1');
  const product = resolveProduct(db, {}, plan.id);
  const user = { subscriptionAccess: {} };
  applyCustomProductAccess(user, product, 'I-SUB-1', 'ACTIVE');
  assert.deepEqual(activeEntitlements(user).sort(), ['create_music.exports', 'create_music.projects']);
  assert.throws(
    () => updatePlan(db, plan.id, { revision: 1, unitAmountCents: 1499 }, 'admin-1', [{ productId: plan.id }]),
    /keeps its original price/,
  );
});
