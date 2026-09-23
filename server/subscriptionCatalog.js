'use strict';

const crypto = require('crypto');

const ALLOWED_ENTITLEMENTS = Object.freeze([
  { key: 'create_music.projects', label: 'Create and save music projects' },
  { key: 'create_music.ai_guidance', label: 'AI brief and lyric guidance' },
  { key: 'create_music.arrangements', label: 'Generate playable arrangements' },
  { key: 'create_music.exports', label: 'Download MIDI, JSON, and lyrics' },
  { key: 'create_music.guide_voice', label: 'Temporary guide voice' },
  { key: 'learn', label: 'Learn mode' },
  { key: 'band', label: 'Band collaboration' },
  { key: 'ready_sheets.unlimited', label: 'Unlimited ready-to-play uploads' },
]);

const ALLOWED_ENTITLEMENT_KEYS = new Set(ALLOWED_ENTITLEMENTS.map((item) => item.key));
const HIDDEN_PUBLIC_CATEGORY_SLUGS = new Set(['create-music']);

const DEFAULT_CREATOR_CATEGORY = Object.freeze({
  id: 'category-create-music',
  slug: 'create-music',
  name: 'Create Music',
  description: 'Guided songwriting, playable arrangements, karaoke rehearsal, and exports.',
  audience: 'creator',
  sortOrder: 30,
  status: 'published',
  system: true,
  revision: 1,
  createdAt: '2026-09-07T00:00:00.000Z',
  updatedAt: '2026-09-07T00:00:00.000Z',
});

const BASE_FEATURES = Object.freeze({
  chill: [
    'Piano, guitar, and supported instrument studios',
    'Unlimited JSON and MIDI ready-to-play uploads',
    '10 shared PDF or audio translations every month',
    'Extra translations for 0.5 Mcoin each',
  ],
  musician: [
    'Everything included in Chill',
    '20 shared PDF or audio translations every month',
    'Extra translations for 0.5 Mcoin each',
  ],
});

function createError(message, status = 400) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function clean(value, maximum = 200) {
  return String(value || '').trim().slice(0, maximum);
}

function slugify(value) {
  return clean(value, 80)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48);
}

function normalizeFeatures(value) {
  const rows = Array.isArray(value) ? value : String(value || '').split(/\r?\n/);
  return [...new Set(rows.map((item) => clean(item, 140)).filter(Boolean))].slice(0, 16);
}

function normalizeEntitlements(value) {
  const rows = Array.isArray(value) ? value : [];
  return [...new Set(rows.map((item) => clean(item, 80)).filter((item) => ALLOWED_ENTITLEMENT_KEYS.has(item)))];
}

function normalizeMoneyCents(value, fallback = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(0, Math.min(100_000_000, Math.round(number)));
}

function ensureSubscriptionCatalog(db) {
  if (!Array.isArray(db.subscriptionCategories)) db.subscriptionCategories = [];
  if (!Array.isArray(db.subscriptionPlans)) db.subscriptionPlans = [];
  if (!Array.isArray(db.subscriptionCatalogEvents)) db.subscriptionCatalogEvents = [];
  if (!db.subscriptionCategories.some((item) => item.id === DEFAULT_CREATOR_CATEGORY.id)) {
    db.subscriptionCategories.push({ ...DEFAULT_CREATOR_CATEGORY });
  }
  return db;
}

function categoryById(db, categoryId) {
  ensureSubscriptionCatalog(db);
  return db.subscriptionCategories.find((item) => item.id === categoryId) || null;
}

function publicCategory(category) {
  return {
    id: category.id,
    slug: category.slug,
    name: category.name,
    description: category.description || '',
    audience: category.audience || 'individual',
    sortOrder: Number(category.sortOrder || 0),
  };
}

function publicCustomProduct(plan, category, { admin = false } = {}) {
  const product = {
    id: plan.id,
    name: plan.name,
    description: plan.description || '',
    badge: plan.badge || '',
    price: (Number(plan.unitAmountCents || 0) / 100).toFixed(2),
    unitAmountCents: Number(plan.unitAmountCents || 0),
    currency: plan.currency || 'USD',
    kind: 'subscription',
    recurring: true,
    interval: plan.interval || 'MONTH',
    intervalCount: Number(plan.intervalCount || 1),
    tier: 'custom',
    audience: category?.audience || 'creator',
    categoryId: category?.id || plan.categoryId,
    categorySlug: category?.slug || '',
    categoryName: category?.name || 'Subscription',
    features: normalizeFeatures(plan.features),
    entitlements: normalizeEntitlements(plan.entitlements),
    sortOrder: Number(plan.sortOrder || 0),
    status: plan.status || 'draft',
    revision: Number(plan.revision || 1),
    checkoutConfigured: Boolean(plan.paypalPlanId),
  };
  if (admin) {
    product.paypalProductId = plan.paypalProductId || '';
    product.paypalPlanId = plan.paypalPlanId || '';
    product.createdAt = plan.createdAt || null;
    product.updatedAt = plan.updatedAt || null;
    product.archivedAt = plan.archivedAt || null;
  }
  return product;
}

function decorateBaseProduct(product) {
  if (product.kind !== 'subscription') {
    return {
      ...product,
      categoryId: 'wallet',
      categorySlug: 'wallet',
      categoryName: 'Wallet',
      features: [],
      entitlements: [],
      checkoutConfigured: true,
    };
  }
  const institution = Boolean(product.institutionTier);
  const tier = product.tier || 'musician';
  return {
    ...product,
    categoryId: institution ? 'institution' : 'individual',
    categorySlug: institution ? 'institution' : 'individual',
    categoryName: institution ? 'Institution' : 'Individual',
    features: institution
      ? [
          `${Number(product.seats || 0).toLocaleString()} individual accounts`,
          'Every member receives Musician abilities',
          'Supported instrument studios and monthly translations',
          'Private access code with seat controls',
        ]
      : BASE_FEATURES[tier] || [],
    entitlements: tier === 'musician'
      ? ['learn', 'band', 'ready_sheets.unlimited']
      : tier === 'chill'
        ? ['ready_sheets.unlimited']
        : [],
    checkoutConfigured: true,
  };
}

function listPublicCatalog(db, baseProducts) {
  ensureSubscriptionCatalog(db);
  const categories = [
    { id: 'individual', slug: 'individual', name: 'Individual', description: 'For one musician.', audience: 'individual', sortOrder: 10 },
    { id: 'institution', slug: 'institution', name: 'Institution', description: 'For classes, cohorts, and schools.', audience: 'institution', sortOrder: 20 },
    ...db.subscriptionCategories
      .filter((item) => item.status === 'published' && !HIDDEN_PUBLIC_CATEGORY_SLUGS.has(item.slug))
      .map(publicCategory),
  ].sort((left, right) => left.sortOrder - right.sortOrder);
  const customProducts = db.subscriptionPlans
    .filter((plan) => {
      if (plan.status !== 'published') return false;
      const category = categoryById(db, plan.categoryId);
      return !HIDDEN_PUBLIC_CATEGORY_SLUGS.has(category?.slug);
    })
    .map((plan) => publicCustomProduct(plan, categoryById(db, plan.categoryId)))
    .sort((left, right) => left.sortOrder - right.sortOrder || left.name.localeCompare(right.name));
  const products = [
    ...Object.values(baseProducts).filter((product) => !product.legacy).map(decorateBaseProduct),
    ...customProducts,
  ];
  return { categories, products };
}

function listAdminCatalog(db) {
  ensureSubscriptionCatalog(db);
  return {
    categories: db.subscriptionCategories
      .slice()
      .sort((left, right) => Number(left.sortOrder || 0) - Number(right.sortOrder || 0)),
    plans: db.subscriptionPlans
      .map((plan) => publicCustomProduct(plan, categoryById(db, plan.categoryId), { admin: true }))
      .sort((left, right) => left.sortOrder - right.sortOrder || left.name.localeCompare(right.name)),
    entitlementOptions: ALLOWED_ENTITLEMENTS,
  };
}

function resolveProduct(db, baseProducts, productId) {
  const id = clean(productId, 100);
  if (baseProducts[id]) return decorateBaseProduct(baseProducts[id]);
  ensureSubscriptionCatalog(db);
  const plan = db.subscriptionPlans.find((item) => item.id === id);
  return plan ? publicCustomProduct(plan, categoryById(db, plan.categoryId), { admin: true }) : null;
}

function createCategory(db, input, actorId) {
  ensureSubscriptionCatalog(db);
  const name = clean(input.name, 80);
  const slug = slugify(input.slug || name);
  if (!name || slug.length < 2) throw createError('Give the subscription category a clear name.');
  if (db.subscriptionCategories.some((item) => item.slug === slug)) {
    throw createError('That subscription category slug already exists.', 409);
  }
  const now = new Date().toISOString();
  const category = {
    id: `category-${crypto.randomUUID()}`,
    slug,
    name,
    description: clean(input.description, 240),
    audience: clean(input.audience || 'creator', 40).toLowerCase(),
    sortOrder: Math.max(0, Math.min(1000, Math.round(Number(input.sortOrder) || 30))),
    status: input.status === 'published' ? 'published' : 'draft',
    system: false,
    revision: 1,
    createdBy: actorId,
    createdAt: now,
    updatedAt: now,
  };
  db.subscriptionCategories.push(category);
  recordCatalogEvent(db, actorId, 'category_created', category.id, { name, slug });
  return category;
}

function updateCategory(db, categoryId, input, actorId) {
  const category = categoryById(db, categoryId);
  if (!category) throw createError('Subscription category not found.', 404);
  const expectedRevision = Number(input.revision || category.revision);
  if (expectedRevision !== Number(category.revision || 1)) throw createError('This category changed in another session. Reload and try again.', 409);
  if (input.name !== undefined) {
    const name = clean(input.name, 80);
    if (!name) throw createError('Category name cannot be empty.');
    category.name = name;
  }
  if (input.description !== undefined) category.description = clean(input.description, 240);
  if (input.audience !== undefined) category.audience = clean(input.audience, 40).toLowerCase() || 'creator';
  if (input.sortOrder !== undefined) category.sortOrder = Math.max(0, Math.min(1000, Math.round(Number(input.sortOrder) || 0)));
  if (input.status !== undefined) {
    if (!['draft', 'published', 'archived'].includes(input.status)) throw createError('Choose draft, published, or archived.');
    category.status = input.status;
  }
  category.revision = Number(category.revision || 1) + 1;
  category.updatedBy = actorId;
  category.updatedAt = new Date().toISOString();
  recordCatalogEvent(db, actorId, 'category_updated', category.id, { revision: category.revision });
  return category;
}

function createPlan(db, input, actorId) {
  ensureSubscriptionCatalog(db);
  const category = categoryById(db, input.categoryId);
  if (!category || category.status === 'archived') throw createError('Choose an active subscription category.');
  const name = clean(input.name, 80);
  const slug = slugify(input.slug || name);
  if (!name || slug.length < 2) throw createError('Give the subscription plan a clear name.');
  if (db.subscriptionPlans.some((item) => item.categoryId === category.id && item.slug === slug && item.status !== 'archived')) {
    throw createError('That plan slug already exists in this category.', 409);
  }
  const interval = clean(input.interval || 'MONTH', 10).toUpperCase();
  if (!['MONTH', 'YEAR'].includes(interval)) throw createError('Billing interval must be monthly or yearly.');
  const unitAmountCents = input.unitAmountCents !== undefined
    ? normalizeMoneyCents(input.unitAmountCents)
    : normalizeMoneyCents(Number(input.price || 0) * 100);
  const entitlements = normalizeEntitlements(input.entitlements);
  if (!entitlements.length) throw createError('Choose at least one server-enforced entitlement.');
  const paypalPlanId = clean(input.paypalPlanId, 120);
  const status = input.status === 'published' ? 'published' : 'draft';
  if (status === 'published' && unitAmountCents > 0 && !paypalPlanId) {
    throw createError('Add the matching PayPal plan ID before publishing a paid plan.');
  }
  const now = new Date().toISOString();
  const plan = {
    id: `plan-${crypto.randomUUID()}`,
    categoryId: category.id,
    slug,
    name,
    description: clean(input.description, 260),
    badge: clean(input.badge, 40),
    features: normalizeFeatures(input.features),
    entitlements,
    unitAmountCents,
    currency: clean(input.currency || 'USD', 3).toUpperCase(),
    interval,
    intervalCount: 1,
    paypalProductId: clean(input.paypalProductId, 120),
    paypalPlanId,
    status,
    sortOrder: Math.max(0, Math.min(1000, Math.round(Number(input.sortOrder) || 0))),
    revision: 1,
    createdBy: actorId,
    createdAt: now,
    updatedAt: now,
  };
  db.subscriptionPlans.push(plan);
  recordCatalogEvent(db, actorId, 'plan_created', plan.id, { name, categoryId: category.id });
  return plan;
}

function updatePlan(db, planId, input, actorId, subscriptions = []) {
  ensureSubscriptionCatalog(db);
  const plan = db.subscriptionPlans.find((item) => item.id === planId);
  if (!plan) throw createError('Subscription plan not found.', 404);
  const expectedRevision = Number(input.revision || plan.revision);
  if (expectedRevision !== Number(plan.revision || 1)) throw createError('This plan changed in another session. Reload and try again.', 409);
  const sold = subscriptions.some((item) => item.productId === plan.id);
  const immutableChanges = [
    ['unitAmountCents', normalizeMoneyCents(input.unitAmountCents, plan.unitAmountCents)],
    ['currency', clean(input.currency || plan.currency, 3).toUpperCase()],
    ['interval', clean(input.interval || plan.interval, 10).toUpperCase()],
    ['paypalPlanId', clean(input.paypalPlanId ?? plan.paypalPlanId, 120)],
  ];
  if (sold && immutableChanges.some(([key, value]) => value !== plan[key])) {
    throw createError('A purchased plan keeps its original price. Archive it and create a new version instead.', 409);
  }
  if (input.name !== undefined) plan.name = clean(input.name, 80) || plan.name;
  if (input.description !== undefined) plan.description = clean(input.description, 260);
  if (input.badge !== undefined) plan.badge = clean(input.badge, 40);
  if (input.features !== undefined) plan.features = normalizeFeatures(input.features);
  if (input.entitlements !== undefined) {
    const entitlements = normalizeEntitlements(input.entitlements);
    if (!entitlements.length) throw createError('Choose at least one server-enforced entitlement.');
    plan.entitlements = entitlements;
  }
  if (!sold) immutableChanges.forEach(([key, value]) => { plan[key] = value; });
  if (input.paypalProductId !== undefined && !sold) plan.paypalProductId = clean(input.paypalProductId, 120);
  if (input.sortOrder !== undefined) plan.sortOrder = Math.max(0, Math.min(1000, Math.round(Number(input.sortOrder) || 0)));
  if (input.status !== undefined) {
    if (!['draft', 'published', 'archived'].includes(input.status)) throw createError('Choose draft, published, or archived.');
    if (input.status === 'published' && plan.unitAmountCents > 0 && !plan.paypalPlanId) {
      throw createError('Add the matching PayPal plan ID before publishing a paid plan.');
    }
    plan.status = input.status;
    if (input.status === 'archived') plan.archivedAt = new Date().toISOString();
  }
  plan.revision = Number(plan.revision || 1) + 1;
  plan.updatedBy = actorId;
  plan.updatedAt = new Date().toISOString();
  recordCatalogEvent(db, actorId, 'plan_updated', plan.id, { revision: plan.revision, status: plan.status });
  return plan;
}

function recordCatalogEvent(db, actorId, action, targetId, details = {}) {
  ensureSubscriptionCatalog(db);
  db.subscriptionCatalogEvents.push({
    id: `catalog-event-${crypto.randomUUID()}`,
    actorId,
    action,
    targetId,
    details,
    createdAt: new Date().toISOString(),
  });
  db.subscriptionCatalogEvents = db.subscriptionCatalogEvents.slice(-5000);
}

function activeEntitlements(user) {
  const access = user?.subscriptionAccess && typeof user.subscriptionAccess === 'object'
    ? Object.values(user.subscriptionAccess)
    : [];
  return [...new Set(access
    .filter((item) => String(item?.status || '').toUpperCase() === 'ACTIVE')
    .flatMap((item) => normalizeEntitlements(item.entitlements)))];
}

function hasEntitlement(user, entitlement) {
  return activeEntitlements(user).includes(entitlement);
}

function applyCustomProductAccess(user, product, subscriptionId, status) {
  if (!user.subscriptionAccess || typeof user.subscriptionAccess !== 'object') user.subscriptionAccess = {};
  const active = String(status || '').toUpperCase() === 'ACTIVE';
  if (!active) {
    delete user.subscriptionAccess[product.id];
    return;
  }
  const previous = user.subscriptionAccess[product.id];
  user.subscriptionAccess[product.id] = {
    productId: product.id,
    subscriptionId,
    status: 'ACTIVE',
    name: product.name,
    categorySlug: product.categorySlug || '',
    interval: product.interval,
    entitlements: normalizeEntitlements(product.entitlements),
    startedAt: previous?.startedAt || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

module.exports = {
  ALLOWED_ENTITLEMENTS,
  DEFAULT_CREATOR_CATEGORY,
  activeEntitlements,
  applyCustomProductAccess,
  createCategory,
  createPlan,
  decorateBaseProduct,
  ensureSubscriptionCatalog,
  hasEntitlement,
  listAdminCatalog,
  listPublicCatalog,
  normalizeEntitlements,
  resolveProduct,
  updateCategory,
  updatePlan,
};
