const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const environment = String(process.env.PAYPAL_ENV || 'sandbox').trim().toLowerCase();
const apiBase = environment === 'live' ? 'https://api-m.paypal.com' : 'https://api-m.sandbox.paypal.com';
const envPath = path.join(__dirname, '.env');
const productName = 'Polymath Musician Memberships v1';

const plans = [
  ['PAYPAL_CHILL_MONTHLY_PLAN_ID', 'Chill Monthly', '7.99', 'MONTH'],
  ['PAYPAL_CHILL_YEARLY_PLAN_ID', 'Chill Yearly', '49.99', 'YEAR'],
  ['PAYPAL_MUSICIAN_MONTHLY_PLAN_ID', 'Musician Monthly', '14.99', 'MONTH'],
  ['PAYPAL_MUSICIAN_YEARLY_PLAN_ID', 'Musician Yearly', '93.99', 'YEAR'],
  ['PAYPAL_INSTITUTION_CLASS_MONTHLY_PLAN_ID', 'Institution Class Monthly', '300.00', 'MONTH'],
  ['PAYPAL_INSTITUTION_CLASS_YEARLY_PLAN_ID', 'Institution Class Yearly', '2880.00', 'YEAR'],
  ['PAYPAL_INSTITUTION_COHORT_MONTHLY_PLAN_ID', 'Institution Cohort Monthly', '2250.00', 'MONTH'],
  ['PAYPAL_INSTITUTION_COHORT_YEARLY_PLAN_ID', 'Institution Cohort Yearly', '21600.00', 'YEAR'],
  ['PAYPAL_INSTITUTION_SCHOOL_MONTHLY_PLAN_ID', 'Institution School Monthly', '7500.00', 'MONTH'],
  ['PAYPAL_INSTITUTION_SCHOOL_YEARLY_PLAN_ID', 'Institution School Yearly', '72000.00', 'YEAR'],
  ['PAYPAL_CHILL_TO_MUSICIAN_MONTHLY_PLAN_ID', 'Chill to Musician Monthly Upgrade', '14.99', 'MONTH', '7.00'],
  ['PAYPAL_CHILL_TO_MUSICIAN_YEARLY_PLAN_ID', 'Chill to Musician Yearly Upgrade', '93.99', 'YEAR', '44.00'],
].map(([envKey, label, price, interval, setupFee = '0.00']) => ({
  envKey,
  label,
  price,
  interval,
  setupFee,
  upgrade: Number(setupFee) > 0,
  name: `Polymath ${label} | USD ${price} | v1`,
}));

async function accessToken() {
  const clientId = String(process.env.PAYPAL_CLIENT_ID || '').trim();
  const secret = String(process.env.PAYPAL_SECRET_KEY || '').trim();
  if (!clientId || !secret) throw new Error('PAYPAL_CLIENT_ID and PAYPAL_SECRET_KEY are required.');
  const response = await fetch(`${apiBase}/v1/oauth2/token`, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${Buffer.from(`${clientId}:${secret}`).toString('base64')}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: 'grant_type=client_credentials',
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error_description || data.error || 'PayPal authentication failed.');
  return data.access_token;
}

async function paypal(token, pathname, options = {}) {
  const response = await fetch(`${apiBase}${pathname}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Prefer: 'return=representation',
      'PayPal-Request-Id': crypto.randomUUID(),
      ...(options.headers || {}),
    },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.details?.map((item) => item.description || item.issue).filter(Boolean).join('; ');
    const error = new Error(detail || data.message || data.error_description || `PayPal returned HTTP ${response.status}.`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function planBody(definition, productId) {
  const regular = {
    frequency: { interval_unit: definition.interval, interval_count: 1 },
    tenure_type: 'REGULAR',
    sequence: definition.upgrade ? 2 : 1,
    total_cycles: 0,
    pricing_scheme: { fixed_price: { value: definition.price, currency_code: 'USD' } },
  };
  const billingCycles = definition.upgrade ? [{
    frequency: { interval_unit: definition.interval, interval_count: 1 },
    tenure_type: 'TRIAL',
    sequence: 1,
    total_cycles: 1,
    pricing_scheme: { fixed_price: { value: '0.00', currency_code: 'USD' } },
  }, regular] : [regular];
  return {
    product_id: productId,
    name: definition.name,
    description: definition.upgrade
      ? `${definition.label}: charge the remaining balance now, then begin regular billing after one ${definition.interval.toLowerCase()}.`
      : `${definition.label} recurring access for Polymath Musician.`,
    status: 'ACTIVE',
    billing_cycles: billingCycles,
    payment_preferences: {
      auto_bill_outstanding: true,
      setup_fee: { value: definition.setupFee, currency_code: 'USD' },
      setup_fee_failure_action: 'CANCEL',
      payment_failure_threshold: 3,
    },
  };
}

function moneyMatches(value, expected) {
  return value?.currency_code === 'USD' && Number(value?.value) === Number(expected);
}

function planMatches(data, definition) {
  const regular = data.billing_cycles?.find((cycle) => cycle.tenure_type === 'REGULAR');
  if (data.status !== 'ACTIVE'
    || regular?.frequency?.interval_unit !== definition.interval
    || regular?.frequency?.interval_count !== 1
    || !moneyMatches(regular?.pricing_scheme?.fixed_price, definition.price)) return false;

  if (!moneyMatches(data.payment_preferences?.setup_fee, definition.setupFee)) return false;
  const trial = data.billing_cycles?.find((cycle) => cycle.tenure_type === 'TRIAL');
  if (!definition.upgrade) return !trial;
  return trial?.total_cycles === 1
    && trial?.frequency?.interval_unit === definition.interval
    && trial?.frequency?.interval_count === 1;
}

async function findOrCreateProduct(token) {
  const list = await paypal(token, '/v1/catalogs/products?page_size=20&page=1&total_required=true', { method: 'GET' });
  const existing = list.products?.find((product) => product.name === productName);
  if (existing) return existing;
  return paypal(token, '/v1/catalogs/products', {
    method: 'POST',
    body: JSON.stringify({
      name: productName,
      description: 'Recurring individual and institution access to Polymath Musician.',
      type: 'SERVICE',
      category: 'SOFTWARE',
      home_url: 'https://polymathmusician67.com',
    }),
  });
}

async function planDetails(token, id) {
  return paypal(token, `/v1/billing/plans/${encodeURIComponent(id)}`, { method: 'GET' });
}

async function ensureActive(token, data) {
  if (data.status === 'CREATED') {
    await paypal(token, `/v1/billing/plans/${encodeURIComponent(data.id)}/activate`, { method: 'POST' });
    return planDetails(token, data.id);
  }
  return data;
}

async function findReusablePlan(token, productId, definition) {
  const list = await paypal(token, `/v1/billing/plans?product_id=${encodeURIComponent(productId)}&page_size=20&page=1&total_required=true`, { method: 'GET' });
  const candidates = (list.plans || []).filter((plan) => plan.name === definition.name);
  for (const candidate of candidates) {
    try {
      const data = await ensureActive(token, await planDetails(token, candidate.id));
      if (planMatches(data, definition)) return data;
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  return null;
}

async function createPlan(token, productId, definition) {
  let data = await paypal(token, '/v1/billing/plans', {
    method: 'POST',
    body: JSON.stringify(planBody(definition, productId)),
  });
  data = await ensureActive(token, data);
  if (!planMatches(data, definition)) data = await planDetails(token, data.id);
  if (!planMatches(data, definition)) {
    throw new Error(`PayPal created ${definition.label}, but its settings did not pass verification.`);
  }
  return data;
}

function updateEnv(values) {
  if (!fs.existsSync(envPath)) throw new Error(`Missing ${envPath}.`);
  let content = fs.readFileSync(envPath, 'utf8').replace(/\r\n/g, '\n').replace(/\s*$/, '\n');
  for (const [key, value] of Object.entries(values)) {
    const line = `${key}=${value}`;
    const pattern = new RegExp(`^${key}=.*$`, 'm');
    content = pattern.test(content) ? content.replace(pattern, line) : `${content}${line}\n`;
  }
  fs.writeFileSync(envPath, content, 'utf8');
}

async function main() {
  console.log(`Provisioning ${plans.length} ${environment.toUpperCase()} PayPal subscription plans...`);
  const token = await accessToken();
  const product = await findOrCreateProduct(token);
  const values = {};
  const replacedPlanIds = [];

  for (const definition of plans) {
    let data = null;
    let action = 'verified existing';
    const configuredId = String(process.env[definition.envKey] || '').trim();
    if (configuredId) {
      try {
        const configured = await ensureActive(token, await planDetails(token, configuredId));
        if (planMatches(configured, definition)) {
          data = configured;
        } else if (configured.product_id === product.id && configured.status === 'ACTIVE') {
          replacedPlanIds.push(configured.id);
        }
      } catch (error) {
        if (error.status !== 404) throw error;
      }
    }
    if (!data) {
      data = await findReusablePlan(token, product.id, definition);
      action = data ? 'reused' : 'created';
    }
    if (!data) data = await createPlan(token, product.id, definition);
    values[definition.envKey] = data.id;
    console.log(`${definition.label}: ${action}`);
  }

  for (const id of new Set(replacedPlanIds)) {
    await paypal(token, `/v1/billing/plans/${encodeURIComponent(id)}/deactivate`, { method: 'POST' });
  }
  updateEnv(values);
  if (replacedPlanIds.length) console.log(`Retired ${new Set(replacedPlanIds).size} replaced plans.`);
  console.log(`Verified ${plans.length} plans and updated server/.env. No credentials or plan IDs were printed.`);
}

main().catch((error) => {
  console.error(`PayPal provisioning failed: ${error.message}`);
  process.exitCode = 1;
});
