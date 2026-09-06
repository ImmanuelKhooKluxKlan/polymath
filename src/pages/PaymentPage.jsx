import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';
import { trackProductEvent } from '../services/productAnalytics.js';

const FALLBACK_PRODUCTS = [
  { id: 'polymath-chill-monthly', name: 'Chill', price: '7.99', currency: 'USD', kind: 'subscription', interval: 'MONTH', tier: 'chill', translations: 10 },
  { id: 'polymath-chill-yearly', name: 'Chill', price: '49.99', currency: 'USD', kind: 'subscription', interval: 'YEAR', tier: 'chill', translations: 10 },
  { id: 'polymath-musician-monthly', name: 'Musician', price: '14.99', currency: 'USD', kind: 'subscription', interval: 'MONTH', tier: 'musician', translations: 20 },
  { id: 'polymath-musician-yearly', name: 'Musician', price: '93.99', currency: 'USD', kind: 'subscription', interval: 'YEAR', tier: 'musician', translations: 20 },
  { id: 'polymath-institution-class-monthly', name: 'Class', price: '300.00', currency: 'USD', kind: 'subscription', interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'class', seats: 30 },
  { id: 'polymath-institution-class-yearly', name: 'Class', price: '2880.00', annualListPrice: '3600.00', annualDiscountPercent: 20, currency: 'USD', kind: 'subscription', interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'class', seats: 30 },
  { id: 'polymath-institution-cohort-monthly', name: 'Cohort', price: '2250.00', currency: 'USD', kind: 'subscription', interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'cohort', seats: 300 },
  { id: 'polymath-institution-cohort-yearly', name: 'Cohort', price: '21600.00', annualListPrice: '27000.00', annualDiscountPercent: 20, currency: 'USD', kind: 'subscription', interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'cohort', seats: 300 },
  { id: 'polymath-institution-school-monthly', name: 'School', price: '7500.00', currency: 'USD', kind: 'subscription', interval: 'MONTH', tier: 'musician', audience: 'institution', institutionTier: 'school', seats: 1000 },
  { id: 'polymath-institution-school-yearly', name: 'School', price: '72000.00', annualListPrice: '90000.00', annualDiscountPercent: 20, currency: 'USD', kind: 'subscription', interval: 'YEAR', tier: 'musician', audience: 'institution', institutionTier: 'school', seats: 1000 },
  { id: 'mcoins-50', name: '50 Mcoins', price: '50.00', currency: 'USD', kind: 'mcoins', mcoins: 50 },
  { id: 'mcoins-100', name: '100 Mcoins', price: '100.00', currency: 'USD', kind: 'mcoins', mcoins: 100 },
  { id: 'mcoins-300', name: '300 Mcoins', price: '300.00', currency: 'USD', kind: 'mcoins', mcoins: 300 },
];

const FEATURES = {
  chill: [
    'Everything in the Regular studio',
    'Unlimited JSON and MIDI ready-to-play uploads',
    '10 shared PDF or audio translations every month',
    'Extra translations for 0.5 Mcoin each',
  ],
  musician: [
    'Everything included in Chill',
    '20 shared PDF or audio translations every month',
    'Full Learn mode across supported instruments',
    'Band creation, joining, rehearsal, and collaboration',
    'Extra translations for 0.5 Mcoin each',
  ],
};

function periodLabel(product) {
  return product?.interval === 'YEAR' ? 'year' : 'month';
}

function usdPrice(value) {
  return Number(value || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function InstitutionPlans({ products, billing, setBilling, user, busy, onChoose }) {
  return (
    <>
      <div className='billing-switch segmented-control' role='group' aria-label='Institution billing period'>
        <button type='button' className={billing === 'MONTH' ? 'active' : ''} onClick={() => setBilling('MONTH')}>Monthly</button>
        <button type='button' className={billing === 'YEAR' ? 'active' : ''} onClick={() => setBilling('YEAR')}>Yearly - 20% off</button>
      </div>
      <div className='institution-plan-grid'>
        {['class', 'cohort', 'school'].map((tier) => {
          const product = products.find((item) => item.institutionTier === tier && item.interval === billing);
          const current = user?.institution?.role === 'owner'
            && user.institution.status === 'ACTIVE'
            && user.institution.plan === tier
            && user.subscriptionInterval === billing;
          return (
            <article key={tier} className={`subscription-plan-card institution-plan-card ${tier}`}>
              <header>
                <div><p className='eyebrow'>Up to {product?.seats || 0} students</p><h2>{product?.name || tier}</h2></div>
                <span className='plan-badge'>Musician for all</span>
              </header>
              {billing === 'YEAR' && product?.annualListPrice && (
                <div className='annual-saving'><s>${usdPrice(product.annualListPrice)}</s><strong>20% off</strong></div>
              )}
              <div className='subscription-price institution-price'>
                <strong>${usdPrice(product?.price)}</strong><span>USD / {periodLabel(product)}</span>
              </div>
              <ul>
                <li>{product?.seats || 0} individual student accounts</li>
                <li>Every member receives full Musician abilities</li>
                <li>Learn, Band, and monthly Musician translations</li>
                <li>One private access code with seat controls</li>
              </ul>
              <button className='primary full' type='button' disabled={busy || current} onClick={() => onChoose(product)}>
                {current ? 'Current plan' : `Choose ${product?.name || tier}`}
              </button>
            </article>
          );
        })}
      </div>
    </>
  );
}

export default function PaymentPage({ user, setUser, productId, paymentStatus, paymentToken, onNavigate }) {
  const initialProductId = productId || 'polymath-chill-monthly';
  const [products, setProducts] = useState(FALLBACK_PRODUCTS);
  const [audience, setAudience] = useState('individual');
  const [billing, setBilling] = useState(initialProductId.includes('yearly') ? 'YEAR' : 'MONTH');
  const [selectedId, setSelectedId] = useState(initialProductId);
  const [walletOpen, setWalletOpen] = useState(initialProductId.startsWith('mcoins-'));
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const confirmationStarted = useRef(false);

  const subscriptions = useMemo(
    () => products.filter((product) => product.kind === 'subscription' && product.audience !== 'institution'),
    [products],
  );
  const institutionSubscriptions = useMemo(
    () => products.filter((product) => product.kind === 'subscription' && product.audience === 'institution'),
    [products],
  );
  const mcoinProducts = useMemo(
    () => products.filter((product) => product.kind === 'mcoins'),
    [products],
  );
  const selected = useMemo(
    () => products.find((product) => product.id === selectedId) || subscriptions[0],
    [products, selectedId, subscriptions],
  );

  useEffect(() => {
    trackProductEvent('subscription_page_viewed', {
      productId: initialProductId,
      signedIn: Boolean(user),
      audience,
    });
  }, []);

  useEffect(() => {
    if (!paymentStatus) return;
    trackProductEvent('checkout_returned', {
      productId: initialProductId,
      outcome: paymentStatus,
    });
  }, [paymentStatus]);

  useEffect(() => {
    apiRequest('/api/catalog')
      .then((data) => setProducts(data.products))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (user?.subscriptionTier !== 'chill') return;
    const interval = user.subscriptionInterval === 'YEAR' ? 'YEAR' : 'MONTH';
    setBilling(interval);
    setSelectedId(interval === 'YEAR'
      ? 'polymath-musician-yearly'
      : 'polymath-musician-monthly');
  }, [user?.subscriptionInterval, user?.subscriptionTier]);

  useEffect(() => {
    if (!user || !paymentToken || confirmationStarted.current) return;
    const subscriptionReturn = paymentStatus === 'subscription-approved';
    const orderReturn = paymentStatus === 'approved';
    if (!subscriptionReturn && !orderReturn) return;
    confirmationStarted.current = true;
    setBusy(true);
    setStatus(subscriptionReturn ? 'Confirming your subscription…' : 'Confirming your Mcoin purchase…');
    apiRequest(subscriptionReturn ? '/api/paypal/confirm-subscription' : '/api/paypal/capture-order', {
      method: 'POST',
      body: JSON.stringify(subscriptionReturn
        ? { subscriptionId: paymentToken }
        : { orderId: paymentToken }),
    })
      .then((data) => {
        setUser(data.user);
        setStatus(subscriptionReturn
          ? data.upgraded
            ? 'Musician is active. Your new billing period and translation allowance start today.'
            : `${data.product.name} is active.`
          : `${data.product.name} was added to your wallet.`);
      })
      .catch((error) => setStatus(error.message))
      .finally(() => setBusy(false));
  }, [paymentStatus, paymentToken, setUser, user]);

  function productFor(tier) {
    return subscriptions.find((product) => product.tier === tier && product.interval === billing);
  }

  async function checkout(product = selected) {
    if (!user) {
      onNavigate('account', { next: 'payment', productId: product?.id || selected?.id });
      return;
    }
    if (!product) return;
    setSelectedId(product.id);
    setBusy(true);
    setStatus(product.kind === 'subscription'
      ? 'Opening secure PayPal subscription checkout…'
      : 'Opening secure PayPal checkout…');
    try {
      const data = await apiRequest(
        product.kind === 'subscription' ? '/api/paypal/create-subscription' : '/api/paypal/create-order',
        { method: 'POST', body: JSON.stringify({ productId: product.id }) },
      );
      if (!data.approveUrl) throw new Error('PayPal did not return an approval link.');
      window.location.assign(data.approveUrl);
    } catch (error) {
      setStatus(error.message);
      setBusy(false);
    }
  }

  function choosePlan(product) {
    if (!product) return;
    setSelectedId(product.id);
  }

  const currentTier = user?.subscriptionTier || (user?.pro ? 'musician' : 'free');
  const currentInterval = user?.subscriptionInterval || 'MONTH';

  return (
    <section className="page-shell subscription-page">
      <div className="page-heading subscription-heading">
        <p className="eyebrow">Subscriptions</p>
        <h1>Choose access that fits you.</h1>
        <p>Start with who the subscription is for. You will only see the relevant options.</p>
      </div>

      <div className="subscription-audience" role="tablist" aria-label="Subscription category">
        <button type="button" role="tab" aria-selected={audience === 'individual'} className={audience === 'individual' ? 'active' : ''} onClick={() => setAudience('individual')}>
          <strong>Individual</strong>
          <span>For one musician</span>
        </button>
        <button type="button" role="tab" aria-selected={audience === 'institution'} className={audience === 'institution' ? 'active' : ''} onClick={() => setAudience('institution')}>
          <strong>Institution</strong>
          <span>For schools and organisations</span>
        </button>
      </div>

      {audience === 'individual' ? (
        <div className="individual-subscriptions" role="tabpanel">
          <div className="billing-switch segmented-control" role="group" aria-label="Billing period">
            <button type="button" className={billing === 'MONTH' ? 'active' : ''} onClick={() => setBilling('MONTH')}>Monthly</button>
            <button type="button" className={billing === 'YEAR' ? 'active' : ''} onClick={() => setBilling('YEAR')}>Yearly · save more</button>
          </div>

          <div className="subscription-plan-grid">
            {['chill', 'musician'].map((tier) => {
              const product = productFor(tier);
              const current = currentTier === tier && currentInterval === billing;
              const upgrade = currentTier === 'chill' && tier === 'musician' && currentInterval === billing;
              const priceDifference = upgrade && product
                ? Number(product.price) - Number(productFor('chill')?.price || 0)
                : 0;
              return (
                <article key={tier} className={`subscription-plan-card ${tier} ${selectedId === product?.id ? 'selected' : ''}`}>
                  <header>
                    <div>
                      <p className="eyebrow">{tier === 'musician' ? 'Complete access' : 'Simple access'}</p>
                      <h2>{tier === 'musician' ? 'Musician' : 'Chill'}</h2>
                    </div>
                    {tier === 'musician' && <span className="plan-badge">Learn + Band</span>}
                  </header>
                  <div className="subscription-price">
                    <strong>${product?.price || '—'}</strong>
                    <span>USD / {periodLabel(product)}</span>
                  </div>
                  <ul>
                    {FEATURES[tier].map((feature) => <li key={feature}>{feature}</li>)}
                  </ul>
                  {upgrade && (
                    <p className="upgrade-note">
                      Upgrade for ${priceDifference.toFixed(2)}. Today becomes day one of your new {periodLabel(product)}.
                    </p>
                  )}
                  <button
                    type="button"
                    className="primary full"
                    disabled={busy || current || (currentTier === 'musician' && tier === 'chill')}
                    onClick={() => { choosePlan(product); checkout(product); }}
                  >
                    {current
                      ? 'Current plan'
                      : currentTier === 'musician' && tier === 'chill'
                        ? 'Musician already includes Chill'
                        : upgrade
                          ? `Upgrade for ${priceDifference.toFixed(2)}`
                          : `Choose ${tier === 'musician' ? 'Musician' : 'Chill'}`}
                  </button>
                </article>
              );
            })}
          </div>
        </div>
      ) : (
        <div className='institution-subscriptions' role='tabpanel'>
          <InstitutionPlans
            products={institutionSubscriptions}
            billing={billing}
            setBilling={setBilling}
            user={user}
            busy={busy}
            onChoose={(product) => { choosePlan(product); checkout(product); }}
          />
        </div>
      )}

      <div className="subscription-economy-note">
        <strong>1 USD = 1 Mcoin</strong>
        <span>No conversion tricks. Subscribers pay 0.5 Mcoin per extra translation; users without a subscription pay 2 Mcoins.</span>
      </div>

      <section className="wallet-section">
        <button className="wallet-reveal" type="button" aria-expanded={walletOpen} onClick={() => setWalletOpen((open) => !open)}>
          <span><strong>Need Mcoins?</strong><small>Show one-time wallet packs</small></span>
          <span>{walletOpen ? '−' : '+'}</span>
        </button>
        {walletOpen && (
          <div className="wallet-pack-grid">
            {mcoinProducts.map((product) => (
              <button key={product.id} type="button" className="payment-product" disabled={busy} onClick={() => checkout(product)}>
                <strong>{product.mcoins} Mcoins</strong>
                <span>${product.price} USD once</span>
              </button>
            ))}
          </div>
        )}
      </section>

      {!user && <button className="ghost subscription-signin" type="button" onClick={() => onNavigate('account', { next: 'payment', productId: selected?.id })}>Sign in before checkout</button>}
      {paymentStatus === 'cancelled' && <p className="form-status">Checkout was cancelled. Nothing was charged.</p>}
      {status && <p className="form-status subscription-status" aria-live="polite">{status}</p>}
    </section>
  );
}
