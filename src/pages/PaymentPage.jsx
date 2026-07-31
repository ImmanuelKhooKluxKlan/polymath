import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

const FALLBACK_PRODUCTS = [
  {
    id: 'polymath-pro',
    name: 'Polymath Musician Pro',
    price: '19.99',
    currency: 'USD',
    kind: 'subscription',
    recurring: true,
    interval: 'MONTH',
    mcoins: 0,
  },
  { id: 'mcoins-50', name: '50 Mcoins', price: '5.00', currency: 'USD', kind: 'mcoins', mcoins: 50 },
  { id: 'mcoins-100', name: '100 Mcoins', price: '10.00', currency: 'USD', kind: 'mcoins', mcoins: 100 },
  { id: 'mcoins-300', name: '300 Mcoins', price: '30.00', currency: 'USD', kind: 'mcoins', mcoins: 300 },
];

function priceLabel(product) {
  if (product.kind === 'subscription') return `$${product.price} USD / month`;
  return `$${product.price} USD one time`;
}

export default function PaymentPage({ user, setUser, productId, paymentStatus, paymentToken, onNavigate }) {
  const [products, setProducts] = useState(FALLBACK_PRODUCTS);
  const [selectedId, setSelectedId] = useState(productId || 'polymath-pro');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const confirmationStarted = useRef(false);
  const selected = useMemo(
    () => products.find((item) => item.id === selectedId) || products[0],
    [products, selectedId],
  );

  useEffect(() => {
    apiRequest('/api/catalog')
      .then((data) => setProducts(data.products))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!user || !paymentToken || confirmationStarted.current) return;
    const isOrderReturn = paymentStatus === 'approved';
    const isSubscriptionReturn = paymentStatus === 'subscription-approved';
    if (!isOrderReturn && !isSubscriptionReturn) return;

    confirmationStarted.current = true;
    setBusy(true);
    setStatus(isSubscriptionReturn ? 'Verifying your recurring Pro subscription…' : 'Capturing your Mcoin payment…');

    const path = isSubscriptionReturn
      ? '/api/paypal/confirm-subscription'
      : '/api/paypal/capture-order';
    const body = isSubscriptionReturn
      ? { subscriptionId: paymentToken }
      : { orderId: paymentToken };

    apiRequest(path, { method: 'POST', body: JSON.stringify(body) })
      .then((data) => {
        setUser(data.user);
        if (isSubscriptionReturn) {
          setStatus(data.active
            ? 'Polymath Musician Pro is active with 20 PDF translations per month.'
            : `PayPal returned subscription status ${data.subscriptionStatus}. Pro will unlock after activation.`);
        } else {
          setStatus(`${data.product.name} has been added to your wallet.`);
        }
      })
      .catch((error) => setStatus(error.message))
      .finally(() => setBusy(false));
  }, [paymentStatus, paymentToken, user?.user_id, setUser]);

  async function checkout() {
    if (!user) {
      setStatus('Sign in before starting checkout.');
      return;
    }

    setBusy(true);
    setStatus(selected.kind === 'subscription'
      ? 'Creating your recurring PayPal subscription…'
      : 'Creating your secure one-time PayPal checkout…');

    try {
      const endpoint = selected.kind === 'subscription'
        ? '/api/paypal/create-subscription'
        : '/api/paypal/create-order';
      const data = await apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify({ productId: selected.id }),
      });
      if (!data.approveUrl) throw new Error('PayPal did not return an approval link.');
      window.location.assign(data.approveUrl);
    } catch (error) {
      setStatus(error.message);
      setBusy(false);
    }
  }

  return (
    <section className="page-shell payment-page">
      <div className="page-heading">
        <p className="eyebrow">Secure USD payment</p>
        <h1>Choose Pro or a transparent Mcoin pack.</h1>
        <p>$1 USD always equals 10 Mcoins. Pro renews monthly; Mcoin packs are one-time purchases.</p>
      </div>

      <div className="currency-trust-card">
        <strong>$1 USD = 10 Mcoins</strong>
        <span>PDF translation costs 30 Mcoins, equal to $3 USD.</span>
      </div>

      <div className="product-selector-grid">
        {products.map((product) => (
          <button
            key={product.id}
            type="button"
            className={`payment-product ${selectedId === product.id ? 'selected' : ''}`}
            onClick={() => setSelectedId(product.id)}
          >
            <span>{product.kind === 'subscription' ? 'MONTHLY PRO' : 'MCOINS'}</span>
            <strong>{product.name}</strong>
            <small>{priceLabel(product)}</small>
            {product.kind === 'subscription' && <small>20 PDF translations monthly</small>}
          </button>
        ))}
      </div>

      <article className="checkout-card">
        <div>
          <p className="eyebrow">Order summary</p>
          <h2>{selected.name}</h2>
          <p className="muted">
            {selected.kind === 'subscription'
              ? 'Recurring monthly Pro access with 20 PDF-to-ready-to-play translations each month. Manage or cancel through PayPal.'
              : `${selected.mcoins.toLocaleString()} Mcoins credited after payment. This exactly matches the $1-to-10-Mcoin rate.`}
          </p>
        </div>
        <strong className="checkout-price">{priceLabel(selected)}</strong>
        <button className="primary checkout-button" type="button" onClick={checkout} disabled={busy}>
          {busy ? 'Processing…' : selected.kind === 'subscription' ? 'Subscribe with PayPal' : 'Buy with PayPal'}
        </button>
        {!user && <button className="ghost" type="button" onClick={() => onNavigate('account')}>Sign in first</button>}
        {paymentStatus === 'cancelled' && <p className="form-status">Checkout was cancelled. Nothing was charged.</p>}
        {status && <p className="form-status">{status}</p>}
      </article>
    </section>
  );
}
