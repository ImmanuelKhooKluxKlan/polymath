import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';
import '../subscriptionCatalogAdmin.css';

const EMPTY_CATEGORY = {
  name: '', slug: '', description: '', audience: 'creator', sortOrder: 30, status: 'draft',
};

const EMPTY_PLAN = {
  id: '', revision: 0, categoryId: 'category-create-music', name: '', slug: '', description: '',
  badge: '', price: '', currency: 'USD', interval: 'MONTH', sortOrder: 0, paypalProductId: '',
  paypalPlanId: '', featuresText: '', entitlements: [
    'create_music.projects', 'create_music.ai_guidance', 'create_music.arrangements',
    'create_music.exports', 'create_music.guide_voice',
  ], status: 'draft',
};

function planToForm(plan) {
  return {
    ...EMPTY_PLAN,
    ...plan,
    price: plan.price || '',
    featuresText: (plan.features || []).join('\n'),
    entitlements: plan.entitlements || [],
  };
}

function usd(value) {
  return Number(value || 0).toLocaleString('en-US', { style: 'currency', currency: 'USD' });
}

export default function SubscriptionCatalogAdmin() {
  const [catalog, setCatalog] = useState({ categories: [], plans: [], corePlans: [], entitlementOptions: [] });
  const [corePriceDrafts, setCorePriceDrafts] = useState({});
  const [view, setView] = useState('plans');
  const [categoryForm, setCategoryForm] = useState(EMPTY_CATEGORY);
  const [planForm, setPlanForm] = useState(EMPTY_PLAN);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Loading subscription catalog…');

  async function loadCatalog() {
    const data = await apiRequest('/api/admin/subscription-catalog');
    setCatalog(data);
    setCorePriceDrafts(Object.fromEntries((data.corePlans || []).map((plan) => [plan.id, {
      price: plan.price,
      paypalPlanId: plan.paypalPlanId || '',
      upgradePaypalPlanId: plan.upgradePaypalPlanId || '',
    }])));
    setPlanForm((current) => ({
      ...current,
      categoryId: data.categories.some((item) => item.id === current.categoryId)
        ? current.categoryId
        : data.categories[0]?.id || '',
    }));
    return data;
  }

  function updateCoreDraft(productId, field, value) {
    setCorePriceDrafts((current) => ({
      ...current,
      [productId]: { ...current[productId], [field]: value },
    }));
  }

  async function saveCorePrice(plan) {
    const draft = corePriceDrafts[plan.id];
    if (!draft) return;
    setBusy(true);
    setStatus(`Verifying ${plan.name} with PayPal...`);
    try {
      const data = await apiRequest(`/api/admin/subscription-products/${encodeURIComponent(plan.id)}`, {
        method: 'PATCH',
        body: JSON.stringify({
          price: Number(draft.price),
          currency: plan.currency,
          paypalPlanId: draft.paypalPlanId,
          upgradePaypalPlanId: draft.upgradePaypalPlanId,
        }),
      });
      await loadCatalog();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    loadCatalog().then(() => setStatus('')).catch((error) => setStatus(error.message));
  }, []);

  const selectedCategory = useMemo(
    () => catalog.categories.find((item) => item.id === planForm.categoryId),
    [catalog.categories, planForm.categoryId],
  );

  async function saveCategory(event) {
    event.preventDefault();
    setBusy(true);
    setStatus('Saving category…');
    try {
      const data = await apiRequest('/api/admin/subscription-categories', {
        method: 'POST', body: JSON.stringify(categoryForm),
      });
      await loadCatalog();
      setCategoryForm(EMPTY_CATEGORY);
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function updateCategory(category, changes) {
    setBusy(true);
    try {
      const data = await apiRequest(`/api/admin/subscription-categories/${encodeURIComponent(category.id)}`, {
        method: 'PATCH', body: JSON.stringify({ revision: category.revision, ...changes }),
      });
      await loadCatalog();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleEntitlement(key) {
    setPlanForm((current) => ({
      ...current,
      entitlements: current.entitlements.includes(key)
        ? current.entitlements.filter((item) => item !== key)
        : [...current.entitlements, key],
    }));
  }

  async function savePlan(event) {
    event.preventDefault();
    setBusy(true);
    setStatus(planForm.id ? 'Updating plan…' : 'Creating plan…');
    const payload = {
      ...planForm,
      unitAmountCents: Math.round(Number(planForm.price || 0) * 100),
      features: planForm.featuresText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
    };
    try {
      const data = await apiRequest(planForm.id
        ? `/api/admin/subscription-plans/${encodeURIComponent(planForm.id)}`
        : '/api/admin/subscription-plans', {
        method: planForm.id ? 'PATCH' : 'POST',
        body: JSON.stringify(payload),
      });
      await loadCatalog();
      setPlanForm({ ...EMPTY_PLAN, categoryId: planForm.categoryId });
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function setPlanStatus(plan, nextStatus) {
    setBusy(true);
    try {
      const data = await apiRequest(`/api/admin/subscription-plans/${encodeURIComponent(plan.id)}`, {
        method: 'PATCH', body: JSON.stringify({ revision: plan.revision, status: nextStatus }),
      });
      await loadCatalog();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function editPlan(plan) {
    setPlanForm(planToForm(plan));
    setView('plans');
    window.setTimeout(() => document.getElementById('subscription-plan-editor')?.scrollIntoView({ behavior: 'smooth' }), 0);
  }

  return (
    <section className='subscription-catalog-admin'>
      <header className='catalog-admin-heading'>
        <div>
          <p className='eyebrow'>Commercial control</p>
          <h2>Subscription catalog</h2>
          <p>Create categories, choose exactly what a plan unlocks, set its price, then publish it.</p>
        </div>
        <div className='catalog-admin-tabs' role='tablist' aria-label='Subscription catalog tools'>
          <button type='button' className={view === 'plans' ? 'active' : ''} onClick={() => setView('plans')}>Plans</button>
          <button type='button' className={view === 'categories' ? 'active' : ''} onClick={() => setView('categories')}>Categories</button>
        </div>
      </header>

      {view === 'categories' && (
        <div className='catalog-admin-layout'>
          <form className='catalog-editor' onSubmit={saveCategory}>
            <h3>New category</h3>
            <label className='field'>Name<input required value={categoryForm.name} placeholder='Create Music' onChange={(event) => setCategoryForm({ ...categoryForm, name: event.target.value })} /></label>
            <label className='field'>URL label<input value={categoryForm.slug} placeholder='create-music' onChange={(event) => setCategoryForm({ ...categoryForm, slug: event.target.value })} /></label>
            <label className='field'>Short description<textarea rows='3' value={categoryForm.description} onChange={(event) => setCategoryForm({ ...categoryForm, description: event.target.value })} /></label>
            <div className='catalog-two-fields'>
              <label className='field'>Audience<select value={categoryForm.audience} onChange={(event) => setCategoryForm({ ...categoryForm, audience: event.target.value })}><option value='creator'>Creator</option><option value='individual'>Individual</option><option value='institution'>Institution</option></select></label>
              <label className='field'>Order<input type='number' min='0' max='1000' value={categoryForm.sortOrder} onChange={(event) => setCategoryForm({ ...categoryForm, sortOrder: Number(event.target.value) })} /></label>
            </div>
            <label className='field'>Visibility<select value={categoryForm.status} onChange={(event) => setCategoryForm({ ...categoryForm, status: event.target.value })}><option value='draft'>Draft</option><option value='published'>Published</option></select></label>
            <button className='primary' type='submit' disabled={busy}>Create category</button>
          </form>
          <div className='catalog-list'>
            {catalog.categories.map((category) => (
              <article key={category.id}>
                <div><strong>{category.name}</strong><span>{category.description || 'No description'}</span><code>{category.slug}</code></div>
                <span className={`catalog-status ${category.status}`}>{category.status}</span>
                <button type='button' className='ghost' disabled={busy} onClick={() => updateCategory(category, { status: category.status === 'published' ? 'draft' : 'published' })}>{category.status === 'published' ? 'Hide' : 'Publish'}</button>
              </article>
            ))}
          </div>
        </div>
      )}

      {view === 'plans' && (
        <>
          <section className='catalog-core-pricing'>
            <header>
              <div><p className='eyebrow'>Live prices</p><h3>Built-in subscriptions</h3></div>
              <p>Changes apply to future checkouts. The server verifies the matching PayPal plan before saving.</p>
            </header>
            <div className='catalog-core-grid'>
              {(catalog.corePlans || []).map((plan) => {
                const draft = corePriceDrafts[plan.id] || {};
                return (
                  <article key={plan.id}>
                    <header><div><strong>{plan.name}</strong><span>{plan.categoryName} · {plan.interval.toLowerCase()}</span></div>{plan.priceEdited && <b>Edited</b>}</header>
                    <label className='field'>Price (USD)<input type='number' min='0.01' max='1000000' step='0.01' value={draft.price ?? ''} onChange={(event) => updateCoreDraft(plan.id, 'price', event.target.value)} /></label>
                    <label className='field'>PayPal plan ID<input value={draft.paypalPlanId ?? ''} placeholder='P-...' onChange={(event) => updateCoreDraft(plan.id, 'paypalPlanId', event.target.value)} /></label>
                    {plan.tier === 'musician' && <label className='field'>Chill upgrade plan ID<input value={draft.upgradePaypalPlanId ?? ''} placeholder='P-...' onChange={(event) => updateCoreDraft(plan.id, 'upgradePaypalPlanId', event.target.value)} /><small>This plan must include the correct remaining-balance setup fee.</small></label>}
                    <button type='button' className='primary' disabled={busy} onClick={() => saveCorePrice(plan)}>Verify and save price</button>
                  </article>
                );
              })}
            </div>
            <div className='catalog-safety-note'><strong>Existing subscribers stay protected.</strong><span>A new price changes future sign-ups only. It never silently changes an existing PayPal billing agreement.</span></div>
          </section>

          <div className='catalog-plan-grid'>
            {catalog.plans.map((plan) => (
              <article key={plan.id} className={plan.status === 'archived' ? 'archived' : ''}>
                <header><div><small>{plan.categoryName}</small><h3>{plan.name}</h3></div>{plan.badge && <b>{plan.badge}</b>}</header>
                <div className='catalog-plan-price'><strong>{usd(plan.price)}</strong><span>/ {plan.interval.toLowerCase()}</span></div>
                <ul>{plan.features.slice(0, 4).map((feature) => <li key={feature}>{feature}</li>)}</ul>
                <footer>
                  <span className={`catalog-status ${plan.status}`}>{plan.status}</span>
                  <button type='button' className='ghost' onClick={() => editPlan(plan)}>Edit</button>
                  {plan.status !== 'archived' && <button type='button' className='ghost' disabled={busy} onClick={() => setPlanStatus(plan, plan.status === 'published' ? 'draft' : 'published')}>{plan.status === 'published' ? 'Hide' : 'Publish'}</button>}
                </footer>
              </article>
            ))}
            {!catalog.plans.length && <div className='empty-state'><strong>No custom plans yet.</strong><span>Create the first plan below. Existing Chill, Musician, and institution plans remain unchanged.</span></div>}
          </div>

          <form id='subscription-plan-editor' className='catalog-editor catalog-plan-editor' onSubmit={savePlan}>
            <header>
              <div><p className='eyebrow'>{planForm.id ? 'Editing version' : 'New subscription'}</p><h3>{planForm.id ? planForm.name : 'Create a plan'}</h3></div>
              {planForm.id && <button type='button' className='ghost' onClick={() => setPlanForm({ ...EMPTY_PLAN, categoryId: planForm.categoryId })}>Cancel editing</button>}
            </header>
            <div className='catalog-form-grid'>
              <label className='field'>Category<select value={planForm.categoryId} onChange={(event) => setPlanForm({ ...planForm, categoryId: event.target.value })}>{catalog.categories.filter((item) => item.status !== 'archived').map((category) => <option key={category.id} value={category.id}>{category.name}</option>)}</select></label>
              <label className='field'>Plan name<input required value={planForm.name} placeholder='Creator' onChange={(event) => setPlanForm({ ...planForm, name: event.target.value })} /></label>
              <label className='field'>URL label<input value={planForm.slug} placeholder='creator-monthly' onChange={(event) => setPlanForm({ ...planForm, slug: event.target.value })} /></label>
              <label className='field'>Badge<input value={planForm.badge} placeholder='Best for songwriters' onChange={(event) => setPlanForm({ ...planForm, badge: event.target.value })} /></label>
              <label className='field'>Price (USD)<input required type='number' min='0' max='1000000' step='0.01' value={planForm.price} onChange={(event) => setPlanForm({ ...planForm, price: event.target.value })} /></label>
              <label className='field'>Billing<select value={planForm.interval} onChange={(event) => setPlanForm({ ...planForm, interval: event.target.value })}><option value='MONTH'>Monthly</option><option value='YEAR'>Yearly</option></select></label>
              <label className='field'>Display order<input type='number' min='0' max='1000' value={planForm.sortOrder} onChange={(event) => setPlanForm({ ...planForm, sortOrder: Number(event.target.value) })} /></label>
              <label className='field'>Visibility<select value={planForm.status} onChange={(event) => setPlanForm({ ...planForm, status: event.target.value })}><option value='draft'>Draft</option><option value='published'>Published</option><option value='archived'>Archived</option></select></label>
            </div>
            <label className='field'>Description<textarea rows='2' value={planForm.description} onChange={(event) => setPlanForm({ ...planForm, description: event.target.value })} /></label>
            <label className='field'>Features shown to customers<textarea rows='6' placeholder={'One feature per line\nAI lyric guidance\nPlayable backing arrangements'} value={planForm.featuresText} onChange={(event) => setPlanForm({ ...planForm, featuresText: event.target.value })} /></label>
            <fieldset className='catalog-entitlements'>
              <legend>Features enforced by the server</legend>
              <p>Display words do not grant access. These switches do.</p>
              <div>{catalog.entitlementOptions.map((option) => <label key={option.key}><input type='checkbox' checked={planForm.entitlements.includes(option.key)} onChange={() => toggleEntitlement(option.key)} /><span>{option.label}</span><code>{option.key}</code></label>)}</div>
            </fieldset>
            <div className='catalog-form-grid'>
              <label className='field'>PayPal product ID<input value={planForm.paypalProductId} placeholder='PROD-… (optional)' onChange={(event) => setPlanForm({ ...planForm, paypalProductId: event.target.value })} /></label>
              <label className='field'>PayPal plan ID<input value={planForm.paypalPlanId} placeholder='P-… required to publish a paid plan' onChange={(event) => setPlanForm({ ...planForm, paypalPlanId: event.target.value })} /></label>
            </div>
            <div className='catalog-safety-note'><strong>Prices are versioned.</strong><span>After a customer uses a plan, its billing terms are locked. Archive it and create a new version to change the price.</span></div>
            <button className='primary' type='submit' disabled={busy || !selectedCategory}>{planForm.id ? 'Save plan' : 'Create plan'}</button>
          </form>
        </>
      )}

      {status && <p className='form-status catalog-admin-status' aria-live='polite'>{status}</p>}
    </section>
  );
}
