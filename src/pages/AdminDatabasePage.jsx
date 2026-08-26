import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';
import ModelLabPage from './ModelLabPage.jsx';

const ADMIN_SECTIONS = [
  ['overview', 'Overview', 'Health, revenue, and storage'],
  ['piano-lab', 'Piano model lab', 'Private 88-key transcription testing'],
  ['devices', 'Phone site review', 'Preview, test, and review mobile pages'],
  ['promotions', 'Discounts', 'Create and pause percentage codes'],
  ['policies', 'Rules & policies', 'Signup and spending minimums'],
  ['users', 'Users & passwords', 'Accounts and secure resets'],
];

const DEVICE_PRESETS = [
  ['small-phone', 'Small phone', 320, 568],
  ['phone', 'Modern phone', 390, 844],
  ['large-phone', 'Large phone', 430, 932],
  ['foldable', 'Foldable narrow', 280, 653],
  ['small-tablet', 'Small tablet', 600, 960],
  ['ipad-mini', 'iPad Mini', 768, 1024],
  ['ipad-pro', 'iPad Pro', 1024, 1366],
  ['laptop', 'Laptop', 1366, 768],
  ['desktop', 'Desktop', 1920, 1080],
  ['ultrawide', 'Large desktop', 2560, 1440],
  ['custom', 'Custom viewport', 390, 844],
].map(([id, label, width, height]) => ({ id, label, width, height }));

const PREVIEW_PAGES = [
  ['studio', 'Piano Studio'], ['guitar', 'Guitar Studio'], ['ensemble', 'Instrument Studio'],
  ['band', 'Band'], ['your-songs', 'Your Songs'], ['published-songs', 'Composers'],
  ['payment', 'Payments'], ['account', 'Account'],
];

const PHONE_REVIEW_STORAGE_KEY = 'polymath-admin-phone-reviews-v1';

function loadPhoneReviews() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(PHONE_REVIEW_STORAGE_KEY) || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch {
    return {};
  }
}

const EMPTY_PROMOTION = {
  code: '', name: '', kind: 'subscription_percent', value: 20, minimumSpendMcoins: 0,
  minimumAccountAgeDays: 0, maxRedemptions: 0, perUserLimit: 1, startsAt: '', expiresAt: '',
};

function formatAmount(amount, currency) {
  if (currency === 'USD') return Number(amount || 0).toLocaleString(undefined, { style: 'currency', currency: 'USD' });
  return `${Number(amount || 0).toLocaleString()} Mcoins`;
}

function promotionKindLabel(kind) {
  if (kind === 'marketplace_percent') return 'Composers percentage';
  if (kind === 'friend_id_percent') return 'Friend ID percentage voucher';
  if (kind === 'subscription_percent') return 'Lucky code subscription discount';
  return 'Retired legacy promotion';
}

function promotionValueLabel(item) {
  return item.retired ? 'Retired' : String(item.value) + '% off';
}

export default function AdminDatabasePage({ user, onNavigate }) {
  const [activeSection, setActiveSection] = useState('overview');
  const [database, setDatabase] = useState({ rows: [], footer: {}, configuration: {} });
  const [promotions, setPromotions] = useState([]);
  const [policies, setPolicies] = useState(null);
  const [status, setStatus] = useState('Loading admin console...');
  const [userSearch, setUserSearch] = useState('');
  const [passwordReset, setPasswordReset] = useState({ userId: '', name: '', email: '', password: '', confirm: '' });
  const [issuedPassword, setIssuedPassword] = useState('');
  const [promotion, setPromotion] = useState(EMPTY_PROMOTION);
  const [deviceId, setDeviceId] = useState('phone');
  const [previewRoute, setPreviewRoute] = useState('studio');
  const [landscape, setLandscape] = useState(false);
  const [previewScale, setPreviewScale] = useState(0.65);
  const [previewVersion, setPreviewVersion] = useState(0);
  const [customViewport, setCustomViewport] = useState({ width: 390, height: 844 });
  const [phoneReviews, setPhoneReviews] = useState(loadPhoneReviews);

  async function loadConsole() {
    const [usersData, policiesData, promotionsData] = await Promise.all([
      apiRequest('/api/admin/users'),
      apiRequest('/api/admin/policies'),
      apiRequest('/api/admin/promotions'),
    ]);
    setDatabase(usersData);
    setPolicies(policiesData.policies);
    setPromotions(promotionsData.promotions);
  }

  useEffect(() => {
    if (!user?.admin) return;
    loadConsole().then(() => setStatus('')).catch((error) => setStatus(error.message));
  }, [user?.admin]);

  useEffect(() => {
    try {
      window.localStorage.setItem(PHONE_REVIEW_STORAGE_KEY, JSON.stringify(phoneReviews));
    } catch {
      // Reviewing still works for this session when browser storage is unavailable.
    }
  }, [phoneReviews]);

  const spendingUsers = useMemo(
    () => database.rows.filter((row) => row.usdSpent > 0 || row.marketplaceSpentMcoins > 0).length,
    [database.rows],
  );
  const filteredUsers = useMemo(() => {
    const query = userSearch.trim().toLowerCase();
    return query
      ? database.rows.filter((row) => `${row.name} ${row.email} ${row.phone} ${row.friendId}`.toLowerCase().includes(query))
      : database.rows;
  }, [database.rows, userSearch]);

  const preset = DEVICE_PRESETS.find((device) => device.id === deviceId) || DEVICE_PRESETS[1];
  const baseWidth = preset.id === 'custom' ? Number(customViewport.width) || 390 : preset.width;
  const baseHeight = preset.id === 'custom' ? Number(customViewport.height) || 844 : preset.height;
  const viewportWidth = landscape ? baseHeight : baseWidth;
  const viewportHeight = landscape ? baseWidth : baseHeight;
  const previewUrl = `${window.location.origin}${window.location.pathname}#${previewRoute}`;
  const reviewOrientation = landscape ? 'landscape' : 'portrait';
  const reviewKey = `${deviceId}:${reviewOrientation}:${previewRoute}`;
  const currentReview = phoneReviews[reviewKey] || { status: '', notes: '' };
  const reviewPrefix = `${deviceId}:${reviewOrientation}:`;
  const reviewedPageCount = PREVIEW_PAGES.filter(([route]) => phoneReviews[`${reviewPrefix}${route}`]?.status).length;

  function updatePhoneReview(changes) {
    setPhoneReviews((current) => ({
      ...current,
      [reviewKey]: {
        status: '',
        notes: '',
        ...current[reviewKey],
        ...changes,
        updatedAt: new Date().toISOString(),
      },
    }));
  }

  function clearPhoneReview() {
    setPhoneReviews((current) => {
      const next = { ...current };
      delete next[reviewKey];
      return next;
    });
  }

  async function savePolicies(event) {
    event.preventDefault();
    setStatus('Saving policies...');
    try {
      const data = await apiRequest('/api/admin/policies', { method: 'PUT', body: JSON.stringify(policies) });
      setPolicies(data.policies);
      setStatus(data.message);
    } catch (error) { setStatus(error.message); }
  }

  async function createPromotion(event) {
    event.preventDefault();
    setStatus('Creating promotion...');
    try {
      const data = await apiRequest('/api/admin/promotions', {
        method: 'POST',
        body: JSON.stringify({
          ...promotion,
          value: Number(promotion.value),
          minimumSpendMcoins: Number(promotion.minimumSpendMcoins),
          minimumAccountAgeDays: Number(promotion.minimumAccountAgeDays),
          maxRedemptions: Number(promotion.maxRedemptions),
          perUserLimit: Number(promotion.perUserLimit),
        }),
      });
      setPromotions((current) => [data.promotion, ...current]);
      setPromotion(EMPTY_PROMOTION);
      setStatus(data.message);
    } catch (error) { setStatus(error.message); }
  }

  async function togglePromotion(item) {
    try {
      const data = await apiRequest(`/api/admin/promotions/${item.id}`, {
        method: 'PATCH', body: JSON.stringify({ active: !item.active }),
      });
      setPromotions((current) => current.map((candidate) => candidate.id === item.id ? data.promotion : candidate));
      setStatus(data.message);
    } catch (error) { setStatus(error.message); }
  }

  async function resetPassword(event) {
    event.preventDefault();
    const row = database.rows.find((candidate) => candidate.userId === passwordReset.userId);
    if (!row) return;
    if (passwordReset.password && passwordReset.password !== passwordReset.confirm) {
      setStatus('The temporary passwords do not match.');
      return;
    }
    if (!window.confirm(`Issue a temporary password for ${row.name}? Every existing session will be signed out.`)) return;
    setStatus(`Resetting ${row.name}'s password...`);
    try {
      const data = await apiRequest(`/api/admin/users/${row.userId}/reset-password`, {
        method: 'POST',
        body: JSON.stringify({ password: passwordReset.password }),
      });
      setIssuedPassword(data.temporaryPassword);
      setPasswordReset((current) => ({ ...current, password: '', confirm: '' }));
      setDatabase((current) => ({
        ...current,
        rows: current.rows.map((candidate) => candidate.userId === row.userId
          ? { ...candidate, passwordResetAt: new Date().toISOString() } : candidate),
      }));
      setStatus(data.message);
    } catch (error) { setStatus(error.message); }
  }

  function openPasswordReset(row) {
    setIssuedPassword('');
    setPasswordReset({ userId: row.userId, name: row.name, email: row.email, password: '', confirm: '' });
  }

  if (!user?.admin) {
    return (
      <section className='page-shell narrow-page'>
        <div className='empty-state'>
          <h1>Administrator access required</h1>
          <p>This console contains private customer, security, and commercial controls.</p>
          <button className='primary' type='button' onClick={() => onNavigate('account')}>Return to account</button>
        </div>
      </section>
    );
  }

  return (
    <section className='page-shell admin-console'>
      <header className='admin-console-heading'>
        <div>
          <p className='eyebrow'>Protected administrator area</p>
          <h1>Admin console</h1>
          <p>One focused workspace at a time, with sensitive actions kept behind administrator authentication.</p>
        </div>
        <span className='boss-badge'>ADMIN</span>
      </header>
      <nav className='admin-section-nav' aria-label='Admin console sections'>
        {ADMIN_SECTIONS.map(([id, label, description]) => (
          <button key={id} type='button' className={activeSection === id ? 'active' : ''} onClick={() => setActiveSection(id)}>
            <strong>{label}</strong><small>{description}</small>
          </button>
        ))}
      </nav>

      {activeSection === 'overview' && (
        <section className='admin-workspace'>
          <div className='admin-summary-grid'>
            <article className='wallet-card'><p className='eyebrow'>Users</p><strong className='wallet-balance'>{database.rows.length}</strong><p className='muted'>{spendingUsers} purchasing accounts</p></article>
            <article className='wallet-card'><p className='eyebrow'>USD revenue</p><strong className='admin-metric'>{formatAmount(database.footer.totalUsdRevenue, 'USD')}</strong><p className='muted'>Completed PayPal orders</p></article>
            <article className='wallet-card'><p className='eyebrow'>Promotions</p><strong className='admin-metric'>{promotions.filter((item) => item.active).length}</strong><p className='muted'>Active vouchers and coupons</p></article>
          </div>
          <div className='admin-option-grid'>
            {ADMIN_SECTIONS.slice(1).map(([id, label, description]) => (
              <button key={id} type='button' onClick={() => setActiveSection(id)}>
                <span>{label}</span><small>{description}</small><b aria-hidden='true'>Open</b>
              </button>
            ))}
          </div>
          <article className='admin-storage-note'>
            <div><p className='eyebrow'>Data persistence</p><h2>{database.configuration.persistence || 'Atomic JSON file'}</h2></div>
            <p>User profiles, salted password hashes, purchases, policies, promotions, and login records are stored in <code>server/{database.configuration.databasePath || 'data/database.json'}</code>. New session credentials are stored as hashes, not as the browser's raw token.</p>
          </article>
        </section>
      )}
      {activeSection === 'piano-lab' && (
        <section className='admin-workspace admin-piano-lab'>
          <ModelLabPage onNavigate={onNavigate} embedded />
        </section>
      )}
      {activeSection === 'devices' && (
        <section className='admin-workspace device-lab'>
          <header className='device-lab-heading'>
            <div><p className='eyebrow'>Mobile quality check</p><h2>Phone site review</h2><p>Use the site like a phone user, then record whether each page is ready or needs work.</p></div>
            <span className='phone-review-count'>{reviewedPageCount}/{PREVIEW_PAGES.length} reviewed</span>
          </header>
          <div className='device-lab-controls'>
            <label>Device
              <select value={deviceId} onChange={(event) => setDeviceId(event.target.value)}>
                {DEVICE_PRESETS.map((device) => <option key={device.id} value={device.id}>{device.label} - {device.width} x {device.height}</option>)}
              </select>
            </label>
            <label>Page
              <select value={previewRoute} onChange={(event) => setPreviewRoute(event.target.value)}>
                {PREVIEW_PAGES.map(([route, label]) => <option key={route} value={route}>{label}</option>)}
              </select>
            </label>
            <label>Preview scale: {Math.round(previewScale * 100)}%
              <input type='range' min='0.25' max='1' step='0.05' value={previewScale} onChange={(event) => setPreviewScale(Number(event.target.value))} />
            </label>
            <button className='ghost' type='button' onClick={() => setLandscape((value) => !value)}>{landscape ? 'Use portrait' : 'Rotate landscape'}</button>
            <button className='ghost' type='button' onClick={() => setPreviewVersion((value) => value + 1)}>Reload</button>
            <button className='primary' type='button' onClick={() => window.open(previewUrl, '_blank', `popup=yes,width=${viewportWidth},height=${viewportHeight}`)}>Open phone window</button>
          </div>
          {deviceId === 'custom' && (
            <div className='custom-viewport-controls'>
              <label className='field'>Width<input type='number' min='240' max='5120' value={customViewport.width} onChange={(event) => setCustomViewport({ ...customViewport, width: event.target.value })} /></label>
              <label className='field'>Height<input type='number' min='320' max='2880' value={customViewport.height} onChange={(event) => setCustomViewport({ ...customViewport, height: event.target.value })} /></label>
            </div>
          )}
          <div className='device-lab-readout'><strong>{preset.label} - {landscape ? 'Landscape' : 'Portrait'}</strong><span>{viewportWidth} x {viewportHeight} CSS pixels</span></div>
          <div className='phone-review-workspace'>
            <div className='device-preview-scroll'>
              <div className='device-preview-scale' style={{ width: `${viewportWidth * previewScale}px`, height: `${viewportHeight * previewScale}px` }}>
                <div className='device-preview-frame' style={{ width: `${viewportWidth}px`, height: `${viewportHeight}px`, transform: `scale(${previewScale})` }}>
                  <iframe key={`${previewRoute}-${deviceId}-${landscape}-${previewVersion}`} title={`${preset.label} responsive preview`} src={previewUrl} width={viewportWidth} height={viewportHeight} />
                </div>
              </div>
            </div>
            <aside className='phone-review-card'>
              <div>
                <p className='eyebrow'>Review this screen</p>
                <h3>{PREVIEW_PAGES.find(([route]) => route === previewRoute)?.[1] || 'Selected page'}</h3>
                <small>{preset.label} · {reviewOrientation}</small>
              </div>
              <div className='phone-review-actions' role='group' aria-label='Review result'>
                <button className={currentReview.status === 'pass' ? 'review-pass active' : 'review-pass'} type='button' aria-pressed={currentReview.status === 'pass'} onClick={() => updatePhoneReview({ status: 'pass' })}>Looks good</button>
                <button className={currentReview.status === 'fix' ? 'review-fix active' : 'review-fix'} type='button' aria-pressed={currentReview.status === 'fix'} onClick={() => updatePhoneReview({ status: 'fix' })}>Needs work</button>
              </div>
              <label className='phone-review-notes'>Notes
                <textarea rows='4' maxLength='600' placeholder='What looks wrong or feels difficult to use?' value={currentReview.notes || ''} onChange={(event) => updatePhoneReview({ notes: event.target.value })} />
              </label>
              {(currentReview.status || currentReview.notes) && <button className='ghost compact-action' type='button' onClick={clearPhoneReview}>Clear this review</button>}
              <div className='phone-review-pages' aria-label='Page review progress'>
                {PREVIEW_PAGES.map(([route, label]) => {
                  const pageReview = phoneReviews[`${reviewPrefix}${route}`];
                  return (
                    <button key={route} className={previewRoute === route ? 'selected' : ''} type='button' onClick={() => setPreviewRoute(route)}>
                      <span>{label}</span>
                      <b className={pageReview?.status || 'pending'}>{pageReview?.status === 'pass' ? 'Good' : pageReview?.status === 'fix' ? 'Fix' : 'Unchecked'}</b>
                    </button>
                  );
                })}
              </div>
              <small className='phone-review-storage-note'>Reviews save automatically on this browser.</small>
            </aside>
          </div>
        </section>
      )}
      {activeSection === 'promotions' && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Commercial tools</p><h2>Percentage discounts</h2><p>Codes reduce a subscription or Composers purchase by a percentage. They never add Mcoins to a wallet.</p></div>
          </div>
          <form className='admin-form-card' onSubmit={createPromotion}>
            <div className='admin-form-grid'>
              <label className='field'>Code<input value={promotion.code} maxLength='32' placeholder='WELCOME50' onChange={(event) => setPromotion({ ...promotion, code: event.target.value.toUpperCase() })} required /></label>
              <label className='field'>Internal name<input value={promotion.name} placeholder='Launch voucher' onChange={(event) => setPromotion({ ...promotion, name: event.target.value })} required /></label>
              <label className='field'>Promotion type<select value={promotion.kind} onChange={(event) => setPromotion({ ...promotion, kind: event.target.value })}><option value='subscription_percent'>Lucky code subscription percentage</option><option value='marketplace_percent'>Composers percentage coupon</option><option value='friend_id_percent'>Friend ID percentage voucher</option></select></label>
              <label className='field'>Percentage off<input type='number' min='1' max='100' value={promotion.value} onChange={(event) => setPromotion({ ...promotion, value: event.target.value })} required /></label>
              <label className='field'>Minimum spend (Mcoins)<input type='number' min='0' value={promotion.minimumSpendMcoins} onChange={(event) => setPromotion({ ...promotion, minimumSpendMcoins: event.target.value })} /></label>
              <label className='field'>Minimum account age (days)<input type='number' min='0' value={promotion.minimumAccountAgeDays} onChange={(event) => setPromotion({ ...promotion, minimumAccountAgeDays: event.target.value })} /></label>
              <label className='field'>Total redemption limit<input type='number' min='0' value={promotion.maxRedemptions} onChange={(event) => setPromotion({ ...promotion, maxRedemptions: event.target.value })} /><small>0 means unlimited</small></label>
              <label className='field'>Uses per buyer<input type='number' min='0' max='100' value={promotion.perUserLimit} onChange={(event) => setPromotion({ ...promotion, perUserLimit: event.target.value })} /><small>0 means unlimited. A friend’s ID can be shared with many buyers.</small></label>
              <label className='field'>Starts<input type='datetime-local' value={promotion.startsAt} onChange={(event) => setPromotion({ ...promotion, startsAt: event.target.value })} /></label>
              <label className='field'>Expires<input type='datetime-local' value={promotion.expiresAt} onChange={(event) => setPromotion({ ...promotion, expiresAt: event.target.value })} /></label>
            </div>
            <button className='primary' type='submit'>Create promotion</button>
          </form>
          <div className='promotion-list'>
            {promotions.map((item) => (
              <article key={item.id} className={`promotion-row ${item.active ? '' : 'inactive'}`}>
                <div><code>{item.code}</code><strong>{item.name}</strong><small>{promotionKindLabel(item.kind)}</small></div>
                <div><strong>{promotionValueLabel(item)}</strong><small>{item.minimumSpendMcoins ? `Minimum ${item.minimumSpendMcoins} Mcoins` : 'No minimum spend'}</small></div>
                <div><strong>{item.redemptionCount.toLocaleString()}</strong><small>{item.maxRedemptions ? `of ${item.maxRedemptions} uses` : 'redemptions'}</small></div>
                <button className='ghost compact-action' type='button' disabled={item.retired} onClick={() => togglePromotion(item)}>{item.retired ? 'Retired' : item.active ? 'Pause' : 'Activate'}</button>
              </article>
            ))}
            {!promotions.length && <div className='empty-state'>No percentage discounts yet.</div>}
          </div>
        </section>
      )}
      {activeSection === 'policies' && policies && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Platform controls</p><h2>Rules and policies</h2><p>These values are enforced by the backend, not only displayed in the browser.</p></div>
          </div>
          <form className='admin-form-card' onSubmit={savePolicies}>
            <label className='rights-check'><input type='checkbox' checked={policies.registrationEnabled} onChange={(event) => setPolicies({ ...policies, registrationEnabled: event.target.checked })} /><span>Allow new account registration</span></label>
            <div className='admin-form-grid policy-form-grid'>
              <label className='field'>Minimum signup age<input type='number' min='0' max='120' value={policies.minimumSignupAge} onChange={(event) => setPolicies({ ...policies, minimumSignupAge: Number(event.target.value) })} /><small>0 disables the age requirement</small></label>
              <label className='field'>Minimum password length<input type='number' min='8' max='64' value={policies.minimumPasswordLength} onChange={(event) => setPolicies({ ...policies, minimumPasswordLength: Number(event.target.value) })} /></label>
              <label className='field'>Minimum listing price<input type='number' min='1' step='10' value={policies.minimumMarketplacePriceMcoins} onChange={(event) => setPolicies({ ...policies, minimumMarketplacePriceMcoins: Number(event.target.value) })} /><small>Mcoins; rounded to a valid 10-Mcoin price</small></label>
              <label className='field'>Minimum withdrawal<input type='number' min='1' value={policies.minimumWithdrawalMcoins} onChange={(event) => setPolicies({ ...policies, minimumWithdrawalMcoins: Number(event.target.value) })} /><small>Mcoins</small></label>
              <label className='field'>Welcome balance<input type='number' min='0' value={policies.welcomeMcoins} onChange={(event) => setPolicies({ ...policies, welcomeMcoins: Number(event.target.value) })} /><small>Applied only to new accounts</small></label>
              <label className='field'>Support email<input type='email' value={policies.supportEmail} onChange={(event) => setPolicies({ ...policies, supportEmail: event.target.value })} /></label>
              <label className='field'>Terms URL<input type='url' placeholder='https://' value={policies.termsUrl} onChange={(event) => setPolicies({ ...policies, termsUrl: event.target.value })} /></label>
              <label className='field'>Privacy URL<input type='url' placeholder='https://' value={policies.privacyUrl} onChange={(event) => setPolicies({ ...policies, privacyUrl: event.target.value })} /></label>
            </div>
            <label className='field'>Registration notice<textarea rows='4' value={policies.policyNotice} onChange={(event) => setPolicies({ ...policies, policyNotice: event.target.value })} placeholder='Short rules shown before signup.' /></label>
            <button className='primary' type='submit'>Save rules and policies</button>
          </form>
        </section>
      )}
      {activeSection === 'users' && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Accounts and recovery</p><h2>Users and password resets</h2><p>Issue a temporary password. The user is signed out everywhere and must choose a private password at next login.</p></div>
            <label className='admin-user-search'>Search users<input value={userSearch} onChange={(event) => setUserSearch(event.target.value)} placeholder='Name, email, or phone' /></label>
          </div>
          <div className='database-table-wrap'>
            <table className='database-table admin-users-table'>
              <thead><tr><th>User</th><th>Contact</th><th>Wallet</th><th>Spent</th><th>Membership</th><th>Last login</th><th>Security</th></tr></thead>
              <tbody>
                {filteredUsers.map((row) => (
                  <tr key={row.userId}>
                    <td><strong>{row.name}</strong><code className='friend-id-chip'>{row.friendId}</code><small>Joined {row.createdAt ? new Date(row.createdAt).toLocaleDateString() : '-'}</small></td>
                    <td>{row.email}<small>{row.phone || 'No phone'}</small></td>
                    <td className='amount-cell'>{row.mcoins.toLocaleString()} Mcoins</td>
                    <td>{formatAmount(row.usdSpent, 'USD')}<small>{row.marketplaceSpentMcoins.toLocaleString()} music-sheet Mcoins</small></td>
                    <td><span className='status-pill'>{row.proStatus}</span></td>
                    <td>{row.lastLoginAt ? new Date(row.lastLoginAt).toLocaleString() : 'Never'}<small>{row.loginCount} recorded sign-ins</small></td>
                    <td><button className='ghost compact-action' type='button' disabled={row.userId === user.user_id} onClick={() => openPasswordReset(row)}>{row.userId === user.user_id ? 'Your account' : 'Reset password'}</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {passwordReset.userId && (
            <form className='temporary-password-card' onSubmit={resetPassword}>
              <div><strong>Reset password for {passwordReset.name}</strong><span>{passwordReset.email}</span></div>
              {!issuedPassword ? (
                <>
                  <p className='muted'>Leave both fields blank to generate a strong temporary password automatically.</p>
                  <label className='field'>Optional temporary password<input type='password' autoComplete='new-password' value={passwordReset.password} onChange={(event) => setPasswordReset({ ...passwordReset, password: event.target.value })} /></label>
                  <label className='field'>Confirm temporary password<input type='password' autoComplete='new-password' value={passwordReset.confirm} onChange={(event) => setPasswordReset({ ...passwordReset, confirm: event.target.value })} /></label>
                  <div className='button-row'><button className='primary' type='submit'>Issue temporary password</button><button className='ghost' type='button' onClick={() => setPasswordReset({ userId: '', name: '', email: '', password: '', confirm: '' })}>Cancel</button></div>
                </>
              ) : (
                <>
                  <p>Copy this password now. It is shown only in this response:</p>
                  <code>{issuedPassword}</code>
                  <div className='button-row'>
                    <button className='primary' type='button' onClick={() => navigator.clipboard.writeText(issuedPassword)}>Copy password</button>
                    <button className='ghost' type='button' onClick={() => { setIssuedPassword(''); setPasswordReset({ userId: '', name: '', email: '', password: '', confirm: '' }); }}>Done</button>
                  </div>
                </>
              )}
            </form>
          )}
          <p className='privacy-note'>Private administrative data. Passwords are never stored in plain text; only scrypt hashes and random salts are persisted.</p>
        </section>
      )}
      {status && <p className='form-status floating-status'>{status}</p>}
    </section>
  );
}
