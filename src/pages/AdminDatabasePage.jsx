import { useEffect, useMemo, useState } from 'react';
import { apiAssetUrl, apiRequest } from '../services/api.js';
import { normalizeTeacherImage } from '../utils/teacherImage.js';
import { validateTeacherGlbFile } from '../utils/teacherModel.js';
import ModelLabPage from './ModelLabPage.jsx';

const ADMIN_SECTIONS = [
  ['overview', 'Overview', 'Health, revenue, and storage'],
  ['piano-lab', 'Machine learning', 'Data, training, checkpoints, accuracy, and model tests'],
  ['devices', 'Phone site review', 'Preview, test, and review mobile pages'],
  ['characters', 'Virtual teachers', 'Upload, publish, and delete custom characters'],
  ['community', 'Community safety', 'Review reported messages and moderation actions'],
  ['promotions', 'Discounts', 'Create percentage or fixed-Mcoin codes'],
  ['withdrawals', 'Payouts', 'Review pending cash-outs and platform outflow'],
  ['policies', 'Rules & policies', 'Security, marketplace, rewards, fees, and outflow limits'],
  ['users', 'Account manager', 'Search, Mcoins, access, and secure resets'],
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
  ['band', 'Band'], ['community', 'Community'], ['find-teacher', 'Find Teacher'], ['your-songs', 'Your Songs'], ['published-songs', 'Composers'],
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

const EMPTY_ACCOUNT_MANAGER = {
  userId: '',
  amountMcoins: 100,
  tier: 'musician',
  interval: 'MONTH',
};

const EMPTY_CHARACTER = {
  name: '',
  title: '',
  description: '',
  voice: '',
  voiceType: 'neutral',
  armTone: 'light',
  requiresAdultConfirmation: false,
};

function normalizedDigits(value) {
  return String(value || '').replace(/\D/g, '');
}

function accountMatchRank(row, query) {
  if (!query) return 0;
  const text = query.toLowerCase();
  const digits = normalizedDigits(query);
  const name = String(row.name || '').toLowerCase();
  const email = String(row.email || '').toLowerCase();
  const phone = normalizedDigits(row.phone);
  if (name.startsWith(text)) return 0;
  if (email.startsWith(text)) return 1;
  if (digits && phone.includes(digits)) return 2;
  if (name.includes(text)) return 3;
  if (email.includes(text)) return 4;
  if (String(row.friendId || '').toLowerCase().includes(text)) return 5;
  return Number.POSITIVE_INFINITY;
}

function formatAmount(amount, currency) {
  if (currency === 'USD') return Number(amount || 0).toLocaleString(undefined, { style: 'currency', currency: 'USD' });
  return `${Number(amount || 0).toLocaleString()} Mcoins`;
}

function promotionKindLabel(kind) {
  if (kind === 'marketplace_percent') return 'Composers percentage';
  if (kind === 'marketplace_fixed') return 'Composers fixed Mcoin discount';
  if (kind === 'friend_id_percent') return 'Friend ID percentage voucher';
  if (kind === 'subscription_percent') return 'Lucky code subscription discount';
  return 'Retired legacy promotion';
}

function promotionValueLabel(item) {
  if (item.retired) return 'Retired';
  return item.kind === 'marketplace_fixed'
    ? `${Number(item.value).toLocaleString()} Mcoins off`
    : `${item.value}% off`;
}

export default function AdminDatabasePage({ user, onNavigate }) {
  const [activeSection, setActiveSection] = useState('overview');
  const [database, setDatabase] = useState({ rows: [], footer: {}, configuration: {} });
  const [promotions, setPromotions] = useState([]);
  const [withdrawals, setWithdrawals] = useState({ withdrawals: [], summary: {} });
  const [policies, setPolicies] = useState(null);
  const [status, setStatus] = useState('Loading admin console...');
  const [userSearch, setUserSearch] = useState('');
  const [userSort, setUserSort] = useState('relevance');
  const [accountManager, setAccountManager] = useState(EMPTY_ACCOUNT_MANAGER);
  const [accountActionBusy, setAccountActionBusy] = useState(false);
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
  const [characters, setCharacters] = useState([]);
  const [communityReports, setCommunityReports] = useState({ reports: [], openCount: 0 });
  const [characterDraft, setCharacterDraft] = useState(EMPTY_CHARACTER);
  const [characterImage, setCharacterImage] = useState(null);
  const [characterImagePreview, setCharacterImagePreview] = useState('');
  const [characterModel, setCharacterModel] = useState(null);
  const [characterBusy, setCharacterBusy] = useState(false);
  const [characterUploadVersion, setCharacterUploadVersion] = useState(0);

  async function loadConsole() {
    const [usersData, policiesData, promotionsData, withdrawalsData, charactersData, communityData] = await Promise.all([
      apiRequest('/api/admin/users'),
      apiRequest('/api/admin/policies'),
      apiRequest('/api/admin/promotions'),
      apiRequest('/api/admin/withdrawals'),
      apiRequest('/api/admin/virtual-teachers'),
      apiRequest('/api/admin/community/reports'),
    ]);
    setDatabase(usersData);
    setPolicies(policiesData.policies);
    setPromotions(promotionsData.promotions);
    setWithdrawals(withdrawalsData);
    setCharacters(Array.isArray(charactersData.characters) ? charactersData.characters : []);
    setCommunityReports(communityData);
  }

  async function reviewCommunityReport(report, statusValue, removeMessage = false) {
    try {
      await apiRequest(`/api/admin/community/reports/${report.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ status: statusValue, removeMessage }),
      });
      const refreshed = await apiRequest('/api/admin/community/reports');
      setCommunityReports(refreshed);
      setStatus(removeMessage ? 'Message removed and report resolved.' : `Report ${statusValue}.`);
    } catch (error) {
      setStatus(error.message);
    }
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

  useEffect(() => () => {
    if (characterImagePreview) URL.revokeObjectURL(characterImagePreview);
  }, [characterImagePreview]);

  const spendingUsers = useMemo(
    () => database.rows.filter((row) => row.usdSpent > 0 || row.marketplaceSpentMcoins > 0).length,
    [database.rows],
  );
  const filteredUsers = useMemo(() => {
    const query = userSearch.trim();
    const matching = query
      ? database.rows.filter((row) => Number.isFinite(accountMatchRank(row, query)))
      : database.rows.slice();
    return matching.sort((left, right) => {
      if (query && userSort === 'relevance') {
        const relevance = accountMatchRank(left, query) - accountMatchRank(right, query);
        if (relevance) return relevance;
      }
      if (userSort === 'email') return String(left.email || '').localeCompare(String(right.email || ''), undefined, { sensitivity: 'base' });
      if (userSort === 'phone') return normalizedDigits(left.phone).localeCompare(normalizedDigits(right.phone), undefined, { numeric: true });
      if (userSort === 'newest') return String(right.createdAt || '').localeCompare(String(left.createdAt || ''));
      return String(left.name || '').localeCompare(String(right.name || ''), undefined, { sensitivity: 'base' });
    });
  }, [database.rows, userSearch, userSort]);
  const selectedAccount = database.rows.find((row) => row.userId === accountManager.userId) || null;

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

  async function chooseCharacterImage(event) {
    const source = event.target.files?.[0];
    if (!source) return;
    setCharacterBusy(true);
    setStatus('Preparing the character image...');
    try {
      const normalized = await normalizeTeacherImage(source);
      setCharacterImage(normalized);
      setCharacterImagePreview(URL.createObjectURL(normalized));
      setStatus('Image fitted to the teacher frame. Add the character details and publish.');
    } catch (error) {
      setCharacterImage(null);
      setCharacterImagePreview('');
      setStatus(error.message);
    } finally {
      setCharacterBusy(false);
    }
  }

  async function createCharacter(event) {
    event.preventDefault();
    if (!characterImage) {
      setStatus('Choose a character image first.');
      return;
    }
    const body = new FormData();
    Object.entries(characterDraft).forEach(([key, value]) => body.append(key, String(value)));
    body.append('image', characterImage, characterImage.name);
    if (characterModel) body.append('model', characterModel, characterModel.name);
    setCharacterBusy(true);
    setStatus(`Publishing ${characterDraft.name || 'character'}...`);
    try {
      const data = await apiRequest('/api/admin/virtual-teachers', { method: 'POST', body });
      setCharacters((current) => [...current, data.character]
        .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { sensitivity: 'base' })));
      setCharacterDraft(EMPTY_CHARACTER);
      setCharacterImage(null);
      setCharacterImagePreview('');
      setCharacterModel(null);
      setCharacterUploadVersion((current) => current + 1);
      window.dispatchEvent(new window.CustomEvent('polymath:virtual-teachers-changed'));
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setCharacterBusy(false);
    }
  }

  async function chooseCharacterModel(event) {
    const source = event.target.files?.[0];
    if (!source) {
      setCharacterModel(null);
      return;
    }
    setCharacterBusy(true);
    setStatus('Checking the 3D skeleton file...');
    try {
      setCharacterModel(await validateTeacherGlbFile(source));
      setStatus('GLB header is valid. The backend will verify its human skeleton when you publish.');
    } catch (error) {
      setCharacterModel(null);
      setStatus(error.message);
      event.target.value = '';
    } finally {
      setCharacterBusy(false);
    }
  }

  async function deleteCharacter(character) {
    const confirmed = window.confirm(
      `Delete ${character.name}?\n\nThis permanently removes the character, portrait, and rigged model. This cannot be undone.`,
    );
    if (!confirmed) return;
    setCharacterBusy(true);
    setStatus(`Deleting ${character.name}...`);
    try {
      const data = await apiRequest(`/api/admin/virtual-teachers/${encodeURIComponent(character.id)}`, { method: 'DELETE' });
      setCharacters((current) => current.filter((candidate) => candidate.id !== character.id));
      window.dispatchEvent(new window.CustomEvent('polymath:virtual-teachers-changed'));
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setCharacterBusy(false);
    }
  }

  async function saveMaximumCashout(event) {
    event.preventDefault();
    setStatus('Saving maximum cash-out...');
    try {
      const data = await apiRequest('/api/admin/policies', { method: 'PUT', body: JSON.stringify(policies) });
      setPolicies(data.policies);
      setStatus(`Maximum cash-out saved: ${data.policies.maximumWithdrawalMcoins > 0 ? `${Number(data.policies.maximumWithdrawalMcoins).toLocaleString()} Mcoins per request` : 'unlimited'}.`);
    } catch (error) { setStatus(error.message); }
  }

  async function reviewWithdrawal(withdrawalId, nextStatus) {
    const prompt = nextStatus === 'paid'
      ? 'Confirm that the payout was completed outside Polymath. Mark this request as paid?'
      : 'Reject this request and return the full requested Mcoins to the user?';
    if (!window.confirm(prompt)) return;
    setStatus(nextStatus === 'paid' ? 'Recording completed payout...' : 'Rejecting and refunding withdrawal...');
    try {
      const data = await apiRequest(`/api/admin/withdrawals/${withdrawalId}`, {
        method: 'PATCH',
        body: JSON.stringify({ status: nextStatus }),
      });
      const refreshed = await apiRequest('/api/admin/withdrawals');
      setWithdrawals(refreshed);
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

  function openAccountManager(row) {
    setAccountManager({
      userId: row.userId,
      amountMcoins: 100,
      tier: row.adminSubscriptionGrant?.tier || (['chill', 'musician'].includes(row.subscriptionTier) ? row.subscriptionTier : 'musician'),
      interval: row.adminSubscriptionGrant?.interval || row.subscriptionInterval || 'MONTH',
    });
    window.setTimeout(() => document.getElementById('admin-account-manager')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0);
  }

  async function grantMcoins(event) {
    event.preventDefault();
    if (!selectedAccount) return;
    const amountMcoins = Number(accountManager.amountMcoins);
    if (!window.confirm(`Give ${amountMcoins.toLocaleString()} Mcoins to ${selectedAccount.name}?`)) return;
    setAccountActionBusy(true);
    setStatus(`Adding Mcoins to ${selectedAccount.name}...`);
    try {
      const data = await apiRequest(`/api/admin/users/${selectedAccount.userId}/mcoins`, {
        method: 'POST',
        body: JSON.stringify({ amountMcoins }),
      });
      await loadConsole();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setAccountActionBusy(false);
    }
  }

  async function grantSubscription(event) {
    event.preventDefault();
    if (!selectedAccount) return;
    const tierLabel = accountManager.tier === 'musician' ? 'Musician' : 'Chill';
    const periodLabel = accountManager.interval === 'YEAR' ? 'one year' : 'one month';
    if (!window.confirm(`Grant or renew ${tierLabel} access for ${selectedAccount.name} for ${periodLabel}?`)) return;
    setAccountActionBusy(true);
    setStatus(`Updating ${selectedAccount.name}'s access...`);
    try {
      const data = await apiRequest(`/api/admin/users/${selectedAccount.userId}/subscription`, {
        method: 'POST',
        body: JSON.stringify({ tier: accountManager.tier, interval: accountManager.interval }),
      });
      await loadConsole();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setAccountActionBusy(false);
    }
  }

  async function removeSubscriptionGrant() {
    if (!selectedAccount?.adminSubscriptionGrant) return;
    if (!window.confirm(`Remove administrator-granted access from ${selectedAccount.name}? PayPal or institution access will not be changed.`)) return;
    setAccountActionBusy(true);
    try {
      const data = await apiRequest(`/api/admin/users/${selectedAccount.userId}/subscription`, { method: 'DELETE' });
      await loadConsole();
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setAccountActionBusy(false);
    }
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
      {activeSection === 'characters' && (
        <section className='admin-workspace admin-character-manager'>
          <div className='admin-section-heading'>
            <div>
              <p className='eyebrow'>Virtual teacher library</p>
              <h2>Create and manage characters</h2>
              <p>Upload one full-body portrait. Polymath fits it to every teacher frame before publishing it.</p>
            </div>
            <span className='status-pill'>{characters.length} custom</span>
          </div>

          <div className='admin-character-workspace'>
            <form className='admin-form-card admin-character-form' onSubmit={createCharacter}>
              <div className='admin-character-upload'>
                <div className='admin-character-preview'>
                  {characterImagePreview
                    ? <img src={characterImagePreview} alt='Prepared character preview' />
                    : <span><strong>Full-body image</strong><small>PNG, JPEG, or WebP</small></span>}
                </div>
                <label className='primary admin-character-file-button'>
                  {characterImage ? 'Replace image' : 'Choose image'}
                  <input key={characterUploadVersion} type='file' accept='image/png,image/jpeg,image/webp' onChange={chooseCharacterImage} disabled={characterBusy} />
                </label>
                <small>Images are contained, centred, and resized to 768 x 960 without stretching.</small>
              </div>
              <div className='admin-character-model-upload'>
                <div><strong>Optional articulated 3D body</strong><small>Upload a rigged binary glTF 2.0 model with named human bones. Maximum 25 MB.</small></div>
                <label className='ghost admin-character-file-button'>
                  {characterModel ? 'Replace GLB' : 'Choose rigged GLB'}
                  <input key={`model-${characterUploadVersion}`} type='file' accept='.glb,model/gltf-binary' onChange={chooseCharacterModel} disabled={characterBusy} />
                </label>
                {characterModel && <span>{characterModel.name} · {(characterModel.size / 1048576).toFixed(1)} MB</span>}
              </div>
              <div className='admin-form-grid'>
                <label className='field'>Character name<input maxLength='50' value={characterDraft.name} onChange={(event) => setCharacterDraft({ ...characterDraft, name: event.target.value })} placeholder='Lyra' required /></label>
                <label className='field'>Teacher role<input maxLength='80' value={characterDraft.title} onChange={(event) => setCharacterDraft({ ...characterDraft, title: event.target.value })} placeholder='Performance coach' required /></label>
                <label className='field'>Voice / style<input maxLength='50' value={characterDraft.voice} onChange={(event) => setCharacterDraft({ ...characterDraft, voice: event.target.value })} placeholder='Encouraging' required /></label>
                <label className='field'>Voice type<select value={characterDraft.voiceType} onChange={(event) => setCharacterDraft({ ...characterDraft, voiceType: event.target.value })}><option value='neutral'>Neutral / automatic</option><option value='feminine'>Feminine</option><option value='masculine'>Masculine</option></select></label>
                <label className='field'>Teacher hand tone<select value={characterDraft.armTone} onChange={(event) => setCharacterDraft({ ...characterDraft, armTone: event.target.value })}><option value='light'>Light</option><option value='dark'>Dark</option></select></label>
              </div>
              <label className='field'>Short description<textarea rows='3' maxLength='240' value={characterDraft.description} onChange={(event) => setCharacterDraft({ ...characterDraft, description: event.target.value })} placeholder='How this teacher helps a student.' required /></label>
              <label className='rights-check'><input type='checkbox' checked={characterDraft.requiresAdultConfirmation} onChange={(event) => setCharacterDraft({ ...characterDraft, requiresAdultConfirmation: event.target.checked })} /><span>Require an 18+ confirmation before this optional character is shown</span></label>
              <button className='primary' type='submit' disabled={characterBusy || !characterImage}>{characterBusy ? 'Working...' : 'Publish character'}</button>
            </form>

            <div className='admin-character-library'>
              <article className='admin-character-protection-note'>
                <strong>Built-in teachers are protected</strong>
                <span>Padme, Anakin, Taylor, and Mace cannot be deleted here. Only administrator uploads can be removed.</span>
              </article>
              <div className='admin-character-list'>
                {characters.map((character) => (
                  <article className='admin-character-row' key={character.id}>
                    <div className='admin-character-row-image'><img src={apiAssetUrl(character.imagePath)} alt={`${character.name} preview`} loading='lazy' /></div>
                    <div><strong>{character.name}</strong><span>{character.title}</span><small>{character.description}</small><div className='admin-character-badges'><b>{character.modelPath ? `Rigged 3D · ${character.rig?.jointCount || '?'} joints` : 'Procedural 3D body'}</b>{character.requiresAdultConfirmation && <b>18+ confirmation</b>}</div></div>
                    <button className='admin-character-delete' type='button' disabled={characterBusy} onClick={() => deleteCharacter(character)}>Delete</button>
                  </article>
                ))}
                {!characters.length && <div className='empty-state'>No custom characters yet. Upload the first one on the left.</div>}
              </div>
            </div>
          </div>
        </section>
      )}
      {activeSection === 'community' && (
        <section className='admin-workspace admin-community-safety'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Community moderation</p><h2>Reported messages</h2><p>Review context before removing content. Reports stay private from the message author.</p></div>
            <span className='status-pill'>{communityReports.openCount || 0} open</span>
          </div>
          <div className='admin-community-report-list'>
            {(communityReports.reports || []).map((report) => (
              <article key={report.id} className={`admin-community-report is-${report.status}`}>
                <header><div><strong>{report.room.name}</strong><small>Reported by {report.reporter} · {new Date(report.createdAt).toLocaleString()}</small></div><span className='status-pill'>{report.status}</span></header>
                <blockquote>{report.message ? <><b>{report.message.author}</b><span>{report.message.text}</span></> : <span>Message was already removed.</span>}</blockquote>
                <p>{report.reason}</p>
                {report.status === 'open' && <div className='button-row'><button type='button' className='primary' disabled={!report.message} onClick={() => reviewCommunityReport(report, 'resolved', true)}>Remove message</button><button type='button' className='ghost' onClick={() => reviewCommunityReport(report, 'dismissed')}>Keep message</button></div>}
              </article>
            ))}
            {!communityReports.reports?.length && <div className='empty-state'>No community reports. Free Flow is clear.</div>}
          </div>
        </section>
      )}
      {activeSection === 'promotions' && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Commercial tools</p><h2>Discount codes</h2><p>Use a percentage for subscriptions or Composers, or take an exact Mcoin amount off a Composers purchase.</p></div>
          </div>
          <form className='admin-form-card' onSubmit={createPromotion}>
            <div className='admin-form-grid'>
              <label className='field'>Code<input value={promotion.code} maxLength='32' placeholder='WELCOME50' onChange={(event) => setPromotion({ ...promotion, code: event.target.value.toUpperCase() })} required /></label>
              <label className='field'>Internal name<input value={promotion.name} placeholder='Launch voucher' onChange={(event) => setPromotion({ ...promotion, name: event.target.value })} required /></label>
              <label className='field'>Promotion type<select value={promotion.kind} onChange={(event) => setPromotion({ ...promotion, kind: event.target.value, value: event.target.value === 'marketplace_fixed' ? 10 : 20 })}><option value='subscription_percent'>Lucky code subscription percentage</option><option value='marketplace_percent'>Composers percentage coupon</option><option value='marketplace_fixed'>Composers fixed Mcoin coupon</option><option value='friend_id_percent'>Friend ID percentage voucher</option></select></label>
              <label className='field'>{promotion.kind === 'marketplace_fixed' ? 'Mcoins off' : 'Percentage off'}<input type='number' min={promotion.kind === 'marketplace_fixed' ? '0.01' : '1'} max={promotion.kind === 'marketplace_fixed' ? '1000000000' : '100'} step={promotion.kind === 'marketplace_fixed' ? '0.01' : '1'} value={promotion.value} onChange={(event) => setPromotion({ ...promotion, value: event.target.value })} required /><small>{promotion.kind === 'marketplace_fixed' ? 'The platform funds this exact discount; it cannot exceed the song price.' : 'Enter a value from 1 to 100.'}</small></label>
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
            {!promotions.length && <div className='empty-state'>No discount codes yet.</div>}
          </div>
        </section>
      )}
      {activeSection === 'withdrawals' && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Manual payout queue</p><h2>Cash-out requests</h2><p>Send the net payout outside Polymath before marking it paid. Rejecting a pending request refunds the user and reverses its platform fee.</p></div>
          </div>
          <div className='admin-summary-grid payout-summary-grid'>
            <article><span>Pending requests</span><strong>{Number(withdrawals.summary?.pendingCount || 0).toLocaleString()}</strong></article>
            <article><span>Pending gross</span><strong>{Number(withdrawals.summary?.pendingGrossMcoins || 0).toLocaleString()} Mcoins</strong></article>
            <article><span>Pending net outflow</span><strong>{Number(withdrawals.summary?.pendingNetMcoins || 0).toLocaleString()} Mcoins</strong></article>
          </div>
          {policies && (
            <form className='admin-form-card' onSubmit={saveMaximumCashout}>
              <div className='admin-section-heading'>
                <div><p className='eyebrow'>Cash-out guardrail</p><h3>Maximum per request</h3><p>This backend rule blocks any single cash-out above your chosen amount.</p></div>
              </div>
              <div className='admin-form-grid'>
                <label className='field'>Maximum cash-out (Mcoins)<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumWithdrawalMcoins} onChange={(event) => setPolicies({ ...policies, maximumWithdrawalMcoins: Number(event.target.value) })} /><small>0 means unlimited. It cannot be lower than the current minimum cash-out.</small></label>
              </div>
              <button className='primary' type='submit'>Save maximum cash-out</button>
            </form>
          )}
          <div className='database-table-wrap'>
            <table className='database-table payout-table'>
              <thead><tr><th>Account</th><th>Requested</th><th>Fee</th><th>Net payout</th><th>Payout email</th><th>Status</th><th>Review</th></tr></thead>
              <tbody>
                {withdrawals.withdrawals.map((item) => {
                  const pending = String(item.status || '').toLowerCase().startsWith('pending');
                  return (
                    <tr key={item.id}>
                      <td><strong>{item.account?.name || 'Deleted account'}</strong><small>{item.account?.email || item.userId}</small></td>
                      <td>{Number(item.amountMcoins || 0).toLocaleString()} Mcoins<small>{new Date(item.createdAt).toLocaleString()}</small></td>
                      <td>{Number(item.feeMcoins || 0).toLocaleString()} Mcoins</td>
                      <td className='amount-cell'>{Number(item.netMcoins || 0).toLocaleString()} Mcoins</td>
                      <td>{item.payoutEmail}</td>
                      <td><span className='status-pill'>{String(item.status || 'pending').replaceAll('_', ' ')}</span>{item.reviewedAt && <small>{new Date(item.reviewedAt).toLocaleString()}</small>}</td>
                      <td>{pending ? <div className='admin-user-actions'><button className='primary compact-action' type='button' onClick={() => reviewWithdrawal(item.id, 'paid')}>Mark paid</button><button className='ghost compact-action' type='button' onClick={() => reviewWithdrawal(item.id, 'rejected')}>Reject + refund</button></div> : <small>Completed</small>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!withdrawals.withdrawals.length && <div className='empty-state'>No cash-out requests yet.</div>}
          </div>
        </section>
      )}
      {activeSection === 'policies' && policies && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Platform controls</p><h2>Rules and policies</h2><p>These values are enforced by the backend, not only displayed in the browser.</p></div>
          </div>
          <form className='admin-form-card' onSubmit={savePolicies}>
            <div className='policy-control-group'>
              <div className='policy-control-heading'><div><h3>Accounts and registration</h3><p>Control who can register and the weakest password the backend accepts.</p></div></div>
              <label className='rights-check'><input type='checkbox' checked={policies.registrationEnabled} onChange={(event) => setPolicies({ ...policies, registrationEnabled: event.target.checked })} /><span>Allow new account registration</span></label>
              <div className='admin-form-grid policy-form-grid'>
                <label className='field'>Minimum signup age<input type='number' min='0' max='120' value={policies.minimumSignupAge} onChange={(event) => setPolicies({ ...policies, minimumSignupAge: Number(event.target.value) })} /><small>0 disables the age requirement</small></label>
                <label className='field'>Minimum password length<input type='number' min='1' max='256' value={policies.minimumPasswordLength} onChange={(event) => setPolicies({ ...policies, minimumPasswordLength: Number(event.target.value) })} /><small>1 is allowed. Under 8 is easy to attack.</small></label>
                <label className='field'>Welcome balance<input type='number' min='0' max='1000000000' step='0.01' value={policies.welcomeMcoins} onChange={(event) => setPolicies({ ...policies, welcomeMcoins: Number(event.target.value) })} /><small>Applied only to new accounts</small></label>
              </div>
            </div>

            <div className='policy-control-group'>
              <div className='policy-control-heading'>
                <div>
                  <h3>Private voice lessons</h3>
                  <p>Sessions use 30-minute blocks. Set one block price; longer lessons scale automatically.</p>
                </div>
              </div>
              <div className='admin-form-grid policy-form-grid'>
                <label className='field'>Price per 30 minutes<input type='number' min='0' max='1000000000' step='0.01' value={policies.virtualLessonPricePer30MinutesMcoins ?? 5} onChange={(event) => setPolicies({ ...policies, virtualLessonPricePer30MinutesMcoins: Number(event.target.value) })} /><small>Mcoins / US dollars; 0 makes private sessions free</small></label>
              </div>
            </div>

            <div className='policy-control-group'>
              <div className='policy-control-heading'><div><h3>Composers marketplace</h3><p>Listings can be sold, free, or pay each listener a reward funded by the composer.</p></div></div>
              <div className='admin-form-grid policy-form-grid'>
                <label className='field'>Minimum sale price<input type='number' min='0' max='1000000000' step='0.01' value={policies.minimumMarketplacePriceMcoins} onChange={(event) => setPolicies({ ...policies, minimumMarketplacePriceMcoins: Number(event.target.value) })} /><small>0 allows zero-price sales</small></label>
                <label className='field'>Maximum sale price<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumMarketplacePriceMcoins} onChange={(event) => setPolicies({ ...policies, maximumMarketplacePriceMcoins: Number(event.target.value) })} /><small>0 means unlimited</small></label>
                <label className='field'>Marketplace fee<input type='number' min='0' max='100' step='0.01' value={policies.marketplaceFeePercent} onChange={(event) => setPolicies({ ...policies, marketplaceFeePercent: Number(event.target.value) })} /><small>Percentage charged on new sale listings</small></label>
                <label className='field'>Maximum reward per listener<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumListenerRewardMcoins} onChange={(event) => setPolicies({ ...policies, maximumListenerRewardMcoins: Number(event.target.value) })} /><small>0 means unlimited</small></label>
                <label className='field'>Maximum reward outflow per listing<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumRewardOutflowPerListingMcoins} onChange={(event) => setPolicies({ ...policies, maximumRewardOutflowPerListingMcoins: Number(event.target.value) })} /><small>Total composer-funded rewards; 0 means unlimited</small></label>
              </div>
              <label className='rights-check'><input type='checkbox' checked={policies.listenerRewardsEnabled} onChange={(event) => setPolicies({ ...policies, listenerRewardsEnabled: event.target.checked })} /><span>Allow composers to pay people to claim and listen to their songs</span></label>
            </div>

            <div className='policy-control-group'>
              <div className='policy-control-heading'><div><h3>Cash-out and maximum outflow</h3><p>Limit individual requests, each account’s daily requests, and the platform’s combined pending payout exposure.</p></div></div>
              <div className='admin-form-grid policy-form-grid'>
                <label className='field'>Minimum withdrawal<input type='number' min='0' max='1000000000' step='0.01' value={policies.minimumWithdrawalMcoins} onChange={(event) => setPolicies({ ...policies, minimumWithdrawalMcoins: Number(event.target.value) })} /><small>0 removes the policy minimum; requests must still exceed 0</small></label>
                <label className='field'>Maximum per withdrawal<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumWithdrawalMcoins} onChange={(event) => setPolicies({ ...policies, maximumWithdrawalMcoins: Number(event.target.value) })} /><small>0 means unlimited</small></label>
                <label className='field'>Daily limit per account<input type='number' min='0' max='1000000000' step='0.01' value={policies.dailyWithdrawalLimitMcoins} onChange={(event) => setPolicies({ ...policies, dailyWithdrawalLimitMcoins: Number(event.target.value) })} /><small>Gross requested Mcoins; 0 means unlimited</small></label>
                <label className='field'>Maximum pending platform outflow<input type='number' min='0' max='1000000000' step='0.01' value={policies.maximumPendingWithdrawalOutflowMcoins} onChange={(event) => setPolicies({ ...policies, maximumPendingWithdrawalOutflowMcoins: Number(event.target.value) })} /><small>Combined net pending payouts; 0 means unlimited</small></label>
                <label className='field'>Cash-out fee<input type='number' min='0' max='100' step='0.01' value={policies.withdrawalFeePercent} onChange={(event) => setPolicies({ ...policies, withdrawalFeePercent: Number(event.target.value) })} /><small>Percentage retained by the platform</small></label>
              </div>
            </div>

            <div className='policy-control-group'>
              <div className='policy-control-heading'><div><h3>Published policy details</h3><p>Support contacts and links shown to users.</p></div></div>
              <div className='admin-form-grid policy-form-grid'>
                <label className='field'>Support email<input type='email' value={policies.supportEmail} onChange={(event) => setPolicies({ ...policies, supportEmail: event.target.value })} /></label>
                <label className='field'>Helpline phone<input type='tel' placeholder='+65 6123 4567' value={policies.supportPhone || ''} onChange={(event) => setPolicies({ ...policies, supportPhone: event.target.value })} /><small>Shown after a user reaches the daily Help limit</small></label>
                <label className='field'>Terms URL<input type='url' placeholder='https://' value={policies.termsUrl} onChange={(event) => setPolicies({ ...policies, termsUrl: event.target.value })} /></label>
                <label className='field'>Privacy URL<input type='url' placeholder='https://' value={policies.privacyUrl} onChange={(event) => setPolicies({ ...policies, privacyUrl: event.target.value })} /></label>
              </div>
              <label className='field'>Registration notice<textarea rows='4' value={policies.policyNotice} onChange={(event) => setPolicies({ ...policies, policyNotice: event.target.value })} placeholder='Short rules shown before signup.' /></label>
            </div>
            <button className='primary' type='submit'>Save rules and policies</button>
          </form>
        </section>
      )}
      {activeSection === 'users' && (
        <section className='admin-workspace'>
          <div className='admin-section-heading'>
            <div><p className='eyebrow'>Accounts and recovery</p><h2>User account manager</h2><p>Find an account, grant Mcoins, renew access, or issue a secure temporary password.</p></div>
            <div className='admin-user-tools'>
              <label className='admin-user-search'>Search users<input value={userSearch} onChange={(event) => setUserSearch(event.target.value)} placeholder='Start typing a name, email, or phone' /></label>
              <label>Arrange by<select value={userSort} onChange={(event) => setUserSort(event.target.value)}><option value='relevance'>Best match</option><option value='name'>Name A–Z</option><option value='email'>Email A–Z</option><option value='phone'>Phone number</option><option value='newest'>Newest account</option></select></label>
              <small>{filteredUsers.length} of {database.rows.length} accounts</small>
            </div>
          </div>
          <div className='database-table-wrap'>
            <table className='database-table admin-users-table'>
              <thead><tr><th>User</th><th>Contact</th><th>Wallet</th><th>Spent</th><th>Membership</th><th>Last login</th><th>Actions</th></tr></thead>
              <tbody>
                {filteredUsers.map((row) => (
                  <tr className={selectedAccount?.userId === row.userId ? 'selected-account-row' : ''} key={row.userId}>
                    <td><strong>{row.name}</strong><code className='friend-id-chip'>{row.friendId}</code><small>Joined {row.createdAt ? new Date(row.createdAt).toLocaleDateString() : '-'}</small></td>
                    <td>{row.email}<small>{row.phone || 'No phone'}</small></td>
                    <td className='amount-cell'>{row.unlimitedMcoins ? '∞ Mcoins' : `${row.mcoins.toLocaleString()} Mcoins`}</td>
                    <td>{formatAmount(row.usdSpent, 'USD')}<small>{row.marketplaceSpentMcoins.toLocaleString()} music-sheet Mcoins</small></td>
                    <td><span className='status-pill'>{row.admin ? 'ADMIN' : row.subscriptionTier === 'musician' ? 'MUSICIAN' : row.subscriptionTier === 'chill' ? 'CHILL' : row.proStatus}</span>{row.adminSubscriptionGrant && <small>{row.adminSubscriptionGrant.active ? 'Admin access until' : 'Admin access expired'} {row.adminSubscriptionGrant.expiresAt ? new Date(row.adminSubscriptionGrant.expiresAt).toLocaleDateString() : ''}</small>}</td>
                    <td>{row.lastLoginAt ? new Date(row.lastLoginAt).toLocaleString() : 'Never'}<small>{row.loginCount} recorded sign-ins</small></td>
                    <td><div className='admin-user-actions'><button className='primary compact-action' type='button' onClick={() => openAccountManager(row)}>Manage</button><button className='ghost compact-action' type='button' disabled={row.userId === user.user_id} onClick={() => openPasswordReset(row)}>{row.userId === user.user_id ? 'Your account' : 'Reset password'}</button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!filteredUsers.length && <div className='empty-state'>No account matches “{userSearch}”. Try fewer letters or digits.</div>}
          {selectedAccount && (
            <section className='admin-account-manager' id='admin-account-manager'>
              <header>
                <div><p className='eyebrow'>Selected account</p><h3>{selectedAccount.name}</h3><span>{selectedAccount.email}{selectedAccount.phone ? ` · ${selectedAccount.phone}` : ''}</span></div>
                <button className='ghost compact-action' type='button' onClick={() => setAccountManager(EMPTY_ACCOUNT_MANAGER)}>Close</button>
              </header>
              {selectedAccount.admin ? (
                <div className='admin-unlimited-notice'><strong>Administrator account</strong><span>This account already has unlimited Mcoins, translations, Learn, and Band access.</span></div>
              ) : (
                <div className='admin-account-actions-grid'>
                  <form onSubmit={grantMcoins}>
                    <div><p className='eyebrow'>Wallet gift</p><h3>{selectedAccount.mcoins.toLocaleString()} Mcoins</h3><small>Add spendable Mcoins. The action is recorded in the user’s ledger.</small></div>
                    <label className='field'>Mcoins to give<input type='number' min='0.01' max='1000000' step='0.01' value={accountManager.amountMcoins} onChange={(event) => setAccountManager({ ...accountManager, amountMcoins: event.target.value })} required /></label>
                    <button className='primary' type='submit' disabled={accountActionBusy}>Give Mcoins</button>
                  </form>
                  <form onSubmit={grantSubscription}>
                    <div><p className='eyebrow'>Subscription help</p><h3>Grant or renew access</h3><small>Extends the same active grant. This does not create, charge, cancel, or alter a PayPal subscription.</small></div>
                    <label className='field'>Plan<select value={accountManager.tier} onChange={(event) => setAccountManager({ ...accountManager, tier: event.target.value })}><option value='chill'>Chill</option><option value='musician'>Musician</option></select></label>
                    <label className='field'>Access period<select value={accountManager.interval} onChange={(event) => setAccountManager({ ...accountManager, interval: event.target.value })}><option value='MONTH'>One month</option><option value='YEAR'>One year</option></select></label>
                    {selectedAccount.adminSubscriptionGrant && <small className={selectedAccount.adminSubscriptionGrant.active ? 'active-grant' : 'expired-grant'}>{selectedAccount.adminSubscriptionGrant.active ? 'Active through' : 'Expired'} {new Date(selectedAccount.adminSubscriptionGrant.expiresAt).toLocaleDateString()}</small>}
                    <div className='button-row'><button className='primary' type='submit' disabled={accountActionBusy}>Grant / renew</button>{selectedAccount.adminSubscriptionGrant && <button className='ghost' type='button' disabled={accountActionBusy} onClick={removeSubscriptionGrant}>Remove manual access</button>}</div>
                  </form>
                </div>
              )}
            </section>
          )}
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
