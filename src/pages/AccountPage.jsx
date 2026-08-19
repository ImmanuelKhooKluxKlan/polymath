import { useEffect, useState } from 'react';
import { apiRequest, setAuthToken } from '../services/api.js';

export default function AccountPage({ user, setUser, onNavigate }) {
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState({ name: '', identifier: '', email: '', phone: '', password: '', birthDate: '', termsAccepted: false });
  const [status, setStatus] = useState('');
  const [withdraw, setWithdraw] = useState({ amountMcoins: '', payoutEmail: '' });
  const [newPassword, setNewPassword] = useState({ password: '', confirm: '' });
  const [voucherCode, setVoucherCode] = useState('');
  const [policies, setPolicies] = useState({
    registrationEnabled: true,
    minimumSignupAge: 0,
    minimumPasswordLength: 8,
    minimumWithdrawalMcoins: 100,
    policyNotice: '',
    termsUrl: '',
    privacyUrl: '',
  });

  useEffect(() => {
    apiRequest('/api/catalog')
      .then((data) => setPolicies((current) => ({ ...current, ...(data.policies || {}) })))
      .catch(() => {});
  }, []);

  async function submit(event) {
    event.preventDefault();
    setStatus('Working…');
    try {
      const authPath = mode === 'admin' ? 'login' : mode;
      const data = await apiRequest(`/api/auth/${authPath}`, {
        method: 'POST',
        body: JSON.stringify(mode === 'register'
          ? form
          : { identifier: form.identifier, password: form.password, admin: mode === 'admin' }),
      });
      setAuthToken(data.token);
      setUser(data.user);
      setStatus(`Welcome, ${data.user.name}.`);
      if (mode === 'admin') onNavigate('admin-database');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function logout() {
    try { await apiRequest('/api/auth/logout', { method: 'POST' }); } catch { /* Local logout still works. */ }
    setAuthToken('');
    setUser(null);
  }

  async function requestWithdrawal(event) {
    event.preventDefault();
    try {
      const data = await apiRequest('/api/wallet/withdraw', {
        method: 'POST',
        body: JSON.stringify({
          amountMcoins: Number(withdraw.amountMcoins),
          payoutEmail: withdraw.payoutEmail,
        }),
      });
      setUser(data.user);
      setStatus(`Withdrawal queued. Net payout balance: ${data.withdrawal.netMcoins} Mcoins. No additional Polymath Musician withdrawal fee was charged.`);
      setWithdraw({ amountMcoins: '', payoutEmail: '' });
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function redeemVoucher(event) {
    event.preventDefault();
    setStatus('Checking voucher...');
    try {
      const data = await apiRequest('/api/promotions/redeem', {
        method: 'POST',
        body: JSON.stringify({ code: voucherCode }),
      });
      setUser(data.user);
      setVoucherCode('');
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function changePassword(event) {
    event.preventDefault();
    if (newPassword.password !== newPassword.confirm) {
      setStatus('The new passwords do not match.');
      return;
    }
    try {
      const data = await apiRequest('/api/auth/change-password', {
        method: 'POST',
        body: JSON.stringify({ password: newPassword.password }),
      });
      setUser(data.user);
      setNewPassword({ password: '', confirm: '' });
      setStatus('Your password was changed. The temporary password no longer works.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (!user) {
    const heading = mode === 'admin'
      ? 'Sign in to the administrator dashboard.'
      : mode === 'login'
        ? 'Sign in to your music library.'
        : 'Create your Polymath Musician account.';

    return (
      <section className="page-shell narrow-page">
        <div className="page-heading">
          <p className="eyebrow">Account</p>
          <h1>{heading}</h1>
          <p>{mode === 'admin' ? 'Only backend-authorized administrator accounts can continue.' : 'Accounts include a secure wallet, purchases, listings, messages, and monthly PDF translation usage.'}</p>
        </div>
        <form className="account-card" onSubmit={submit}>
          <div className="segmented-control account-mode-control">
            <button type="button" className={mode === 'login' ? 'active' : ''} onClick={() => setMode('login')}>Sign in</button>
            <button type="button" className={mode === 'admin' ? 'active' : ''} onClick={() => setMode('admin')}>Admin sign in</button>
            <button type="button" className={mode === 'register' ? 'active' : ''} onClick={() => setMode('register')}>Register</button>
          </div>
          {mode === 'register' && (
            <>
              {!policies.registrationEnabled && <p className='form-status'>New account registration is currently closed by the administrator.</p>}
              {policies.policyNotice && <p className='policy-notice'>{policies.policyNotice}</p>}
              {(policies.termsUrl || policies.privacyUrl) && (
                <p className='policy-links'>
                  {policies.termsUrl && <a href={policies.termsUrl} target='_blank' rel='noreferrer'>Terms</a>}
                  {policies.privacyUrl && <a href={policies.privacyUrl} target='_blank' rel='noreferrer'>Privacy policy</a>}
                </p>
              )}
              {policies.minimumSignupAge > 0 && (
                <label className='field'>Date of birth<input type='date' value={form.birthDate} onChange={(event) => setForm({ ...form, birthDate: event.target.value })} required /></label>
              )}
              {(policies.minimumSignupAge > 0 || policies.policyNotice || policies.termsUrl || policies.privacyUrl) && (
                <label className='rights-check'>
                  <input type='checkbox' checked={form.termsAccepted} onChange={(event) => setForm({ ...form, termsAccepted: event.target.checked })} required />
                  <span>I confirm I meet the signup requirements and accept the registration rules and policies.</span>
                </label>
              )}
              <label className="field">Display name<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></label>
              <label className="field">Phone number<input type="tel" autoComplete="tel" placeholder="+65 8123 4567" value={form.phone} onChange={(event) => setForm({ ...form, phone: event.target.value })} required /></label>
            </>
          )}
          {mode !== 'register' ? (
            <label className="field">Email or phone number<input type="text" autoComplete="username" placeholder="Email or phone number" value={form.identifier} onChange={(event) => setForm({ ...form, identifier: event.target.value })} required /></label>
          ) : (
            <label className="field">Email<input type="email" autoComplete="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></label>
          )}
          <label className='field'>Password<input type='password' minLength={mode === 'register' ? policies.minimumPasswordLength : 8} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} required /></label>
          <button className='primary' type='submit' disabled={mode === 'register' && !policies.registrationEnabled}>{mode === 'register' ? 'Create account' : mode === 'admin' ? 'Open admin dashboard' : 'Sign in'}</button>
          {status && <p className="form-status">{status}</p>}
        </form>
      </section>
    );
  }

  if (user.mustChangePassword) {
    return (
      <section className="page-shell narrow-page">
        <div className="page-heading">
          <p className="eyebrow">Security required</p>
          <h1>Choose your private password.</h1>
          <p>You signed in with an administrator-issued temporary password. Replace it before continuing.</p>
        </div>
        <form className="account-card" onSubmit={changePassword}>
          <label className='field'>New password<input type='password' minLength={Math.max(12, policies.minimumPasswordLength)} autoComplete='new-password' value={newPassword.password} onChange={(event) => setNewPassword({ ...newPassword, password: event.target.value })} required /></label>
          <label className='field'>Confirm new password<input type='password' minLength={Math.max(12, policies.minimumPasswordLength)} autoComplete='new-password' value={newPassword.confirm} onChange={(event) => setNewPassword({ ...newPassword, confirm: event.target.value })} required /></label>
          <button className="primary" type="submit">Save new password</button>
          {status && <p className="form-status">{status}</p>}
        </form>
      </section>
    );
  }

  const allowance = user.translationAllowance || {
    limit: user.pro ? 20 : 1,
    used: 0,
    remaining: user.pro ? 20 : 1,
    resetAt: '',
  };
  const unlimitedTranslations = Boolean(user.admin || allowance.unlimited);
  return (
    <section className="page-shell">
      <div className="page-heading">
        <p className="eyebrow">Account & wallet</p>
        <h1>{user.name}</h1>
        <div className='friend-id-card'>
          <div><p className='eyebrow'>Your Friend ID</p><code>{user.friend_id || 'Loading...'}</code></div>
          <button className='ghost compact-action' type='button' disabled={!user.friend_id} onClick={() => navigator.clipboard.writeText(user.friend_id)}>Copy</button>
          <small>Share this short ID with friends. Many friends can use the same ID when an administrator activates a Friend ID voucher.</small>
        </div>
        <div className="button-row profile-shortcuts">
          <button className="primary" type="button" onClick={() => onNavigate('your-songs')}>Open Your Songs</button>
          <button className="ghost" type="button" onClick={() => onNavigate('published-songs')}>View marketplace listings</button>
          {user.admin && <button className='ghost' type='button' onClick={() => onNavigate('admin-database')}>Open admin console</button>}
        </div>
      </div>
      <div className="account-grid">
        <article className="wallet-card featured-card">
          <p className="eyebrow">Mcoin wallet</p>
          <strong className="wallet-balance">{user.mcoins.toLocaleString()}</strong>
          <p className="muted">$1 USD equals 10 Mcoins. Use Mcoins for marketplace purchases and 30-Mcoin PDF translations. Seller-earned withdrawable balance: {(user.withdrawableMcoins || 0).toLocaleString()} Mcoins.</p>
          <div className="button-row">
            <button className="primary" type="button" onClick={() => onNavigate('payment', { productId: 'mcoins-100' })}>Buy Mcoins</button>
            <button className="ghost" type="button" onClick={() => onNavigate('payment', { productId: 'polymath-pro' })}>{user.pro ? 'Pro active' : 'Subscribe to Pro'}</button>
          </div>
        </article>

        <article className="wallet-card translation-allowance-card">
          <p className="eyebrow">PDF translation allowance</p>
          <strong className="allowance-number">{unlimitedTranslations ? 'Unlimited' : allowance.remaining}</strong>
          <h2>{unlimitedTranslations ? 'administrator translations' : `of ${allowance.limit} remaining this month`}</h2>
          <p className="muted">{unlimitedTranslations ? 'Administrator access includes unlimited PDF and audio/video translations with no Mcoin charge.' : user.pro ? 'Pro includes 20 PDF-to-ready-to-play translations each month.' : 'Free accounts include one PDF-to-ready-to-play translation each month.'}</p>
          {!unlimitedTranslations && allowance.resetAt && <small>Resets {new Date(allowance.resetAt).toLocaleDateString()}</small>}
          <button className="ghost full" type="button" onClick={() => onNavigate(unlimitedTranslations || user.pro ? 'ensemble' : 'payment', unlimitedTranslations || user.pro ? {} : { productId: 'polymath-pro' })}>{unlimitedTranslations || user.pro ? 'Open Instrument Studio' : 'Get 20 with Pro'}</button>
        </article>

        <article className="wallet-card">
          <p className="eyebrow">Seller cash-out</p>
          <h2>No additional platform withdrawal fee</h2>
          <p className="muted">The 10% marketplace fee is deducted at the time of sale. Only the seller’s 90% earnings become withdrawable. Payout-provider, tax, or currency-conversion charges may still apply.</p>
          <form onSubmit={requestWithdrawal}>
            <label className='field'>Mcoins to withdraw<input type='number' min={policies.minimumWithdrawalMcoins} value={withdraw.amountMcoins} onChange={(event) => setWithdraw({ ...withdraw, amountMcoins: event.target.value })} required /></label>
            <label className="field">PayPal payout email<input type="email" value={withdraw.payoutEmail} onChange={(event) => setWithdraw({ ...withdraw, payoutEmail: event.target.value })} required /></label>
            <button className="ghost full" type="submit">Request withdrawal</button>
          </form>
        </article>

        <article className="wallet-card">
          <p className="eyebrow">Membership</p>
          <h2>{user.admin ? 'Administrator - Unlimited' : user.pro ? 'Polymath Musician Pro' : 'Free account'}</h2>
          <p className="muted">
            {user.admin
              ? 'Administrator translation limits and translation charges are disabled for this account.'
              : user.pro
              ? `Pro tools are active. PayPal subscription status: ${user.proStatus || 'ACTIVE'}.`
              : 'Pro is a recurring $19.99/month PayPal subscription with 20 PDF translations per month.'}
          </p>
          {!user.admin && !user.pro && <button className="primary" type="button" onClick={() => onNavigate('payment', { productId: 'polymath-pro' })}>Subscribe to Pro</button>}
          <button className="ghost" type="button" onClick={logout}>Sign out</button>
        </article>
        <article className='wallet-card'>
          <p className='eyebrow'>Voucher</p>
          <h2>Redeem Mcoins</h2>
          <p className='muted'>Enter an administrator-issued Mcoin voucher. Marketplace discount coupons are entered on the Marketplace page.</p>
          <form onSubmit={redeemVoucher}>
            <label className='field'>Voucher code<input value={voucherCode} maxLength='32' placeholder='WELCOME50' onChange={(event) => setVoucherCode(event.target.value.toUpperCase())} required /></label>
            <button className='ghost full' type='submit'>Redeem voucher</button>
          </form>
        </article>
      </div>

      {/* Device lab moved to the focused admin console.
      {user.admin && (
        <section className="device-lab">
          <header className="device-lab-heading">
            <div>
              <p className="eyebrow">Boss tools · private</p>
              <h2>Responsive Device Lab</h2>
              <p>Test the live application at real CSS viewport sizes. This panel is available only to backend-authorized administrator accounts.</p>
            </div>
            <span className="boss-badge">BOSS ACCOUNT</span>
          </header>

          <div className="device-lab-controls">
            <label>Device
              <select value={deviceId} onChange={(event) => setDeviceId(event.target.value)}>
                {DEVICE_PRESETS.map((device) => (
                  <option key={device.id} value={device.id}>{device.label} · {device.width}×{device.height}</option>
                ))}
              </select>
            </label>
            <label>Page
              <select value={previewRoute} onChange={(event) => setPreviewRoute(event.target.value)}>
                <option value="studio">Piano Studio</option>
                <option value="guitar">Guitar Studio</option>
                <option value="ensemble">Instrument Studio</option>
                <option value="band">Band</option>
                <option value="your-songs">Your Songs</option>
                <option value="published-songs">Marketplace</option>
              </select>
            </label>
            <label>Preview scale: {Math.round(previewScale * 100)}%
              <input type="range" min="0.35" max="1" step="0.05" value={previewScale} onChange={(event) => setPreviewScale(Number(event.target.value))} />
            </label>
            <button className="ghost" type="button" onClick={() => setLandscape((value) => !value)}>{landscape ? 'Use portrait' : 'Rotate landscape'}</button>
            <button className="ghost" type="button" onClick={() => setPreviewVersion((value) => value + 1)}>Reload preview</button>
            <button className="primary" type="button" onClick={() => window.open(previewUrl, '_blank', `popup=yes,width=${viewportWidth},height=${viewportHeight}`)}>Open test window</button>
          </div>

          <div className="device-lab-readout">
            <strong>{selectedDevice.label} · {landscape ? 'Landscape' : 'Portrait'}</strong>
            <span>{viewportWidth} × {viewportHeight} CSS pixels</span>
          </div>

          <div className="device-preview-scroll">
            <div
              className="device-preview-scale"
              style={{
                width: `${viewportWidth * previewScale}px`,
                height: `${viewportHeight * previewScale}px`,
              }}
            >
              <div
                className="device-preview-frame"
                style={{
                  width: `${viewportWidth}px`,
                  height: `${viewportHeight}px`,
                  transform: `scale(${previewScale})`,
                }}
              >
                <iframe
                  key={`${previewRoute}-${deviceId}-${landscape}-${previewVersion}`}
                  title={`${selectedDevice.label} responsive preview`}
                  src={previewUrl}
                  width={viewportWidth}
                  height={viewportHeight}
                />
              </div>
            </div>
          </div>
        </section>
      )}
      */}
      {status && <p className="form-status centered-status">{status}</p>}
    </section>
  );
}
