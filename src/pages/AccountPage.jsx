import { useState } from 'react';
import { apiRequest, setAuthToken } from '../services/api.js';

const DEVICE_PRESETS = [
  { id: 'phone', label: 'Phone', width: 390, height: 844 },
  { id: 'large-phone', label: 'Large phone', width: 430, height: 932 },
  { id: 'tablet', label: 'Tablet', width: 768, height: 1024 },
  { id: 'ipad', label: 'iPad Pro', width: 1024, height: 1366 },
  { id: 'laptop', label: 'Laptop', width: 1366, height: 768 },
  { id: 'desktop', label: 'Desktop', width: 1920, height: 1080 },
];

export default function AccountPage({ user, setUser, onNavigate }) {
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState({ name: '', identifier: '', email: '', phone: '', password: '' });
  const [status, setStatus] = useState('');
  const [withdraw, setWithdraw] = useState({ amountMcoins: '', payoutEmail: '' });
  const [newPassword, setNewPassword] = useState({ password: '', confirm: '' });
  const [deviceId, setDeviceId] = useState('phone');
  const [previewRoute, setPreviewRoute] = useState('studio');
  const [landscape, setLandscape] = useState(false);
  const [previewScale, setPreviewScale] = useState(0.75);
  const [previewVersion, setPreviewVersion] = useState(0);

  async function submit(event) {
    event.preventDefault();
    setStatus('Working…');
    try {
      const data = await apiRequest(`/api/auth/${mode}`, {
        method: 'POST',
        body: JSON.stringify(mode === 'login'
          ? { identifier: form.identifier, password: form.password }
          : form),
      });
      setAuthToken(data.token);
      setUser(data.user);
      setStatus(`Welcome, ${data.user.name}.`);
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
    return (
      <section className="page-shell narrow-page">
        <div className="page-heading">
          <p className="eyebrow">Account</p>
          <h1>{mode === 'login' ? 'Sign in to your music library.' : 'Create your Polymath Musician account.'}</h1>
          <p>Accounts include a secure wallet, purchases, listings, messages, and monthly PDF translation usage.</p>
        </div>
        <form className="account-card" onSubmit={submit}>
          <div className="segmented-control">
            <button type="button" className={mode === 'login' ? 'active' : ''} onClick={() => setMode('login')}>Sign in</button>
            <button type="button" className={mode === 'register' ? 'active' : ''} onClick={() => setMode('register')}>Register</button>
          </div>
          {mode === 'register' && (
            <>
              <label className="field">Display name<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></label>
              <label className="field">Phone number<input type="tel" autoComplete="tel" placeholder="+65 8123 4567" value={form.phone} onChange={(event) => setForm({ ...form, phone: event.target.value })} required /></label>
            </>
          )}
          {mode === 'login' ? (
            <label className="field">Email or phone number<input type="text" autoComplete="username" placeholder="Email or phone number" value={form.identifier} onChange={(event) => setForm({ ...form, identifier: event.target.value })} required /></label>
          ) : (
            <label className="field">Email<input type="email" autoComplete="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></label>
          )}
          <label className="field">Password<input type="password" minLength="8" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} required /></label>
          <button className="primary" type="submit">{mode === 'login' ? 'Sign in' : 'Create account'}</button>
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
          <label className="field">New password<input type="password" minLength="12" autoComplete="new-password" value={newPassword.password} onChange={(event) => setNewPassword({ ...newPassword, password: event.target.value })} required /></label>
          <label className="field">Confirm new password<input type="password" minLength="12" autoComplete="new-password" value={newPassword.confirm} onChange={(event) => setNewPassword({ ...newPassword, confirm: event.target.value })} required /></label>
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
  const selectedDevice = DEVICE_PRESETS.find((device) => device.id === deviceId) || DEVICE_PRESETS[0];
  const viewportWidth = landscape ? selectedDevice.height : selectedDevice.width;
  const viewportHeight = landscape ? selectedDevice.width : selectedDevice.height;
  const previewUrl = `${window.location.origin}${window.location.pathname}#${previewRoute}`;

  return (
    <section className="page-shell">
      <div className="page-heading">
        <p className="eyebrow">Account & wallet</p>
        <h1>{user.name}</h1>
        <p>User ID: <code>{user.user_id}</code></p>
        <div className="button-row profile-shortcuts">
          <button className="primary" type="button" onClick={() => onNavigate('your-songs')}>Open Your Songs</button>
          <button className="ghost" type="button" onClick={() => onNavigate('published-songs')}>View marketplace listings</button>
          {user.admin && <button className="ghost" type="button" onClick={() => onNavigate('admin-database')}>Open database table</button>}
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
          <strong className="allowance-number">{allowance.remaining}</strong>
          <h2>of {allowance.limit} remaining this month</h2>
          <p className="muted">{user.pro ? 'Pro includes 20 PDF-to-ready-to-play translations each month.' : 'Free accounts include one PDF-to-ready-to-play translation each month.'}</p>
          {allowance.resetAt && <small>Resets {new Date(allowance.resetAt).toLocaleDateString()}</small>}
          <button className="ghost full" type="button" onClick={() => onNavigate(user.pro ? 'ensemble' : 'payment', user.pro ? {} : { productId: 'polymath-pro' })}>{user.pro ? 'Open Instrument Studio' : 'Get 20 with Pro'}</button>
        </article>

        <article className="wallet-card">
          <p className="eyebrow">Seller cash-out</p>
          <h2>No additional platform withdrawal fee</h2>
          <p className="muted">The 10% marketplace fee is deducted at the time of sale. Only the seller’s 90% earnings become withdrawable. Payout-provider, tax, or currency-conversion charges may still apply.</p>
          <form onSubmit={requestWithdrawal}>
            <label className="field">Mcoins to withdraw<input type="number" min="100" value={withdraw.amountMcoins} onChange={(event) => setWithdraw({ ...withdraw, amountMcoins: event.target.value })} required /></label>
            <label className="field">PayPal payout email<input type="email" value={withdraw.payoutEmail} onChange={(event) => setWithdraw({ ...withdraw, payoutEmail: event.target.value })} required /></label>
            <button className="ghost full" type="submit">Request withdrawal</button>
          </form>
        </article>

        <article className="wallet-card">
          <p className="eyebrow">Membership</p>
          <h2>{user.pro ? 'Polymath Musician Pro' : 'Free account'}</h2>
          <p className="muted">
            {user.pro
              ? `Pro tools are active. PayPal subscription status: ${user.proStatus || 'ACTIVE'}.`
              : 'Pro is a recurring $19.99/month PayPal subscription with 20 PDF translations per month.'}
          </p>
          {!user.pro && <button className="primary" type="button" onClick={() => onNavigate('payment', { productId: 'polymath-pro' })}>Subscribe to Pro</button>}
          <button className="ghost" type="button" onClick={logout}>Sign out</button>
        </article>
      </div>

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
      {status && <p className="form-status centered-status">{status}</p>}
    </section>
  );
}
