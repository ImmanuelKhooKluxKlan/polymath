import { useEffect, useState } from 'react';
import { apiRequest, setAuthToken } from '../services/api.js';

export default function AccountPage({ user, setUser, onNavigate, returnPage, returnProductId }) {
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState({ name: '', identifier: '', contactMethod: 'email', email: '', phone: '', luckyCode: '', password: '', verificationCode: '', birthDate: '', termsAccepted: false });
  const [verification, setVerification] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [withdraw, setWithdraw] = useState({ amountMcoins: '', payoutEmail: '' });
  const [newPassword, setNewPassword] = useState({ password: '', confirm: '' });
  const [withdrawalFeeRate, setWithdrawalFeeRate] = useState(0.25);
  const [policies, setPolicies] = useState({
    registrationEnabled: true,
    minimumSignupAge: 0,
    minimumPasswordLength: 8,
    minimumWithdrawalMcoins: 20,
    policyNotice: '',
    termsUrl: '',
    privacyUrl: '',
  });

  useEffect(() => {
    apiRequest('/api/catalog')
      .then((data) => {
        setPolicies((current) => ({ ...current, ...(data.policies || {}) }));
        setWithdrawalFeeRate(Number(data.withdrawalFeeRate ?? 0.25));
      })
      .catch(() => {});
  }, []);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setStatus('Working...');
    try {
      if (mode === 'register' && !verification) {
        const channel = form.contactMethod;
        const data = await apiRequest('/api/auth/register/otp', {
          method: 'POST',
          body: JSON.stringify({
            channel,
            email: channel === 'email' ? form.email : '',
            phone: channel === 'phone' ? form.phone : '',
          }),
        });
        setVerification(data);
        setForm((current) => ({ ...current, verificationCode: '' }));
        setStatus(data.message);
        return;
      }

      const data = await apiRequest(`/api/auth/${mode}`, {
        method: 'POST',
        body: JSON.stringify(mode === 'register'
          ? {
            ...form,
            email: form.contactMethod === 'email' ? form.email : '',
            phone: form.contactMethod === 'phone' ? form.phone : '',
            challengeId: verification?.challengeId,
          }
          : { identifier: form.identifier, password: form.password }),
      });
      setAuthToken(data.token);
      setUser(data.user);
      setStatus(`Welcome, ${data.user.name}.`);
      if (data.user.admin) onNavigate('admin-database');
      else if (returnPage === 'payment') onNavigate('payment', { productId: returnProductId || 'polymath-chill-monthly' });
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  function changeMode(nextMode) {
    setMode(nextMode);
    setVerification(null);
    setForm((current) => ({ ...current, verificationCode: '' }));
    setStatus('');
  }

  function changeContactMethod(contactMethod) {
    setVerification(null);
    setForm((current) => ({ ...current, contactMethod, verificationCode: '' }));
    setStatus('');
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
      setStatus(`Cash-out queued. Fee: ${data.withdrawal.feeMcoins} Mcoins. Net payout: ${data.withdrawal.netMcoins} Mcoins.`);
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
    const heading = mode === 'login'
      ? 'Sign in to your music library.'
      : 'Create your Polymath Musician account.';

    return (
      <section className="page-shell narrow-page">
        <div className="page-heading">
          <p className="eyebrow">Account</p>
          <h1>{heading}</h1>
          <p>Accounts include a secure wallet, purchases, listings, messages, and monthly PDF translation usage.</p>
        </div>
        <form className="account-card" onSubmit={submit}>
          <div className="segmented-control account-mode-control">
            <button type="button" className={mode === 'login' ? 'active' : ''} onClick={() => changeMode('login')}>Sign in</button>
            <button type="button" className={mode === 'register' ? 'active' : ''} onClick={() => changeMode('register')}>Register</button>
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
              <div className="field">
                <span>Where should we send your verification code?</span>
                <div className="segmented-control registration-contact-control" role="group" aria-label="Verification contact method">
                  <button type="button" disabled={Boolean(verification)} className={form.contactMethod === 'email' ? 'active' : ''} onClick={() => changeContactMethod('email')}>Email</button>
                  <button type="button" disabled={Boolean(verification)} className={form.contactMethod === 'phone' ? 'active' : ''} onClick={() => changeContactMethod('phone')}>Phone</button>
                </div>
              </div>
              {form.contactMethod === 'email' ? (
                <label className="field">Email address<input type="email" autoComplete="email" disabled={Boolean(verification)} value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></label>
              ) : (
                <label className="field">Phone number<input type="tel" autoComplete="tel" disabled={Boolean(verification)} placeholder="+65 8123 4567" value={form.phone} onChange={(event) => setForm({ ...form, phone: event.target.value })} required /></label>
              )}
              <label className='field'>Lucky code<input type='text' autoComplete='off' maxLength={64} disabled={Boolean(verification)} value={form.luckyCode} onChange={(event) => setForm({ ...form, luckyCode: event.target.value })} /></label>
            </>
          )}
          {mode !== 'register' && (
            <label className="field">Email or phone number<input type="text" autoComplete="username" placeholder="Email or phone number" value={form.identifier} onChange={(event) => setForm({ ...form, identifier: event.target.value })} required /></label>
          )}
          <label className='field'>Password<input type='password' autoComplete={mode === 'register' ? 'new-password' : 'current-password'} minLength={mode === 'register' ? policies.minimumPasswordLength : 8} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} required /></label>
          {mode === 'register' && verification && (
            <>
              <label className="field">6-digit verification code<input type="text" inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={form.verificationCode} onChange={(event) => setForm({ ...form, verificationCode: event.target.value.replace(/\D/g, '').slice(0, 6) })} required /></label>
              <small className="field-help">Sent to {verification.destinationHint}. The code expires shortly.</small>
              <button type="button" className="ghost" onClick={() => { setVerification(null); setForm((current) => ({ ...current, verificationCode: '' })); setStatus(''); }}>Use a different email or phone</button>
            </>
          )}
          <button className='primary' type='submit' disabled={busy || (mode === 'register' && !policies.registrationEnabled)}>{mode === 'register' ? (verification ? 'Verify & create account' : 'Send verification code') : 'Sign in'}</button>
          {status && <p className="form-status" aria-live="polite">{status}</p>}
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
    limit: user.subscriptionTier === 'chill' ? 10 : user.pro ? 20 : 0,
    used: 0,
    remaining: user.subscriptionTier === 'chill' ? 10 : user.pro ? 20 : 0,
    resetAt: '',
  };
  const unlimitedTranslations = Boolean(user.admin || allowance.unlimited);
  const planName = user.subscriptionTier === 'musician'
    ? 'Musician'
    : user.subscriptionTier === 'chill'
      ? 'Chill'
      : 'Free';
  const withdrawalAmount = Math.max(0, Number(withdraw.amountMcoins) || 0);
  const withdrawalFeePreview = Number((withdrawalAmount * withdrawalFeeRate).toFixed(2));
  const withdrawalNetPreview = Number((withdrawalAmount - withdrawalFeePreview).toFixed(2));
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
          <button className="ghost" type="button" onClick={() => onNavigate('published-songs')}>Browse composers</button>
          {user.admin && <button className='ghost' type='button' onClick={() => onNavigate('admin-database')}>Open admin console</button>}
        </div>
      </div>
      <div className="account-grid">
        <article className="wallet-card featured-card">
          <p className="eyebrow">Mcoin wallet</p>
          <strong className="wallet-balance">{user.mcoins.toLocaleString()}</strong>
          <p className="muted">$1 USD equals 1 Mcoin. Cash-out eligible: {(user.cashoutEligibleMcoins ?? user.mcoins).toLocaleString()} Mcoins.</p>
          <div className="button-row">
            <button className="primary" type="button" onClick={() => onNavigate('payment', { productId: 'mcoins-100' })}>Buy Mcoins</button>
            <button className="ghost" type="button" onClick={() => onNavigate('payment', { productId: user.subscriptionTier === 'chill' ? 'polymath-musician-monthly' : 'polymath-chill-monthly' })}>{user.pro ? planName + ' active' : 'See subscriptions'}</button>
          </div>
        </article>

        <article className="wallet-card translation-allowance-card">
          <p className="eyebrow">Translation allowance</p>
          <strong className="allowance-number">{unlimitedTranslations ? 'Unlimited' : allowance.remaining}</strong>
          <h2>{unlimitedTranslations ? 'administrator translations' : `of ${allowance.limit} remaining this month`}</h2>
          <p className="muted">{unlimitedTranslations ? 'Administrator access includes unlimited PDF and audio/video translations with no Mcoin charge.' : user.pro ? planName + ' includes ' + allowance.limit + ' shared PDF or audio translations each month.' : 'Choose Chill or Musician for a monthly translation allowance.'}</p>
          {!unlimitedTranslations && allowance.resetAt && <small>Resets {new Date(allowance.resetAt).toLocaleDateString()}</small>}
          <button className="ghost full" type="button" onClick={() => onNavigate(unlimitedTranslations || user.pro ? 'ensemble' : 'payment', unlimitedTranslations || user.pro ? {} : { productId: 'polymath-chill-monthly' })}>{unlimitedTranslations || user.pro ? 'Open Instrument Studio' : 'See subscription options'}</button>
        </article>

        {user.institution && (
          <article className='wallet-card institution-access-card'>
            <p className='eyebrow'>Institution access</p>
            <h2>{user.institution.name || 'Institution Musician'}</h2>
            <p className='muted'>{user.institution.seats.toLocaleString()} student seats - {user.institution.status.toLowerCase()}.</p>
            {user.institution.role === 'owner' && user.institution.accessCode && (
              <div className='institution-code-row'>
                <code>{user.institution.accessCode}</code>
                <button className='ghost compact-action' type='button' onClick={() => navigator.clipboard.writeText(user.institution.accessCode)}>Copy</button>
              </div>
            )}
          </article>
        )}

        <article className='wallet-card'>
          <p className='eyebrow'>Cash out</p>
          <h2>Available to every account</h2>
          <p className="muted">A 25% cash-out fee applies. You receive 75% of the requested amount.</p>
          <form onSubmit={requestWithdrawal}>
            <label className='field'>Mcoins to withdraw<input type='number' min={policies.minimumWithdrawalMcoins} max={user.mcoins} step='0.01' value={withdraw.amountMcoins} onChange={(event) => setWithdraw({ ...withdraw, amountMcoins: event.target.value })} required /><small>Minimum: {policies.minimumWithdrawalMcoins} Mcoins · Available: {user.mcoins.toLocaleString()}</small></label>
            {withdrawalAmount > 0 && <div className='cashout-preview'><span>Fee: {withdrawalFeePreview.toLocaleString()} Mcoins</span><strong>You receive: {withdrawalNetPreview.toLocaleString()} Mcoins</strong></div>}
            <label className="field">PayPal payout email<input type="email" value={withdraw.payoutEmail} onChange={(event) => setWithdraw({ ...withdraw, payoutEmail: event.target.value })} required /></label>
            <button className="ghost full" type="submit">Request cash-out</button>
          </form>
        </article>

        <article className="wallet-card">
          <p className="eyebrow">Membership</p>
          <h2>{user.admin ? 'Administrator - Unlimited' : user.pro ? planName : 'Free account'}</h2>
          <p className="muted">
            {user.admin
              ? 'Administrator translation limits and translation charges are disabled for this account.'
              : user.pro
              ? planName + ' is active. PayPal subscription status: ' + (user.proStatus || 'ACTIVE') + '.'
              : 'Chill starts at $7.99/month. Musician unlocks Learn and Band.'}
          </p>
          {!user.admin && <button className="primary" type="button" onClick={() => onNavigate('payment', { productId: user.subscriptionTier === 'chill' ? 'polymath-musician-monthly' : 'polymath-chill-monthly' })}>{user.subscriptionTier === 'chill' ? 'Upgrade to Musician' : user.pro ? 'View plan' : 'See subscriptions'}</button>}
          <button className="ghost" type="button" onClick={logout}>Sign out</button>
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
                <option value="published-songs">Composers</option>
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
