import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';

function formatAmount(amount, currency) {
  if (currency === 'USD') return Number(amount).toLocaleString(undefined, { style: 'currency', currency: 'USD' });
  return `${Number(amount).toLocaleString()} Mcoins`;
}

export default function AdminDatabasePage({ user, onNavigate }) {
  const [database, setDatabase] = useState({ rows: [], footer: {} });
  const [status, setStatus] = useState('Loading user accounts...');
  const [passwordReset, setPasswordReset] = useState({ userId: '', name: '', email: '', password: '', confirm: '' });

  useEffect(() => {
    if (!user?.admin) return;
    apiRequest('/api/admin/users')
      .then((data) => {
        setDatabase(data);
        setStatus('');
      })
      .catch((error) => setStatus(error.message));
  }, [user?.admin]);

  const spendingUsers = useMemo(
    () => database.rows.filter((row) => row.usdSpent > 0 || row.marketplaceSpentMcoins > 0).length,
    [database.rows],
  );

  async function resetPassword(event) {
    event.preventDefault();
    const row = database.rows.find((candidate) => candidate.userId === passwordReset.userId);
    if (!row) return;
    if (passwordReset.password !== passwordReset.confirm) {
      setStatus('The passwords do not match.');
      return;
    }
    const confirmed = window.confirm(`Reset the password for ${row.name} (${row.email})? Their existing sessions will be signed out.`);
    if (!confirmed) return;
    setStatus(`Resetting ${row.name}'s password...`);
    try {
      const data = await apiRequest(`/api/admin/users/${row.userId}/reset-password`, {
        method: 'POST',
        body: JSON.stringify({ password: passwordReset.password }),
      });
      setPasswordReset({ userId: '', name: '', email: '', password: '', confirm: '' });
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (!user?.admin) {
    return (
      <section className="page-shell narrow-page">
        <div className="empty-state">
          <h1>Administrator access required</h1>
          <p>This table contains private customer and purchase information.</p>
          <button className="primary" type="button" onClick={() => onNavigate('account')}>Return to account</button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-shell admin-database-page">
      <div className="page-heading">
        <p className="eyebrow">Protected backend records</p>
        <h1>Registered user database</h1>
        <p>{database.rows.length} registered users, including {database.rows.length - spendingUsers} accounts with no purchases.</p>
      </div>
      <div className="database-table-wrap">
        <table className="database-table">
          <thead><tr><th>User</th><th>Email</th><th>Phone</th><th>Mcoins owned</th><th>USD paid</th><th>Marketplace spent</th><th>Purchases</th><th>Membership</th><th>Joined</th><th>Security</th></tr></thead>
          <tbody>
            {database.rows.map((row) => (
              <tr key={row.userId}>
                <td>{row.name}</td>
                <td>{row.email}</td>
                <td>{row.phone || <span className="muted">Not provided</span>}</td>
                <td className="amount-cell">{row.mcoins.toLocaleString()} <small>(${row.mcoinUsdEquivalent.toFixed(2)} USD)</small></td>
                <td className="amount-cell">{formatAmount(row.usdSpent, 'USD')}</td>
                <td>{row.marketplaceSpentMcoins.toLocaleString()} Mcoins <small>(${row.marketplaceSpentUsdEquivalent.toFixed(2)})</small></td>
                <td>{row.purchaseCount}</td>
                <td><span className="status-pill">{row.proStatus}</span></td>
                <td>{row.createdAt ? new Date(row.createdAt).toLocaleString() : '—'}</td>
                <td><button className="ghost compact-action" type="button" disabled={row.userId === user.user_id} onClick={() => setPasswordReset({ userId: row.userId, name: row.name, email: row.email, password: '', confirm: '' })}>{row.userId === user.user_id ? 'Your account' : 'Set password'}</button></td>
              </tr>
            ))}
            {!database.rows.length && !status && <tr><td colSpan="10">No registered users yet.</td></tr>}
          </tbody>
          <tfoot>
            <tr>
              <th colSpan="3">Totals · {database.footer.userCount || 0} users</th>
              <th colSpan="7">
                <span className="database-total">{Number(database.footer.totalMcoinsHeld || 0).toLocaleString()} Mcoins held (${Number(database.footer.totalMcoinsHeldUsdEquivalent || 0).toFixed(2)} USD equivalent)</span>
                <span className="database-total">Total USD revenue: {formatAmount(database.footer.totalUsdRevenue || 0, 'USD')}</span>
                <span className="database-total">Marketplace fees: {Number(database.footer.marketplaceFeesMcoins || 0).toLocaleString()} Mcoins (${Number(database.footer.marketplaceFeesUsdEquivalent || 0).toFixed(2)} equivalent)</span>
              </th>
            </tr>
          </tfoot>
        </table>
      </div>
      {passwordReset.userId && (
        <form className="temporary-password-card" onSubmit={resetPassword}>
          <strong>Set password for {passwordReset.name}</strong>
          <span>{passwordReset.email}</span>
          <label className="field">New password<input type="password" minLength="8" autoComplete="new-password" value={passwordReset.password} onChange={(event) => setPasswordReset({ ...passwordReset, password: event.target.value })} required /></label>
          <label className="field">Confirm password<input type="password" minLength="8" autoComplete="new-password" value={passwordReset.confirm} onChange={(event) => setPasswordReset({ ...passwordReset, confirm: event.target.value })} required /></label>
          <div>
            <button className="primary" type="submit">Save password</button>
            <button className="ghost" type="button" onClick={() => setPasswordReset({ userId: '', name: '', email: '', password: '', confirm: '' })}>Cancel</button>
          </div>
          <small>The user can sign in with this password immediately. Their existing sessions will be signed out.</small>
        </form>
      )}
      {status && <p className="form-status">{status}</p>}
      <p className="privacy-note">Private administrative data. “Total USD revenue” counts completed USD payment orders. Marketplace fees are shown separately because adding their Mcoin equivalent would double-count money already used to buy Mcoins.</p>
    </section>
  );
}
