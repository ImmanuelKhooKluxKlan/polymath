export default function HeaderActions({ user, onNavigate, route }) {
  const isLesson = ['studio', 'guitar', 'ensemble', 'band'].includes(route);
  return (
    <div className={`global-actions ${isLesson ? 'lesson-actions' : ''}`}>
      <button className="mcoin-button" type="button" onClick={() => onNavigate('payment', { productId: 'mcoins-100' })}>
        <span>{user ? 'Wallet' : 'Mcoins'}</span>
        <small>{user ? `${user.mcoins.toLocaleString()} balance` : '$1 = 10 Mcoins'}</small>
      </button>
      <button className="buy-pro-button compact-pro" type="button" onClick={() => user?.admin || user?.pro ? onNavigate('account') : onNavigate('payment', { productId: 'polymath-pro' })}>
        <span>{user?.admin ? 'Admin' : user?.pro ? 'Pro' : 'Upgrade'}</span>
        <small>{user?.admin ? 'Unlimited translations' : user?.pro ? `${user.translationAllowance?.remaining ?? 20} translations left` : '$19.99 / month'}</small>
      </button>
    </div>
  );
}
