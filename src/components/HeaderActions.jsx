export default function HeaderActions({ user, onNavigate, route }) {
  const isLesson = ['studio', 'guitar', 'ensemble', 'band'].includes(route);
  const walletValue = user ? user.mcoins.toLocaleString() : 'Mcoins';
  const planLabel = user?.admin ? 'Admin' : user?.pro ? 'Pro' : 'Upgrade';
  const planDetail = user?.admin
    ? 'Unlimited'
    : user?.pro
      ? (user.translationAllowance?.remaining ?? 20) + ' left'
      : 'Get Pro';

  return (
    <div className={'global-actions ' + (isLesson ? 'lesson-actions' : '')}>
      <button
        className='mcoin-button header-status-button'
        type='button'
        onClick={() => user ? onNavigate('account') : onNavigate('payment', { productId: 'mcoins-100' })}
        aria-label={user ? 'Open wallet with ' + walletValue + ' Mcoins' : 'Learn about Mcoins'}
      >
        <span>Wallet</span>
        <strong>{walletValue}</strong>
      </button>
      <button
        className='buy-pro-button compact-pro header-status-button'
        type='button'
        onClick={() => user?.admin || user?.pro ? onNavigate('account') : onNavigate('payment', { productId: 'polymath-pro' })}
        aria-label={user?.admin || user?.pro ? 'Open ' + planLabel + ' account' : 'Upgrade to Pro'}
      >
        <span>{planLabel}</span>
        <strong>{planDetail}</strong>
      </button>
    </div>
  );
}
