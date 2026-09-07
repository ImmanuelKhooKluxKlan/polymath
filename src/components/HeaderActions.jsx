export default function HeaderActions({ user, onNavigate, route }) {
  const isLesson = ['studio', 'guitar', 'ensemble', 'band'].includes(route);
  const creatorSubscription = user?.activeSubscriptions?.find((item) => item.categorySlug === 'create-music');
  const hasPaidAccess = Boolean(user?.admin || user?.pro || creatorSubscription);
  const walletValue = user ? (user.unlimitedMcoins ? '∞' : user.mcoins.toLocaleString()) : 'Mcoins';
  const planLabel = user?.admin
    ? 'Admin'
    : user?.subscriptionTier === 'musician'
      ? 'Musician'
      : user?.subscriptionTier === 'chill'
        ? 'Chill'
        : creatorSubscription
          ? creatorSubscription.name || 'Creator'
        : 'Upgrade';
  const planDetail = user?.admin
    ? 'Unlimited'
    : user?.pro
      ? (user.translationAllowance?.remaining ?? 20) + ' left'
      : creatorSubscription
        ? 'Create Music'
      : 'See plans';

  return (
    <div className={'global-actions ' + (isLesson ? 'lesson-actions' : '')}>
      <button
        className='mcoin-button header-status-button'
        type='button'
        onClick={() => user ? onNavigate('account') : onNavigate('payment', { productId: 'mcoins-100' })}
        aria-label={user ? (user.unlimitedMcoins ? 'Open unlimited administrator wallet' : 'Open wallet with ' + walletValue + ' Mcoins') : 'Learn about Mcoins'}
      >
        <span>Wallet</span>
        <strong>{walletValue}</strong>
      </button>
      <button
        className='buy-pro-button compact-pro header-status-button'
        type='button'
        onClick={() => hasPaidAccess ? onNavigate('account') : onNavigate('payment', { productId: 'polymath-chill-monthly' })}
        aria-label={hasPaidAccess ? 'Open ' + planLabel + ' account' : 'See subscription plans'}
      >
        <span>{planLabel}</span>
        <strong>{planDetail}</strong>
      </button>
    </div>
  );
}
