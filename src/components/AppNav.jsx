import { normalizePublicSiteConfiguration } from '../config/siteConfiguration.js';

function brandInitials(brand) {
  const parts = [brand.name, brand.suffix].filter(Boolean);
  return parts.map((part) => String(part).trim().charAt(0)).join('').slice(0, 2).toUpperCase() || 'PM';
}

export default function AppNav({
  route,
  onNavigate,
  user,
  focusedCampaign = false,
  siteConfiguration,
}) {
  const configuration = normalizePublicSiteConfiguration(siteConfiguration);
  const configuredItems = configuration.navigation
    .filter((item) => item.visible && (item.access !== 'signed-in' || user));
  const primaryItems = configuredItems.filter((item) => item.group === 'primary');
  const configuredMoreItems = configuredItems.filter((item) => item.group === 'more');
  const mobileOverflowItems = primaryItems.slice(2);
  const systemItems = [
    ...(user?.admin ? [['chat-boss', 'Chat Boss']] : []),
    ['account', user ? 'Account' : 'Sign in'],
  ];
  const moreIsActive = configuredMoreItems.some((item) => route === item.id)
    || mobileOverflowItems.some((item) => route === item.id)
    || systemItems.some(([value]) => route === value)
    || route === 'messages'
    || route === 'admin-database'
    || route === 'payment';

  function navigateFromMenu(event, page) {
    event.currentTarget.closest('details')?.removeAttribute('open');
    onNavigate(page);
  }

  return (
    <nav className={`app-nav ${focusedCampaign ? 'is-campaign-nav' : ''}`} aria-label='Main navigation'>
      <button
        className='brand-button'
        type='button'
        onClick={() => onNavigate('studio')}
        aria-label={`Open ${configuration.brand.name} ${configuration.brand.suffix} home`}
      >
        <span className='brand-mark'>{brandInitials(configuration.brand)}</span>
        <span className='brand-copy'>
          <strong>{configuration.brand.name}</strong>
          <small>{configuration.brand.suffix}</small>
        </span>
      </button>
      {!focusedCampaign && <div className='nav-links'>
        {primaryItems.map((item) => (
          <button
            key={item.id}
            type='button'
            className={`nav-primary-item ${route === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            {item.label}
          </button>
        ))}
        <details className='nav-more'>
          <summary className={moreIsActive ? 'active' : ''}>More</summary>
          <div className='nav-more-menu'>
            {mobileOverflowItems.map((item) => (
              <button
                key={`mobile-${item.id}`}
                type='button'
                className={`nav-mobile-overflow-item ${route === item.id ? 'active' : ''}`}
                onClick={(event) => navigateFromMenu(event, item.id)}
              >
                {item.label}
              </button>
            ))}
            {configuredMoreItems.map((item) => (
              <button
                key={item.id}
                type='button'
                className={route === item.id ? 'active' : ''}
                onClick={(event) => navigateFromMenu(event, item.id)}
              >
                {item.label}
              </button>
            ))}
            {systemItems.map(([value, label]) => (
              <button
                key={value}
                type='button'
                className={route === value ? 'active' : ''}
                onClick={(event) => navigateFromMenu(event, value)}
              >
                {label}
              </button>
            ))}
          </div>
        </details>
      </div>}
    </nav>
  );
}
