export default function AppNav({ route, onNavigate, user }) {
  const primaryItems = [
    ['studio', 'Piano'],
    ...(user ? [['your-songs', 'My songs']] : []),
  ];
  const moreItems = [
    ['guitar', 'Guitar'],
    ['ensemble', 'Other instruments'],
    ['published-songs', 'Composers'],
    ['find-teacher', 'Find Teacher'],
    ['band', 'Band'],
    ['community', 'Community'],
    ...(user?.admin ? [['chat-boss', 'Chat Boss']] : []),
    ['account', user ? 'Account' : 'Sign in'],
  ];
  const moreIsActive = moreItems.some(([value]) => route === value)
    || route === 'messages'
    || route === 'admin-database'
    || route === 'payment';

  function navigateFromMenu(event, page) {
    event.currentTarget.closest('details')?.removeAttribute('open');
    onNavigate(page);
  }

  return (
    <nav className='app-nav' aria-label='Main navigation'>
      <button className='brand-button' type='button' onClick={() => onNavigate('studio')} aria-label='Open Polymath Musician piano studio'>
        <span className='brand-mark'>PM</span>
        <span className='brand-copy'>
          <strong>Polymath</strong>
          <small>Musician</small>
        </span>
      </button>
      <div className='nav-links'>
        {primaryItems.map(([value, label]) => (
          <button
            key={value}
            type='button'
            className={route === value ? 'active' : ''}
            onClick={() => onNavigate(value)}
          >
            {label}
          </button>
        ))}
        <details className='nav-more'>
          <summary className={moreIsActive ? 'active' : ''}>More</summary>
          <div className='nav-more-menu'>
            {moreItems.map(([value, label]) => (
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
      </div>
    </nav>
  );
}
