const PRIMARY_ITEMS = [
  ['studio', 'Piano'],
  ['guitar', 'Guitar'],
  ['ensemble', 'Instruments'],
  ['published-songs', 'Composers'],
];

export default function AppNav({ route, onNavigate, user }) {
  const moreItems = [
    ['find-teacher', 'Find Teacher'],
    ['band', 'Band'],
    ...(user ? [['your-songs', 'Your Songs']] : []),
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
        {PRIMARY_ITEMS.map(([value, label]) => (
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
            <button
              type='button'
              className={`mobile-nav-only ${route === 'published-songs' ? 'active' : ''}`}
              onClick={(event) => navigateFromMenu(event, 'published-songs')}
            >
              Composers
            </button>
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
