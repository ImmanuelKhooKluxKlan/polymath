const LEARN_ITEMS = [
  ['studio', 'Piano'],
  ['guitar', 'Guitar'],
  ['ensemble', 'Instruments'],
];

export default function AppNav({ route, onNavigate, user }) {
  return (
    <nav className="app-nav" aria-label="Main navigation">
      <button className="brand-button" type="button" onClick={() => onNavigate('studio')} aria-label="Open Polymath Musician piano studio">
        <span className="brand-mark">PM</span>
        <span>Polymath Musician</span>
      </button>
      <div className="nav-links">
        {LEARN_ITEMS.map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={route === value ? 'active' : ''}
            onClick={() => onNavigate(value)}
          >
            {label}
          </button>
        ))}
        <button
          type="button"
          className={route === 'published-songs' ? 'active' : ''}
          onClick={() => onNavigate('published-songs')}
        >
          Marketplace
        </button>
        {user && (
          <button
            type="button"
            className={route === 'your-songs' ? 'active' : ''}
            onClick={() => onNavigate('your-songs')}
          >
            Your Songs
          </button>
        )}
        <button
          type="button"
          className={route === 'band' ? 'active' : ''}
          onClick={() => onNavigate('band')}
        >
          Band
        </button>
        <button
          type="button"
          className={route === 'account' ? 'active' : ''}
          onClick={() => onNavigate('account')}
        >
          {user ? 'Account' : 'Sign in'}
        </button>
      </div>
    </nav>
  );
}
