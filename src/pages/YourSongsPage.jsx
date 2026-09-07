import { useCallback, useEffect, useState } from 'react';
import { apiRequest, downloadProtectedFile } from '../services/api.js';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import InstrumentIcon from '../components/InstrumentIcon.jsx';

const EXTENSIONS = {
  JSON: 'json',
  MIDI: 'mid',
  MUSICXML: 'musicxml',
  PDF: 'pdf',
};

function listingOfferLabel(listing) {
  if (listing.listingMode === 'listener-reward') return `Pays ${Number(listing.listenerRewardMcoins || 0).toLocaleString()} Mcoins per listener`;
  if (listing.listingMode === 'free') return 'Free';
  return `${Number(listing.priceMcoins || 0).toLocaleString()} Mcoins`;
}

function EditableListing({ listing, policies, onSaved }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({
    artist: listing.artist,
    title: listing.title,
    instrument: listing.instrument,
    listingMode: listing.listingMode || 'sale',
    priceMcoins: listing.priceMcoins,
    listenerRewardMcoins: listing.listenerRewardMcoins || 1,
    description: listing.description || '',
  });
  const [status, setStatus] = useState('');

  async function save(event) {
    event.preventDefault();
    setStatus('Saving changes...');
    try {
      await apiRequest(`/api/listings/${listing.id}`, {
        method: 'PUT',
        body: JSON.stringify({ ...form, priceMcoins: Number(form.priceMcoins), listenerRewardMcoins: Number(form.listenerRewardMcoins) }),
      });
      setEditing(false);
      setStatus('Listing updated.');
      await onSaved();
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (editing) {
    return (
      <form className="library-edit-form" onSubmit={save}>
        <label className="field">Song title<input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} required /></label>
        <label className="field">Artist<input value={form.artist} onChange={(event) => setForm({ ...form, artist: event.target.value })} required /></label>
        <label className="field">Instrument
          <select value={form.instrument} onChange={(event) => setForm({ ...form, instrument: event.target.value })}>
            {INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}
          </select>
        </label>
        <label className="field">Listing type<select value={form.listingMode} onChange={(event) => setForm({ ...form, listingMode: event.target.value })}><option value="sale">Sell</option><option value="free">Free</option>{policies.listenerRewardsEnabled && <option value="listener-reward">Reward listeners</option>}</select></label>
        {form.listingMode === 'sale' && <label className="field">Price in Mcoins<input type="number" min={policies.minimumMarketplacePriceMcoins || 0} max={policies.maximumMarketplacePriceMcoins || undefined} step="0.01" value={form.priceMcoins} onChange={(event) => setForm({ ...form, priceMcoins: event.target.value })} required /></label>}
        {form.listingMode === 'listener-reward' && <label className="field">Pay each listener<input type="number" min="0.01" max={policies.maximumListenerRewardMcoins || undefined} step="0.01" value={form.listenerRewardMcoins} onChange={(event) => setForm({ ...form, listenerRewardMcoins: event.target.value })} required /></label>}
        <label className="field library-description-field">Description<textarea rows="3" value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} /></label>
        <div className="button-row">
          <button className="primary" type="submit">Save changes</button>
          <button className="ghost" type="button" onClick={() => setEditing(false)}>Cancel</button>
        </div>
        {status && <p className="form-status">{status}</p>}
      </form>
    );
  }

  return (
    <article className="seller-song-row">
      <InstrumentIcon instrument={listing.instrument} size="sm" />
      <div>
        <strong>{listing.title}</strong>
        <span>{listing.artist} · {listing.format}</span>
      </div>
      <div className="seller-song-price">
        <strong>{listingOfferLabel(listing)}</strong>
        {listing.listingMode === 'listener-reward' && <small>{Number(listing.rewardPaidMcoins || 0).toLocaleString()} Mcoins paid so far</small>}
        <small>{listing.updatedAt ? `Updated ${new Date(listing.updatedAt).toLocaleDateString()}` : 'Published listing'}</small>
      </div>
      <button className="ghost" type="button" onClick={() => setEditing(true)}>Amend listing</button>
      {status && <p className="form-status">{status}</p>}
    </article>
  );
}

export default function YourSongsPage({ user, onNavigate }) {
  const [library, setLibrary] = useState({ personalSongs: [], purchasedSongs: [], sellingSongs: [] });
  const [policies, setPolicies] = useState({ minimumMarketplacePriceMcoins: 0, maximumMarketplacePriceMcoins: 100000, listenerRewardsEnabled: true, maximumListenerRewardMcoins: 100 });
  const [status, setStatus] = useState('');

  const loadLibrary = useCallback(async () => {
    if (!user) return;
    try {
      const data = await apiRequest('/api/library');
      setLibrary({
        personalSongs: data.personalSongs || [],
        purchasedSongs: data.purchasedSongs || [],
        sellingSongs: data.sellingSongs || [],
      });
      setStatus('');
    } catch (error) {
      setStatus(error.message);
    }
  }, [user]);

  useEffect(() => { loadLibrary(); }, [loadLibrary]);
  useEffect(() => {
    apiRequest('/api/catalog').then((data) => setPolicies((current) => ({ ...current, ...(data.policies || {}) }))).catch(() => {});
  }, []);

  async function download(song) {
    try {
      await downloadProtectedFile(`/api/listings/${song.id}/download`, `${song.title}.${EXTENSIONS[song.format] || song.format.toLowerCase()}`);
      setStatus(`Downloaded "${song.title}". It remains saved in Your Songs.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function downloadPersonalSong(song) {
    try {
      await downloadProtectedFile(`/api/personal-songs/${song.id}/download`, song.filename || `${song.title}.json`);
      setStatus(`Downloaded "${song.title}".`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function deletePersonalSong(song) {
    if (!window.confirm(`Remove "${song.title}" from your cloud songs?`)) return;
    try {
      await apiRequest(`/api/personal-songs/${song.id}`, { method: 'DELETE' });
      setLibrary((current) => ({
        ...current,
        personalSongs: current.personalSongs.filter((candidate) => candidate.id !== song.id),
      }));
      setStatus(`Removed "${song.title}" from your cloud songs.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (!user) {
    return (
      <section className="page-shell narrow-page">
        <div className="empty-state">
          <h1>Sign in to open Your Songs.</h1>
          <p>Your purchased music and published sheets are saved to your account.</p>
          <button className="primary" type="button" onClick={() => onNavigate('account')}>Sign in</button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-shell library-page">
      <div className="page-heading library-heading">
        <div>
          <p className="eyebrow">Your music library</p>
          <h1>Your Songs</h1>
          <p>Music you buy stays attached to your account, ready to download again whenever you need it.</p>
        </div>
        <button className="primary" type="button" onClick={() => onNavigate('published-songs')}>Browse composers</button>
      </div>

      <section className="library-section">
        <div className="section-title-row">
          <div><p className="eyebrow">Private account storage</p><h2>Your uploaded songs</h2></div>
          <span className="library-count">{library.personalSongs.length}</span>
        </div>
        <div className="playlist-table">
          {library.personalSongs.map((song, index) => {
            const instrument = INSTRUMENT_BY_ID[song.instrument] || { label: song.instrument };
            return (
              <article className="playlist-row personal-song-row" key={song.id}>
                <span className="playlist-number">{String(index + 1).padStart(2, '0')}</span>
                <InstrumentIcon instrument={song.instrument} size="sm" />
                <div className="playlist-title"><strong>{song.title}</strong><span>{song.artist || 'Unknown artist'}</span></div>
                <span>{instrument.label}</span>
                <span>{song.format}</span>
                <span>{new Date(song.createdAt).toLocaleDateString()}</span>
                <div className="personal-song-actions">
                  <button className="primary" type="button" onClick={() => downloadPersonalSong(song)}>Download</button>
                  <button className="ghost" type="button" onClick={() => deletePersonalSong(song)}>Remove</button>
                </div>
              </article>
            );
          })}
          {!library.personalSongs.length && <div className="empty-state">Ready-to-play songs you upload will be saved here.</div>}
        </div>
      </section>

      <section className="library-section">
        <div className="section-title-row">
          <div><p className="eyebrow">Purchased playlist</p><h2>Bought songs</h2></div>
          <span className="library-count">{library.purchasedSongs.length}</span>
        </div>
        <div className="playlist-table">
          {library.purchasedSongs.map((song, index) => {
            const instrument = INSTRUMENT_BY_ID[song.instrument] || { label: song.instrument };
            return (
              <article className="playlist-row" key={song.id}>
                <span className="playlist-number">{String(index + 1).padStart(2, '0')}</span>
                <InstrumentIcon instrument={song.instrument} size="sm" />
                <div className="playlist-title"><strong>{song.title}</strong><span>{song.artist}</span></div>
                <span>{instrument.label}</span>
                <span>{song.format}</span>
                <span>{new Date(song.purchasedAt).toLocaleDateString()}</span>
                <button className="primary" type="button" onClick={() => download(song)}>Download</button>
              </article>
            );
          })}
          {!library.purchasedSongs.length && <div className="empty-state">Songs you purchase will appear here automatically.</div>}
        </div>
      </section>

      <section className="library-section">
        <div className="section-title-row">
          <div><p className="eyebrow">Composer profile</p><h2>Your published music sheets</h2></div>
          <span className="library-count">{library.sellingSongs.length}</span>
        </div>
        <div className="seller-song-list">
          {library.sellingSongs.map((listing) => <EditableListing key={listing.id} listing={listing} policies={policies} onSaved={loadLibrary} />)}
          {!library.sellingSongs.length && (
            <div className="empty-state">
              <p>You have not published any songs yet.</p>
              <button className="ghost" type="button" onClick={() => onNavigate('published-songs')}>Sell a music sheet</button>
            </div>
          )}
        </div>
      </section>
      {status && <p className="form-status floating-status">{status}</p>}
    </section>
  );
}
