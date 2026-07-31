import { useEffect, useMemo, useState } from 'react';
import { apiRequest, downloadProtectedFile, fileToBase64 } from '../services/api.js';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import InstrumentIcon from '../components/InstrumentIcon.jsx';

const ARTISTS = [
  'Taylor Swift',
  'Adele',
  'Ed Sheeran',
  'Bruno Mars',
  'Billie Eilish',
  'The Beatles',
  'Traditional',
  'Public Domain',
];

const FORMAT_DETAILS = {
  JSON: { label: 'Ready-to-Play JSON', playable: true, extension: 'json' },
  MIDI: { label: 'Ready-to-Play MIDI', playable: true, extension: 'mid' },
  MUSICXML: { label: 'MusicXML Score', playable: true, extension: 'musicxml' },
  PDF: { label: 'PDF Music Sheet', playable: false, extension: 'pdf' },
};

function marketplaceFee(price) {
  return Math.max(0, (Number(price) || 0) / 10);
}

export default function MarketplacePage({ user, setUser, onNavigate }) {
  const [listings, setListings] = useState([]);
  const [filters, setFilters] = useState({ query: '', artist: '', instrument: '', format: '' });
  const [showPublish, setShowPublish] = useState(false);
  const [status, setStatus] = useState('');
  const [publish, setPublish] = useState({
    artistChoice: '',
    artist: '',
    title: '',
    instrument: 'piano',
    format: 'JSON',
    priceMcoins: 100,
    description: '',
    file: null,
    rightsConfirmed: false,
    feeConfirmed: false,
  });

  async function loadListings() {
    try {
      const data = await apiRequest('/api/listings');
      setListings(data.listings);
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadListings(); }, [user?.user_id]);

  const artists = useMemo(() => [...new Set(listings.map((item) => item.artist))].sort(), [listings]);
  const filtered = useMemo(() => listings.filter((listing) => {
    const text = `${listing.artist} ${listing.title}`.toLowerCase();
    return (!filters.query || text.includes(filters.query.toLowerCase()))
      && (!filters.artist || listing.artist === filters.artist)
      && (!filters.instrument || listing.instrument === filters.instrument)
      && (!filters.format || listing.format === filters.format);
  }), [listings, filters]);

  const feePreview = marketplaceFee(publish.priceMcoins);
  const sellerPreview = Math.max(0, Number(publish.priceMcoins || 0) - feePreview);

  async function purchase(listing) {
    if (!user) {
      onNavigate('account');
      return;
    }
    try {
      const data = await apiRequest(`/api/listings/${listing.id}/purchase`, { method: 'POST' });
      setUser(data.user);
      setStatus(`Purchased “${listing.title}”. The ${FORMAT_DETAILS[listing.format]?.label || listing.format} is now available to download.`);
      await loadListings();
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function download(listing) {
    try {
      const format = FORMAT_DETAILS[listing.format] || { extension: listing.format.toLowerCase() };
      await downloadProtectedFile(`/api/listings/${listing.id}/download`, `${listing.title}.${format.extension}`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function publishListing(event) {
    event.preventDefault();
    if (!user) {
      setStatus('Sign in before publishing.');
      return;
    }
    if (!publish.file) {
      setStatus('Attach a ready-to-play JSON/MIDI/MusicXML file or a PDF music sheet.');
      return;
    }
    setStatus('Publishing music sheet…');
    try {
      const contentBase64 = await fileToBase64(publish.file);
      await apiRequest('/api/listings', {
        method: 'POST',
        body: JSON.stringify({
          artist: publish.artist,
          title: publish.title,
          instrument: publish.instrument,
          format: publish.format,
          priceMcoins: Number(publish.priceMcoins),
          description: publish.description,
          filename: publish.file.name,
          contentBase64,
          rightsConfirmed: publish.rightsConfirmed,
          feeConfirmed: publish.feeConfirmed,
        }),
      });
      setShowPublish(false);
      setStatus('Music sheet published successfully. Buyers can clearly see its file type before purchase.');
      setPublish({
        artistChoice: '', artist: '', title: '', instrument: 'piano', format: 'JSON', priceMcoins: 100,
        description: '', file: null, rightsConfirmed: false, feeConfirmed: false,
      });
      await loadListings();
    } catch (error) {
      setStatus(error.message);
    }
  }

  return (
    <section className="page-shell marketplace-page">
      <div className="page-heading marketplace-heading">
        <div>
          <p className="eyebrow">Music sheet marketplace</p>
          <h1>Buy playable sheets and readable scores.</h1>
          <p>Every listing shows its instrument and exact file type before payment. Ready-to-play files open in the studio; PDFs may require translation.</p>
        </div>
        <button className="primary publish-button" type="button" onClick={() => setShowPublish((value) => !value)}>Sell a music sheet</button>
      </div>

      <div className="market-trust-strip">
        <span><strong>$1 = 10 Mcoins</strong><small>Clear USD-backed wallet value</small></span>
        <span><strong>10% sale fee</strong><small>Sellers receive 90% of every sale</small></span>
        <span><strong>File type shown</strong><small>Know whether you are buying JSON, MIDI, or PDF</small></span>
      </div>

      {showPublish && (
        <form className="publish-panel" onSubmit={publishListing}>
          <div className="publish-panel-heading">
            <div><p className="eyebrow">Seller listing</p><h2>Publish a music sheet</h2></div>
            <button className="ghost" type="button" onClick={() => setShowPublish(false)}>Close</button>
          </div>
          <div className="artist-chip-row">
            {ARTISTS.map((artist) => (
              <button
                key={artist}
                type="button"
                className={publish.artistChoice === artist ? 'selected' : ''}
                onClick={() => setPublish({ ...publish, artistChoice: artist, artist })}
              >
                {artist}
              </button>
            ))}
          </div>
          <div className="publish-form-grid">
            <label className="field">Artist name<input placeholder="Enter artist name" value={publish.artist} onChange={(event) => setPublish({ ...publish, artist: event.target.value, artistChoice: '' })} required /></label>
            <label className="field">Song title<input placeholder="Song title" value={publish.title} onChange={(event) => setPublish({ ...publish, title: event.target.value })} required /></label>
            <label className="field">Instrument
              <select value={publish.instrument} onChange={(event) => setPublish({ ...publish, instrument: event.target.value })}>
                {INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}
              </select>
            </label>
            <label className="field">File type
              <select value={publish.format} onChange={(event) => setPublish({ ...publish, format: event.target.value })}>
                {Object.entries(FORMAT_DETAILS).map(([value, detail]) => <option key={value} value={value}>{detail.label}</option>)}
              </select>
            </label>
            <label className="field">Price in Mcoins<input type="number" min="10" max="100000" step="10" value={publish.priceMcoins} onChange={(event) => setPublish({ ...publish, priceMcoins: event.target.value })} required /></label>
            <label className="field">Music sheet file<input type="file" accept=".json,.pdf,.mid,.midi,.musicxml,.xml" onChange={(event) => setPublish({ ...publish, file: event.target.files?.[0] || null })} required /></label>
          </div>
          <label className="field">Description<textarea rows="3" placeholder="Describe exactly what the buyer receives." value={publish.description} onChange={(event) => setPublish({ ...publish, description: event.target.value })} /></label>

          <div className="seller-earnings-preview">
            <div><span>Listing price</span><strong>{Number(publish.priceMcoins || 0).toLocaleString()} Mcoins</strong></div>
            <div><span>Polymath Musician fee (10%)</span><strong>−{feePreview.toLocaleString()} Mcoins</strong></div>
            <div><span>You receive</span><strong>{sellerPreview.toLocaleString()} Mcoins</strong></div>
          </div>

          <label className="rights-check"><input type="checkbox" checked={publish.rightsConfirmed} onChange={(event) => setPublish({ ...publish, rightsConfirmed: event.target.checked })} required /><span>I own the rights to this file or have permission to sell it.</span></label>
          <label className="rights-check"><input type="checkbox" checked={publish.feeConfirmed} onChange={(event) => setPublish({ ...publish, feeConfirmed: event.target.checked })} required /><span>I understand that Polymath Musician receives 10% of every successful sale and I receive 90%.</span></label>
          <button className="primary" type="submit">Publish listing</button>
        </form>
      )}

      <div className="instrument-picker instrument-picker-expanded" aria-label="Choose an instrument">
        <div><p className="eyebrow">Choose instrument</p><strong>What do you want to shop for?</strong></div>
        <button type="button" className={!filters.instrument ? 'active' : ''} onClick={() => setFilters({ ...filters, instrument: '' })}>All</button>
        {INSTRUMENTS.map((instrument) => (
          <button key={instrument.id} type="button" className={filters.instrument === instrument.id ? 'active' : ''} onClick={() => setFilters({ ...filters, instrument: instrument.id })}>
            <InstrumentIcon instrument={instrument.id} size="sm" /> {instrument.shortLabel}
          </button>
        ))}
      </div>

      <div className="market-toolbar">
        <label className="market-search"><span>Search</span><input placeholder="Artist or song title" value={filters.query} onChange={(event) => setFilters({ ...filters, query: event.target.value })} /></label>
        <label>Artist<select value={filters.artist} onChange={(event) => setFilters({ ...filters, artist: event.target.value })}><option value="">All artists</option>{artists.map((artist) => <option key={artist}>{artist}</option>)}</select></label>
        <label>Instrument<select value={filters.instrument} onChange={(event) => setFilters({ ...filters, instrument: event.target.value })}><option value="">All instruments</option>{INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}</select></label>
        <label>File type<select value={filters.format} onChange={(event) => setFilters({ ...filters, format: event.target.value })}><option value="">All file types</option>{Object.entries(FORMAT_DETAILS).map(([value, detail]) => <option key={value} value={value}>{detail.label}</option>)}</select></label>
      </div>

      <div className="listing-grid">
        {filtered.map((listing) => {
          const format = FORMAT_DETAILS[listing.format] || { label: listing.format, playable: false };
          const instrument = INSTRUMENT_BY_ID[listing.instrument] || { id: 'music', label: listing.instrument };
          return (
            <article className="listing-card" key={listing.id}>
              <div className={`listing-cover ${listing.instrument}`}><InstrumentIcon instrument={instrument.id} size="lg" /><small>{instrument.label}</small></div>
              <div className="listing-body">
                <div className={`format-badge ${format.playable ? 'ready' : 'pdf'}`}>{format.label}</div>
                <p className="listing-artist">{listing.artist}</p>
                <h2>{listing.title}</h2>
                <p className="muted">{listing.description}</p>
                <div className="buyer-file-notice">
                  <strong>{format.playable ? 'Ready to play' : 'Readable PDF sheet'}</strong>
                  <span>{format.playable ? 'Open this file in a supported Polymath Musician studio.' : 'This PDF may need a 30-Mcoin translation before automatic playback.'}</span>
                </div>
                <p className="seller-line">Sold by <strong>{listing.seller?.name}</strong></p>
              </div>
              <div className="listing-footer">
                <div><strong>{listing.priceMcoins.toLocaleString()} Mcoins</strong><small>${(listing.priceMcoins / 10).toFixed(2)} USD equivalent</small></div>
                <div className="listing-actions">
                  {listing.purchased || listing.owned ? (
                    <button className="primary" type="button" onClick={() => download(listing)}>Download {listing.format}</button>
                  ) : (
                    <button className="primary" type="button" onClick={() => purchase(listing)}>Buy now</button>
                  )}
                  {listing.owned && (
                    <button className="ghost" type="button" onClick={() => onNavigate('your-songs')}>Manage listing</button>
                  )}
                  {listing.seller?.user_id !== user?.user_id && (
                    <button className="ghost" type="button" onClick={() => onNavigate('messages', { userId: listing.seller?.user_id, name: listing.seller?.name })}>Chat</button>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
      {!filtered.length && <div className="empty-state">No music sheets match those filters.</div>}
      {status && <p className="form-status floating-status">{status}</p>}
    </section>
  );
}
