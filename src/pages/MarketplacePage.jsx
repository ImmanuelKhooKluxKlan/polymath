import { useEffect, useMemo, useState } from 'react';
import { apiRequest, downloadProtectedFile, fileToBase64 } from '../services/api.js';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import InstrumentIcon from '../components/InstrumentIcon.jsx';
import ReputationScore from '../components/ReputationScore.jsx';

const ARTISTS = [
  'Taylor Swift', 'Adele', 'Ed Sheeran', 'Bruno Mars', 'Billie Eilish',
  'The Beatles', 'Traditional', 'Public Domain',
];

const FORMAT_DETAILS = {
  JSON: { label: 'Ready-to-Play JSON', extension: 'json' },
  MIDI: { label: 'Ready-to-Play MIDI', extension: 'mid' },
  MUSICXML: { label: 'MusicXML Score', extension: 'musicxml' },
  PDF: { label: 'PDF Music Sheet', extension: 'pdf' },
};

const MARKETPLACE_FEE_RATE = 0.25;

function marketplaceFee(price) {
  return Math.max(0, (Number(price) || 0) * MARKETPLACE_FEE_RATE);
}

function initials(name) {
  return String(name || 'Composer').split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
}

function StarRating({ value = 0, onChange = null, label = 'Rating' }) {
  const rounded = Math.round(Number(value) || 0);
  return (
    <div className={`star-rating ${onChange ? 'interactive' : ''}`} aria-label={`${label}: ${Number(value || 0).toFixed(1)} out of 5`}>
      {[1, 2, 3, 4, 5].map((star) => onChange ? (
        <button key={star} type='button' className={star <= rounded ? 'filled' : ''} aria-label={`${star} star${star === 1 ? '' : 's'}`} onClick={() => onChange(star)}>★</button>
      ) : <span key={star} className={star <= rounded ? 'filled' : ''} aria-hidden='true'>★</span>)}
    </div>
  );
}

function ComposerAvatar({ composer, size = 'normal' }) {
  return composer?.avatarUrl
    ? <img className={`composer-avatar ${size}`} src={composer.avatarUrl} alt={`${composer.name} profile`} />
    : <span className={`composer-avatar composer-avatar-fallback ${size}`} aria-hidden='true'>{initials(composer?.name)}</span>;
}

export default function MarketplacePage({ user, setUser, onNavigate }) {
  const [listings, setListings] = useState([]);
  const [filters, setFilters] = useState({ query: '', artist: '', format: '' });
  const [showPublish, setShowPublish] = useState(false);
  const [checkoutId, setCheckoutId] = useState('');
  const [promotionCode, setPromotionCode] = useState('');
  const [friendId, setFriendId] = useState('');
  const [openReviews, setOpenReviews] = useState('');
  const [reviewsByListing, setReviewsByListing] = useState({});
  const [reviewDrafts, setReviewDrafts] = useState({});
  const [composerProfile, setComposerProfile] = useState(null);
  const [composerLoading, setComposerLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [publish, setPublish] = useState({
    artistChoice: '', artist: '', title: '', instrument: 'piano', format: 'JSON',
    priceMcoins: 100, description: '', file: null, rightsConfirmed: false, feeConfirmed: false,
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
    const queryText = `${listing.artist} ${listing.title} ${listing.seller?.name}`.toLowerCase();
    return (!filters.query || queryText.includes(filters.query.toLowerCase()))
      && (!filters.artist || listing.artist === filters.artist)
      && (!filters.format || listing.format === filters.format);
  }), [listings, filters]);

  const feePreview = marketplaceFee(publish.priceMcoins);
  const sellerPreview = Math.max(0, Number(publish.priceMcoins || 0) - feePreview);

  function openCheckout(listing) {
    if (!user) {
      onNavigate('account', { next: 'published-songs' });
      return;
    }
    setCheckoutId((current) => current === listing.id ? '' : listing.id);
    setPromotionCode('');
    setFriendId('');
  }

  async function purchase(listing) {
    try {
      const data = await apiRequest(`/api/listings/${listing.id}/purchase`, {
        method: 'POST',
        body: JSON.stringify({ promotionCode, friendId }),
      });
      setUser(data.user);
      setCheckoutId('');
      setPromotionCode('');
      setFriendId('');
      setStatus(`Purchased “${listing.title}”.`);
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

  async function toggleReviews(listing) {
    if (openReviews === listing.id) {
      setOpenReviews('');
      return;
    }
    setOpenReviews(listing.id);
    try {
      const data = await apiRequest(`/api/listings/${listing.id}/reviews`);
      setReviewsByListing((current) => ({ ...current, [listing.id]: data.reviews }));
      const mine = data.reviews.find((review) => review.mine);
      if (mine) {
        setReviewDrafts((current) => ({ ...current, [listing.id]: { rating: mine.rating, comment: mine.comment } }));
      }
    } catch (error) {
      setStatus(error.message);
    }
  }

  function updateReviewDraft(listingId, changes) {
    setReviewDrafts((current) => ({
      ...current,
      [listingId]: { rating: 5, comment: '', ...current[listingId], ...changes },
    }));
  }

  async function submitReview(event, listing) {
    event.preventDefault();
    if (!user) {
      onNavigate('account', { next: 'published-songs' });
      return;
    }
    const draft = { rating: 5, comment: '', ...reviewDrafts[listing.id] };
    try {
      const data = await apiRequest(`/api/listings/${listing.id}/reviews`, {
        method: 'POST',
        body: JSON.stringify(draft),
      });
      const reviews = reviewsByListing[listing.id] || [];
      const nextReviews = reviews.some((review) => review.id === data.review.id)
        ? reviews.map((review) => review.id === data.review.id ? data.review : review)
        : [data.review, ...reviews];
      setReviewsByListing((current) => ({ ...current, [listing.id]: nextReviews }));
      setListings((current) => current.map((item) => item.id === listing.id
        ? { ...item, reviewSummary: data.summary }
        : item));
      setStatus('Your verified review is public.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function openComposer(composerId) {
    if (!composerId) return;
    setComposerLoading(true);
    try {
      const data = await apiRequest(`/api/composers/${composerId}`);
      setComposerProfile(data);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setComposerLoading(false);
    }
  }

  async function toggleFollow() {
    if (!composerProfile?.composer) return;
    if (!user) {
      onNavigate('account', { next: 'published-songs' });
      return;
    }
    const composer = composerProfile.composer;
    try {
      const data = await apiRequest(`/api/composers/${composer.user_id}/follow`, {
        method: composer.isFollowing ? 'DELETE' : 'POST',
      });
      setComposerProfile((current) => ({ ...current, composer: data.composer }));
      setListings((current) => current.map((listing) => listing.seller?.user_id === data.composer.user_id
        ? { ...listing, seller: { ...listing.seller, ...data.composer } }
        : listing));
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
      setStatus('Attach a JSON, MIDI, MusicXML, or PDF music sheet.');
      return;
    }
    setStatus('Publishing music sheet...');
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
      setStatus('Music sheet published.');
      setPublish({
        artistChoice: '', artist: '', title: '', instrument: 'piano', format: 'JSON',
        priceMcoins: 100, description: '', file: null, rightsConfirmed: false, feeConfirmed: false,
      });
      await loadListings();
    } catch (error) {
      setStatus(error.message);
    }
  }

  function focusListing(listingId) {
    setComposerProfile(null);
    window.setTimeout(() => document.getElementById(`composer-listing-${listingId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 0);
  }

  return (
    <section className='page-shell marketplace-page composers-page'>
      <header className='composers-heading'>
        <h1>Composers</h1>
        <button className='primary publish-button' type='button' onClick={() => setShowPublish((value) => !value)}>Sell a music sheet</button>
      </header>

      {showPublish && (
        <form className='publish-panel' onSubmit={publishListing}>
          <div className='publish-panel-heading'>
            <div><p className='eyebrow'>For sellers</p><h2>Publish a music sheet</h2><p>Buyers will see the file type, price, composer profile, and public reviews.</p></div>
            <button className='ghost' type='button' onClick={() => setShowPublish(false)}>Close</button>
          </div>
          <div className='artist-chip-row'>
            {ARTISTS.map((artist) => (
              <button key={artist} type='button' className={publish.artistChoice === artist ? 'selected' : ''} onClick={() => setPublish({ ...publish, artistChoice: artist, artist })}>{artist}</button>
            ))}
          </div>
          <div className='publish-form-grid'>
            <label className='field'>Artist name<input placeholder='Enter artist name' value={publish.artist} onChange={(event) => setPublish({ ...publish, artist: event.target.value, artistChoice: '' })} required /></label>
            <label className='field'>Song title<input placeholder='Song title' value={publish.title} onChange={(event) => setPublish({ ...publish, title: event.target.value })} required /></label>
            <label className='field'>Instrument<select value={publish.instrument} onChange={(event) => setPublish({ ...publish, instrument: event.target.value })}>{INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}</select></label>
            <label className='field'>File type<select value={publish.format} onChange={(event) => setPublish({ ...publish, format: event.target.value })}>{Object.entries(FORMAT_DETAILS).map(([value, detail]) => <option key={value} value={value}>{detail.label}</option>)}</select></label>
            <label className='field'>Price in Mcoins<input type='number' min='10' max='100000' step='10' value={publish.priceMcoins} onChange={(event) => setPublish({ ...publish, priceMcoins: event.target.value })} required /></label>
            <label className='field'>Music sheet file<input type='file' accept='.json,.pdf,.mid,.midi,.musicxml,.xml' onChange={(event) => setPublish({ ...publish, file: event.target.files?.[0] || null })} required /></label>
          </div>
          <label className='field'>Description<textarea rows='3' placeholder='Describe the arrangement.' value={publish.description} onChange={(event) => setPublish({ ...publish, description: event.target.value })} /></label>
          <div className='seller-earnings-preview'>
            <div><span>Listing price</span><strong>{Number(publish.priceMcoins || 0).toLocaleString()} Mcoins</strong></div>
            <div><span>Polymath fee (25%)</span><strong>−{feePreview.toLocaleString()} Mcoins</strong></div>
            <div><span>You receive</span><strong>{sellerPreview.toLocaleString()} Mcoins</strong></div>
          </div>
          <label className='rights-check'><input type='checkbox' checked={publish.rightsConfirmed} onChange={(event) => setPublish({ ...publish, rightsConfirmed: event.target.checked })} required /><span>I own the rights to this file or have permission to sell it.</span></label>
          <label className='rights-check'><input type='checkbox' checked={publish.feeConfirmed} onChange={(event) => setPublish({ ...publish, feeConfirmed: event.target.checked })} required /><span>I accept the 25% sale fee.</span></label>
          <button className='primary' type='submit'>Publish</button>
        </form>
      )}

      <div className='composer-browser-bar'>
        <input aria-label='Search songs, artists, or composers' placeholder='Search songs or composers' value={filters.query} onChange={(event) => setFilters({ ...filters, query: event.target.value })} />
        <select aria-label='Filter by artist' value={filters.artist} onChange={(event) => setFilters({ ...filters, artist: event.target.value })}><option value=''>All artists</option>{artists.map((artist) => <option key={artist}>{artist}</option>)}</select>
        <select aria-label='Filter by file type' value={filters.format} onChange={(event) => setFilters({ ...filters, format: event.target.value })}><option value=''>All types</option>{Object.entries(FORMAT_DETAILS).map(([value, detail]) => <option key={value} value={value}>{detail.label}</option>)}</select>
      </div>

      {composerProfile && (
        <section className='composer-profile-panel'>
          <button className='composer-profile-close' type='button' aria-label='Close composer profile' onClick={() => setComposerProfile(null)}>×</button>
          <div className='composer-profile-identity'>
            <ComposerAvatar composer={composerProfile.composer} size='large' />
            <div><p className='eyebrow'>Composer</p><h2>{composerProfile.composer.name}</h2><StarRating value={composerProfile.composer.averageRating} label={`${composerProfile.composer.name} average rating`} /></div>
          </div>
          <div className='composer-profile-score'>
            <ReputationScore ranking={composerProfile.composer.ranking} audienceLabel='buyers' />
            <div className='composer-profile-stats'>
              <span><strong>{composerProfile.composer.buyerCount}</strong><small>unique buyers</small></span>
              <span><strong>{composerProfile.composer.ratingCount}</strong><small>ratings</small></span>
              <span><strong>{composerProfile.composer.followerCount}</strong><small>followers</small></span>
              <span><strong>{composerProfile.composer.publishedCount}</strong><small>sheets</small></span>
            </div>
          </div>
          <div className='composer-profile-actions'>
            {!composerProfile.composer.isSelf && <button className={composerProfile.composer.isFollowing ? 'ghost following' : 'primary'} type='button' onClick={toggleFollow}>{composerProfile.composer.isFollowing ? 'Following' : 'Follow'}</button>}
            {!composerProfile.composer.isSelf && user && <button className='ghost' type='button' onClick={() => onNavigate('messages', { userId: composerProfile.composer.user_id, name: composerProfile.composer.name })}>Chat</button>}
          </div>
          <div className='composer-profile-songs'>
            {composerProfile.listings.map((listing) => (
              <button key={listing.id} type='button' onClick={() => focusListing(listing.id)}>
                <span><strong>{listing.title}</strong><small>{listing.artist} · {FORMAT_DETAILS[listing.format]?.label || listing.format}</small></span>
                <b>{listing.priceMcoins.toLocaleString()} Mcoins</b>
              </button>
            ))}
          </div>
        </section>
      )}

      {composerLoading && <p className='composer-loading'>Opening composer...</p>}

      <div className='listing-grid composer-listing-grid'>
        {filtered.map((listing) => {
          const format = FORMAT_DETAILS[listing.format] || { label: listing.format, extension: listing.format.toLowerCase() };
          const instrument = INSTRUMENT_BY_ID[listing.instrument] || { id: 'music', label: listing.instrument };
          const reviews = reviewsByListing[listing.id] || [];
          const draft = { rating: 5, comment: '', ...reviewDrafts[listing.id] };
          return (
            <article className='listing-card composer-listing-card' id={`composer-listing-${listing.id}`} key={listing.id}>
              <div className={`listing-cover ${listing.instrument}`}><InstrumentIcon instrument={instrument.id} size='lg' /></div>
              <div className='listing-body'>
                <div className='listing-topline'><span className='format-badge'>{format.label}</span><span>{instrument.label}</span></div>
                <p className='listing-artist'>{listing.artist}</p>
                <h2>{listing.title}</h2>
                {listing.description && <p className='listing-description'>{listing.description}</p>}
                <button className='listing-rating' type='button' onClick={() => toggleReviews(listing)}>
                  <StarRating value={listing.reviewSummary?.averageRating || 0} label={`${listing.title} rating`} />
                  <span>{listing.reviewSummary?.reviewCount ? `${listing.reviewSummary.averageRating} (${listing.reviewSummary.reviewCount})` : 'No reviews'}</span>
                </button>
                <ReputationScore compact ranking={listing.seller?.ranking} audienceLabel='buyers' />
                <button className='composer-link' type='button' onClick={() => openComposer(listing.seller?.user_id)}>
                  <ComposerAvatar composer={listing.seller} size='small' />
                  <span><small>Composer</small><strong>{listing.seller?.name}</strong></span>
                </button>
              </div>
              <footer className='listing-footer'>
                <strong>{listing.priceMcoins.toLocaleString()} Mcoins</strong>
                <div className='listing-actions'>
                  {listing.purchased || listing.owned
                    ? <button className='primary' type='button' onClick={() => download(listing)}>Download</button>
                    : <button className='primary' type='button' onClick={() => openCheckout(listing)}>Buy</button>}
                  <button className='ghost' type='button' onClick={() => toggleReviews(listing)}>Reviews</button>
                  {listing.owned && <button className='ghost' type='button' onClick={() => onNavigate('your-songs')}>Manage</button>}
                </div>
              </footer>

              {checkoutId === listing.id && (
                <div className='listing-checkout'>
                  <input aria-label='Coupon code' value={promotionCode} maxLength='32' placeholder='Coupon (optional)' disabled={Boolean(friendId)} onChange={(event) => setPromotionCode(event.target.value.toUpperCase())} />
                  <input aria-label='Friend ID' value={friendId} maxLength='10' placeholder='Friend ID (optional)' disabled={Boolean(promotionCode)} onChange={(event) => setFriendId(event.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '').slice(0, 10))} />
                  <button className='primary' type='button' onClick={() => purchase(listing)}>Confirm {listing.priceMcoins.toLocaleString()} Mcoins</button>
                  <button className='ghost' type='button' onClick={() => setCheckoutId('')}>Cancel</button>
                </div>
              )}

              {openReviews === listing.id && (
                <section className='listing-reviews'>
                  {(listing.purchased && !listing.owned) && (
                    <form className='review-form' onSubmit={(event) => submitReview(event, listing)}>
                      <StarRating value={draft.rating} onChange={(rating) => updateReviewDraft(listing.id, { rating })} label='Your rating' />
                      <textarea rows='3' maxLength='1000' placeholder='Leave an honest review' value={draft.comment} onChange={(event) => updateReviewDraft(listing.id, { comment: event.target.value })} required />
                      <button className='primary' type='submit'>Post review</button>
                      <small>Verified reviews are permanent. Composers cannot remove them.</small>
                    </form>
                  )}
                  <div className='review-list'>
                    {reviews.map((review) => (
                      <article className='review-card' key={review.id}>
                        <div className='review-author'><ComposerAvatar composer={review.author} size='small' /><span><strong>{review.author.name}</strong>{review.verifiedPurchase && <small>Verified buyer</small>}</span><StarRating value={review.rating} label={`${review.author.name} rating`} /></div>
                        <p>{review.comment}</p>
                        <time dateTime={review.updatedAt || review.createdAt}>{new Date(review.updatedAt || review.createdAt).toLocaleDateString()}</time>
                      </article>
                    ))}
                    {!reviews.length && <p className='no-reviews'>No reviews yet.</p>}
                  </div>
                </section>
              )}
            </article>
          );
        })}
      </div>
      {!filtered.length && <div className='empty-state'>No music sheets found.</div>}
      {status && <p className='form-status floating-status'>{status}</p>}
    </section>
  );
}
