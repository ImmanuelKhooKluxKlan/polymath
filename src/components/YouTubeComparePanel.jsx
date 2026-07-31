import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';

function cleanWords(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function buildVideoSearchQuery(source, instrument = 'piano') {
  const explicit = cleanWords(
    source?.youtubeSearchQuery ||
    source?.videoSearchQuery,
  );

  const suffix = instrument === 'guitar'
    ? 'guitar cover tutorial'
    : 'piano cover tutorial';

  if (explicit) {
    return explicit.toLowerCase().includes(instrument)
      ? explicit
      : `${explicit} ${suffix}`;
  }

  const folderStyle = cleanWords(
    source?.sourceFolderName ||
    source?.style ||
    source?.arrangement,
  );

  const artist = cleanWords(
    source?.artist ||
    source?.composer,
  );

  const title = cleanWords(
    source?.title ||
    source?.name,
  );

  return [folderStyle, artist, title, suffix]
    .filter(Boolean)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractYouTubeVideoId(value) {
  const text = String(value || '').trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(text)) return text;

  try {
    const url = new URL(text);

    if (url.hostname === 'youtu.be') {
      return url.pathname.split('/').filter(Boolean)[0] || '';
    }

    if (url.hostname.endsWith('youtube.com')) {
      if (url.searchParams.get('v')) return url.searchParams.get('v');

      const parts = url.pathname.split('/').filter(Boolean);
      const markerIndex = parts.findIndex((part) => ['embed', 'shorts', 'live'].includes(part));
      if (markerIndex >= 0) return parts[markerIndex + 1] || '';
    }
  } catch {
    return '';
  }

  return '';
}

export default function YouTubeComparePanel({
  source,
  instrument = 'piano',
  compact = false,
}) {
  const suggestedQuery = useMemo(
    () => buildVideoSearchQuery(source, instrument),
    [source, instrument],
  );

  const [query, setQuery] = useState(suggestedQuery);
  const [results, setResults] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);

  async function search(searchQuery = query) {
    const cleaned = cleanWords(searchQuery);
    if (!cleaned) return;

    setLoading(true);
    setStatus('Searching YouTube…');

    try {
      const data = await apiRequest(`/api/youtube/search?q=${encodeURIComponent(cleaned)}&instrument=${encodeURIComponent(instrument)}`);
      const videos = Array.isArray(data.videos) ? data.videos : [];
      setResults(videos);
      setSelectedVideoId((previous) => previous || videos[0]?.videoId || '');
      setStatus(videos.length ? '' : 'No embeddable videos were found for that search.');
    } catch (error) {
      setResults([]);
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setQuery(suggestedQuery);
    setSelectedVideoId('');

    if (!suggestedQuery) return undefined;

    const timer = window.setTimeout(() => {
      search(suggestedQuery);
    }, 300);

    return () => window.clearTimeout(timer);
  }, [suggestedQuery, instrument]);

  function loadUrl() {
    const id = extractYouTubeVideoId(videoUrl);
    if (!id) {
      setStatus('Paste a valid YouTube video link or 11-character video ID.');
      return;
    }

    setSelectedVideoId(id);
    setStatus('Video loaded.');
  }

  function openYouTubeSearch() {
    const url = `https://www.youtube.com/results?search_query=${encodeURIComponent(query || suggestedQuery)}`;
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  return (
    <section className={`youtube-compare ${compact ? 'compact' : ''}`}>
      <div className="youtube-heading">
        <div>
          <p className="eyebrow">Compare with video</p>
          <h3>YouTube reference</h3>
        </div>
        <span className="youtube-badge">▶</span>
      </div>

      <div className="youtube-player-shell">
        {selectedVideoId ? (
          <iframe
            src={`https://www.youtube-nocookie.com/embed/${selectedVideoId}?rel=0&playsinline=1`}
            title="YouTube comparison video"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        ) : (
          <div className="youtube-empty">
            <strong>Select a video</strong>
            <small>Search below or paste a YouTube link.</small>
          </div>
        )}
      </div>

      <form
        className="youtube-search-form"
        onSubmit={(event) => {
          event.preventDefault();
          search();
        }}
      >
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={`${instrument} video, tutorial, or performance`}
          aria-label="Search YouTube"
        />
        <button className="primary" type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      <div className="youtube-url-row">
        <input
          value={videoUrl}
          onChange={(event) => setVideoUrl(event.target.value)}
          placeholder="Paste YouTube link"
          aria-label="YouTube video URL"
        />
        <button className="ghost" type="button" onClick={loadUrl}>Load</button>
      </div>

      {results.length > 0 && (
        <div className="youtube-results" aria-label="YouTube search results">
          {results.slice(0, compact ? 4 : 6).map((video) => (
            <button
              key={video.videoId}
              type="button"
              className={selectedVideoId === video.videoId ? 'selected' : ''}
              onClick={() => setSelectedVideoId(video.videoId)}
            >
              {video.thumbnail && <img src={video.thumbnail} alt="" loading="lazy" />}
              <span>
                <strong>{video.title}</strong>
                <small>{video.channelTitle}</small>
              </span>
            </button>
          ))}
        </div>
      )}

      {status && <p className="youtube-status">{status}</p>}
      <button className="youtube-external" type="button" onClick={openYouTubeSearch}>
        Open this search on YouTube
      </button>
    </section>
  );
}
