import { apiRequest, fetchProtectedFile } from './api.js';

export const FEATURED_SONGS_UPDATED_EVENT = 'polymath:featured-songs-updated';

export async function loadFeaturedSongCatalog() {
  const data = await apiRequest('/api/featured-songs', {
    globalProgress: false,
    cache: 'no-store',
  });
  return Array.isArray(data?.songs) ? data.songs : [];
}

export async function fetchFeaturedSongFile(song) {
  if (!song?.id) throw new Error('Choose an available song first.');
  return fetchProtectedFile(
    song.downloadPath || `/api/featured-songs/${encodeURIComponent(song.id)}/download`,
    song.filename || `${song.title || 'available-song'}.${String(song.format || 'json').toLowerCase()}`,
  );
}

export function announceFeaturedSongsUpdated() {
  window.dispatchEvent(new window.CustomEvent(FEATURED_SONGS_UPDATED_EVENT));
}
