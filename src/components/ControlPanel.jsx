import { useMemo } from 'react';
import MusicChoiceDisclosure from './MusicChoiceDisclosure.jsx';
import SearchableSongSelect from './SearchableSongSelect.jsx';
import { downloadSongJson } from '../utils/exporters.js';

function songLabel(song) {
  const artist = String(song?.artist || song?.composer || '').trim();
  const genericArtists = new Set(['Unknown composer', 'CSV/JSON import', 'MIDI import']);
  return artist && !genericArtists.has(artist) ? `${song.title} (${artist})` : song.title;
}

export default function ControlPanel({
  song,
  songs,
  onSongChange,
  onPlayNow,
  expanded,
  onToggle,
  personalSongs = [],
  featuredSongs = [],
  onPersonalSongChange,
  onFeaturedSongChange,
  loadingPersonalSongId = '',
  personalSongStatus = '',
}) {
  const choices = useMemo(() => [
    ...songs
      .filter((candidate) => !candidate.personalSongId && !candidate.featuredSongId)
      .map((candidate, index) => ({
        id: `local:${candidate.libraryId || `${candidate.title}:${index}`}`,
        label: songLabel(candidate),
        searchText: `${candidate.title} ${candidate.artist || candidate.composer || ''}`,
        song: candidate,
      })),
    ...featuredSongs
      .filter((candidate) => candidate.instrument === 'piano')
      .map((candidate) => ({
        id: `featured:${candidate.id}`,
        label: songLabel(candidate),
        searchText: `${candidate.title} ${candidate.artist || ''}`,
        featuredSong: candidate,
      })),
    ...personalSongs.map((candidate) => ({
      id: `personal:${candidate.id}`,
      label: songLabel(candidate),
      searchText: `${candidate.title} ${candidate.artist || ''}`,
      personalSong: candidate,
    })),
  ], [featuredSongs, personalSongs, songs]);
  const selectedValue = song.featuredSongId
    ? `featured:${song.featuredSongId}`
    : song.personalSongId
    ? `personal:${song.personalSongId}`
    : choices.find((choice) => choice.song === song || choice.song?.libraryId === song.libraryId)?.id || choices[0]?.id || '';

  function choose(value) {
    const choice = choices.find((candidate) => candidate.id === value);
    if (choice?.featuredSong) onFeaturedSongChange?.(choice.featuredSong);
    else if (choice?.personalSong) onPersonalSongChange?.(choice.personalSong);
    else if (choice?.song) onSongChange(choice.song.libraryId);
  }

  return (
    <aside className="control-panel">
      <MusicChoiceDisclosure
        id="piano-available-songs"
        title="Choose available songs to play"
        summary={songLabel(song)}
        expanded={expanded}
        onToggle={onToggle}
      >
        <SearchableSongSelect
          id="piano-song-library"
          choices={choices}
          value={selectedValue}
          onChange={choose}
          busy={Boolean(loadingPersonalSongId)}
        />
        {personalSongStatus && <small className="song-library-search-status">{personalSongStatus}</small>}

        <button className="primary song-play-now" type="button" onClick={onPlayNow} disabled={Boolean(loadingPersonalSongId)}>
          {loadingPersonalSongId ? 'Loading song…' : 'Play now'}
        </button>
        <button className="ghost song-download" type="button" onClick={() => downloadSongJson(song)}>
          Download song
        </button>
      </MusicChoiceDisclosure>
    </aside>
  );
}
