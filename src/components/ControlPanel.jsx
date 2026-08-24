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
}) {
  return (
    <aside className="control-panel">
      <div>
        <p className="eyebrow">Available songs</p>
        <h2>{songLabel(song)}</h2>
      </div>

      <label className="field">
        Select song
        <select value={song.title} onChange={(event) => onSongChange(event.target.value)}>
          {songs.map((candidate) => <option key={candidate.title} value={candidate.title}>{songLabel(candidate)}</option>)}
        </select>
      </label>

      <button className="primary song-play-now" type="button" onClick={onPlayNow}>
        Play now
      </button>
    </aside>
  );
}
