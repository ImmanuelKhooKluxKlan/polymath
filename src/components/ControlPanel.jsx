import MusicChoiceDisclosure from './MusicChoiceDisclosure.jsx';
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
}) {
  return (
    <aside className="control-panel">
      <MusicChoiceDisclosure
        id="piano-available-songs"
        title="Choose available songs to play"
        summary={songLabel(song)}
        expanded={expanded}
        onToggle={onToggle}
      >
        <label className="field">
          Song
          <select value={song.title} onChange={(event) => onSongChange(event.target.value)}>
            {songs.map((candidate) => <option key={candidate.title} value={candidate.title}>{songLabel(candidate)}</option>)}
          </select>
        </label>

        <button className="primary song-play-now" type="button" onClick={onPlayNow}>
          Play now
        </button>
        <button className="ghost song-download" type="button" onClick={() => downloadSongJson(song)}>
          Download song
        </button>
      </MusicChoiceDisclosure>
    </aside>
  );
}
