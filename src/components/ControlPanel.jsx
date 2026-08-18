import { TONE_MODE_LABELS } from '../engine/audioEngine.js';

export default function ControlPanel({
  song,
  songs,
  onSongChange,
  leadTime,
  setLeadTime,
  toneMode,
  setToneMode,
  autoplayVolume,
  setAutoplayVolume,
}) {
  return (
    <aside className="control-panel">
      <div>
        <p className="eyebrow">Current song</p>
        <h2>{song.title}</h2>
        <p className="muted">{song.composer} • {song.bpm} BPM • {song.notes.length} notes • {song.pedals?.length || 0} pedal events</p>
      </div>

      <label className="field">
        Available songs
        <select value={song.title} onChange={(event) => onSongChange(event.target.value)}>
          {songs.map((candidate) => <option key={candidate.title} value={candidate.title}>{candidate.title}</option>)}
        </select>
      </label>

      <details className="lesson-options">
        <summary>Lesson and sound options</summary>
        <div className="lesson-options-content">
          <label className="field">
            Piano sound
            <select value={toneMode} onChange={(event) => setToneMode(event.target.value)}>
              <option value="pianella">{TONE_MODE_LABELS.pianella}</option>
              <option value="grand">{TONE_MODE_LABELS.grand}</option>
            </select>
          </label>
          <label className="field">
            Autoplay volume: {Math.round((autoplayVolume ?? 1) * 100)}%
            <input type="range" min="0.25" max="1.25" step="0.05" value={autoplayVolume ?? 1} onChange={(event) => setAutoplayVolume(Number(event.target.value))} />
          </label>
          <label className="field">
            Notes appear {leadTime.toFixed(1)}s early
            <input type="range" min="1.4" max="5.5" step="0.1" value={leadTime} onChange={(event) => setLeadTime(Number(event.target.value))} />
          </label>
        </div>
      </details>

    </aside>
  );
}
