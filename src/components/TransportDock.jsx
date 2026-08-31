import { useEffect, useRef, useState } from 'react';
import { getSongDuration } from '../engine/scheduler.js';

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

export default function TransportDock({
  song,
  isPlaying,
  onPlayPause,
  onStop,
  currentTime,
  speed,
  setSpeed,
  pedalDown,
  onPedalChange,
  onSeekStart,
  onSeekPreview,
  onSeekCommit,
  onRewind,
  onForward,
  showKeyNotes = true,
  onShowKeyNotesChange,
  minSpeed = 0.45,
}) {
  const duration = getSongDuration(song);
  const [scrubTime, setScrubTime] = useState(null);
  const scrubbingRef = useRef(false);
  const scrubTimeRef = useRef(0);

  const displayedTime = scrubTime === null
    ? currentTime
    : scrubTime;

  function beginScrub() {
    if (scrubbingRef.current) return;
    scrubbingRef.current = true;
    scrubTimeRef.current = displayedTime;
    setScrubTime(displayedTime);
    onSeekStart?.();
  }

  function previewScrub(value) {
    const next = Math.max(0, Math.min(duration, Number(value) || 0));
    scrubTimeRef.current = next;
    setScrubTime(next);
    onSeekPreview?.(next);
  }

  function finishScrub() {
    if (!scrubbingRef.current) return;
    scrubbingRef.current = false;
    const target = scrubTimeRef.current;
    setScrubTime(null);
    onSeekCommit?.(target);
  }

  useEffect(() => {
    function handlePointerUp() {
      finishScrub();
    }

    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);

    return () => {
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
    };
  });

  return (
    <section className="transport-dock" aria-label="Playback controls">
      <div className="transport-button-group">
        <button className="primary transport-main" type="button" onClick={onPlayPause}>
          {isPlaying ? 'Pause' : currentTime > 0 ? 'Resume' : 'Play'}
        </button>
        <button className="ghost" type="button" onClick={onStop}>Stop</button>
        <button className="ghost seek-jump" type="button" onClick={onRewind} aria-label="Rewind 10 seconds">↶ 10s</button>
        <button className="ghost seek-jump" type="button" onClick={onForward} aria-label="Forward 10 seconds">10s ↷</button>
        <button
          className={`ghost key-notes-toggle ${showKeyNotes ? 'active' : ''}`}
          type="button"
          aria-label={`${showKeyNotes ? 'Hide' : 'Show'} piano key notes`}
          aria-pressed={showKeyNotes}
          onClick={() => onShowKeyNotesChange?.(!showKeyNotes)}
        >
          <span className="key-notes-full">Key notes</span>
          <span className="key-notes-short" aria-hidden="true">Notes</span>
        </button>
      </div>

      <label className="dock-speed">
        <span>Speed {speed.toFixed(2)}×</span>
        <input
          type="range"
          min={minSpeed}
          max="1.75"
          step="0.05"
          value={speed}
          onChange={(event) => setSpeed(Number(event.target.value))}
        />
      </label>

      <button
        type="button"
        className={`pedal-button ${pedalDown ? 'active' : ''}`}
        onPointerDown={() => onPedalChange(true)}
        onPointerUp={() => onPedalChange(false)}
        onPointerCancel={() => onPedalChange(false)}
        onPointerLeave={(event) => {
          if (event.buttons) onPedalChange(false);
        }}
      >
        Sustain pedal
        <small>{pedalDown ? 'DOWN' : 'Space bar'}</small>
      </button>

      <label className="dock-progress">
        <span className="timeline-time">{formatTime(displayedTime)}</span>
        <input
          className="timeline-range"
          type="range"
          min="0"
          max={Math.max(duration, 0.01)}
          step="0.01"
          value={Math.min(displayedTime, duration)}
          onPointerDown={beginScrub}
          onKeyDown={beginScrub}
          onChange={(event) => previewScrub(event.target.value)}
          onKeyUp={finishScrub}
          onBlur={finishScrub}
          aria-label="Song position"
        />
        <span className="timeline-time end">{formatTime(duration)}</span>
      </label>
    </section>
  );
}
