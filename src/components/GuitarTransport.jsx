import { useEffect, useRef, useState } from 'react';

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, '0')}`;
}

export default function GuitarTransport({
  duration,
  currentTime,
  isPlaying,
  speed,
  onSpeedChange,
  onPlayPause,
  onStop,
  onSeekStart,
  onSeekPreview,
  onSeekCommit,
  onRewind,
  onForward,
  minSpeed = 0.45,
}) {
  const [scrubTime, setScrubTime] = useState(null);
  const dragging = useRef(false);
  const valueRef = useRef(0);
  const shownTime = scrubTime === null ? currentTime : scrubTime;

  function begin() {
    if (dragging.current) return;
    dragging.current = true;
    valueRef.current = shownTime;
    setScrubTime(shownTime);
    onSeekStart?.();
  }

  function preview(value) {
    const next = Math.max(0, Math.min(duration, Number(value) || 0));
    valueRef.current = next;
    setScrubTime(next);
    onSeekPreview?.(next);
  }

  function finish() {
    if (!dragging.current) return;
    dragging.current = false;
    const next = valueRef.current;
    setScrubTime(null);
    onSeekCommit?.(next);
  }

  useEffect(() => {
    window.addEventListener('pointerup', finish);
    window.addEventListener('pointercancel', finish);
    return () => {
      window.removeEventListener('pointerup', finish);
      window.removeEventListener('pointercancel', finish);
    };
  });

  return (
    <section className="guitar-transport-dock" aria-label="Guitar playback controls">
      <div className="transport-button-group">
        <button className="primary" type="button" onClick={onPlayPause}>{isPlaying ? 'Pause' : currentTime > 0 ? 'Resume' : 'Play'}</button>
        <button className="ghost" type="button" onClick={onStop}>Stop</button>
        <button className="ghost seek-jump" type="button" onClick={onRewind}>↶ 10s</button>
        <button className="ghost seek-jump" type="button" onClick={onForward}>10s ↷</button>
      </div>

      <label className="dock-speed">
        <span>Speed {speed.toFixed(2)}×</span>
        <input type="range" min={minSpeed} max="1.75" step="0.05" value={speed} onChange={(event) => onSpeedChange(Number(event.target.value))} />
      </label>

      <label className="dock-progress">
        <span className="timeline-time">{formatTime(shownTime)}</span>
        <input
          className="timeline-range"
          type="range"
          min="0"
          max={Math.max(duration, 0.01)}
          step="0.01"
          value={Math.min(shownTime, duration)}
          onPointerDown={begin}
          onKeyDown={begin}
          onChange={(event) => preview(event.target.value)}
          onKeyUp={finish}
          onBlur={finish}
          aria-label="Guitar lesson position"
        />
        <span className="timeline-time end">{formatTime(duration)}</span>
      </label>
    </section>
  );
}
