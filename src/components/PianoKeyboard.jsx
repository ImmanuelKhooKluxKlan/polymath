import { useEffect } from 'react';
import { BLACK_KEY_WIDTH_RATIO } from '../engine/grandPianoLayout.js';
import { noteToDisplayName } from '../engine/noteMath.js';
import { TeacherKeyboardPresence, TeacherRowHands } from './TeacherKeyboardOverlay.jsx';

// QWERTY keys cover useful center notes; click/touch covers the full adaptive piano.
const keyboardMap = {
  a: 'C4', w: 'C#4', s: 'D4', e: 'D#4', d: 'E4', f: 'F4', t: 'F#4',
  g: 'G4', y: 'G#4', h: 'A4', u: 'A#4', j: 'B4', k: 'C5', o: 'C#5', l: 'D5', p: 'D#5', ';': 'E5', "'": 'F5',
  z: 'C2', x: 'D2', c: 'E2', v: 'F2', b: 'G2', n: 'A2', m: 'B2', ',': 'C3', '.': 'D3', '/': 'E3',
};

const activePointers = new Map();

function labelFor(note) {
  const found = Object.entries(keyboardMap).find(([, value]) => value === note);
  return found ? found[0].toUpperCase() : '';
}

function pointerHandlers(note, onPress, onRelease) {
  function releasePointer(event) {
    const pointerId = event.pointerId;
    if (activePointers.get(pointerId) !== note) return;

    activePointers.delete(pointerId);
    event.preventDefault();
    try {
      if (event.currentTarget.hasPointerCapture?.(pointerId)) {
        event.currentTarget.releasePointerCapture(pointerId);
      }
    } catch {
      // Capture may already have been released by the browser.
    }

    onRelease(note, {
      pointerId,
      pointerType: event.pointerType,
    });
  }

  return {
    onPointerDown: (event) => {
      event.preventDefault();
      if (activePointers.has(event.pointerId)) return;

      activePointers.set(event.pointerId, note);
      event.currentTarget.setPointerCapture?.(event.pointerId);
      onPress(note, {
        pointerId: event.pointerId,
        pointerType: event.pointerType,
        pressure: event.pressure,
      });
    },
    onPointerUp: releasePointer,
    onPointerCancel: releasePointer,
    onLostPointerCapture: releasePointer,
    onContextMenu: (event) => event.preventDefault(),
    onDragStart: (event) => event.preventDefault(),
  };
}

function strikeVersionFor(strikeVersions, note) {
  return strikeVersions?.get?.(note) || 0;
}

function PianoRow({
  row,
  rowIndex,
  activeNotes,
  strikeVersions,
  onPress,
  onRelease,
  disabled,
  showKeyNotes,
  teacher,
  teacherTargets,
  teacherHandMode,
}) {
  return (
    <div
      className="piano-row-shell"
      style={{ '--white-count': row.whiteCount, '--black-width-ratio': BLACK_KEY_WIDTH_RATIO }}
    >
      <div className="piano-range-label">{row.label} • {row.startNote} to {row.endNote}</div>
      <div className="keyboard-deck">
        <div className="white-layer">
          {row.whiteKeys.map((key) => {
            const version = strikeVersionFor(strikeVersions, key.note);
            return (
              <button
                key={key.note}
                className={`piano-key white ${activeNotes.has(key.note) ? 'active' : ''}`}
                disabled={disabled}
                {...(disabled ? {} : pointerHandlers(key.note, onPress, onRelease))}
              >
                {version > 0 && <span key={`${key.note}-${version}`} className="key-strike-flash" />}
                {showKeyNotes && <span className="key-note">{noteToDisplayName(key.midi, true)}</span>}
                <span className="computer-key">{labelFor(key.note)}</span>
              </button>
            );
          })}
        </div>

        <div className="black-layer" aria-hidden="true">
          {row.blackKeys.map((key) => {
            const version = strikeVersionFor(strikeVersions, key.note);
            return (
              <button
                key={key.note}
                className={`piano-key black ${activeNotes.has(key.note) ? 'active' : ''}`}
                style={{
                  left: `calc(${key.position.leftEdgeWhiteUnits} * (100% / var(--white-count)))`,
                }}
                disabled={disabled}
                {...(disabled ? {} : pointerHandlers(key.note, onPress, onRelease))}
              >
                {version > 0 && <span key={`${key.note}-${version}`} className="key-strike-flash black-flash" />}
                {showKeyNotes && <span className="key-note black-label">{noteToDisplayName(key.midi, true)}</span>}
                <span className="computer-key black-computer-key">{labelFor(key.note)}</span>
              </button>
            );
          })}
        </div>
        {teacher && (
          <TeacherRowHands
            teacher={teacher}
            targets={teacherTargets}
            row={row}
            handMode={teacherHandMode}
            showAtRest={rowIndex === 0 && !teacherTargets?.hasTargets}
          />
        )}
      </div>
    </div>
  );
}

export default function PianoKeyboard({
  layout,
  activeNotes,
  strikeVersions,
  onPress,
  onRelease,
  showKeyNotes = true,
  preparationStatus = 'locked',
  preparationProgress = 0,
  preparationStage = 'Tap once to prepare',
  performanceTier = 'full',
  deviceClass = 'desktop',
  onPrepare,
  teacher = null,
  teacherTargets = null,
  teacherHandMode = 'both',
  teacherIsPlaying = false,
}) {
  useEffect(() => () => {
    activePointers.clear();
  }, []);

  const disabled = preparationStatus !== 'ready';
  const isPreparing = preparationStatus === 'loading' || preparationStatus === 'calibrating';
  const deviceLabel = deviceClass === 'phone'
    ? 'Phone'
    : deviceClass === 'tablet'
      ? 'Tablet'
      : 'Computer';
  const liteWarning = performanceTier === 'lite';
  return (
    <section
      className={`piano-shell ${layout.isTwoStorey ? 'two-storey' : 'single-storey'} ${disabled ? 'is-locked' : 'is-ready'} ${teacher ? 'has-main-teacher' : ''}`}
      aria-label={`Playable piano section, ${layout.rangeLabel}`}
    >
      <div className="piano-mode-label">
        <span>{layout.isTwoStorey ? 'Two-storey A0-C8 grand piano' : 'Polymath Musician A1-C7 row'} • Song range {layout.songRange.minNote}-{layout.songRange.maxNote}</span>
        <small className="performance-tier-badge">{deviceLabel} · {performanceTier}</small>
      </div>
      <div className="piano-glow" />
      {teacher && <TeacherKeyboardPresence teacher={teacher} isPlaying={teacherIsPlaying} />}
      <div className="piano-rows">
        {layout.rows.map((row, rowIndex) => (
          <PianoRow
            key={row.id}
            row={row}
            rowIndex={rowIndex}
            activeNotes={activeNotes}
            strikeVersions={strikeVersions}
            onPress={onPress}
            onRelease={onRelease}
            disabled={disabled}
            showKeyNotes={showKeyNotes}
            teacher={teacher}
            teacherTargets={teacherTargets}
            teacherHandMode={teacherHandMode}
          />
        ))}
      </div>
      {!disabled && liteWarning && (
        <small className="device-performance-notice" role="status">
          Lite mode: reduced effects for smoother playback on this device.
        </small>
      )}
      {disabled && (
        <div className="piano-preparation" aria-live="polite">
          {isPreparing ? (
            <>
              <strong>{preparationStage}</strong>
              <progress max="100" value={preparationProgress} aria-label="Piano preparation progress" />
              <small>{preparationProgress}% · Keep this page open.</small>
            </>
          ) : (
            <>
              <button type="button" className="primary" onClick={onPrepare}>
                {preparationStatus === 'error' ? 'Try keyboard again' : 'Unlock keyboard'}
              </button>
              <small>
                {deviceClass === 'desktop'
                  ? 'Loads this instrument only.'
                  : `Loads a smaller ${deviceLabel.toLowerCase()} piano first. Weaker devices may use Lite mode.`}
              </small>
            </>
          )}
        </div>
      )}
    </section>
  );
}

export { keyboardMap };
