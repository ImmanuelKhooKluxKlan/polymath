import { parseNote, clamp, noteToDisplayName } from '../engine/noteMath.js';
import { PERFORMANCE_TIERS, normalizePerformanceTier } from '../engine/devicePerformance.js';
import { fallingNoteGeometry } from '../engine/fallingNoteGeometry.js';
const MAX_VISUAL_NOTE_SECONDS = 12;

function findFirstRelevantNote(notes, minimumTime) {
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (Number(notes[middle]?.time || 0) < minimumTime) low = middle + 1;
    else high = middle;
  }
  return low;
}

function handClassFor(event) {
  const role = String(event.scoreRole || '').toLowerCase();
  const hand = String(event.hand || '').toLowerCase();
  if (hand === 'left' || role.includes('bass') || role.includes('left')) return 'left-hand-note';
  if (hand === 'right' || role.includes('melody') || role.includes('right')) return 'right-hand-note';
  return 'both-hand-note';
}

function FallingRow({
  row,
  layout,
  visibleNotes,
  currentTime,
  leadTime,
  activeNotes,
  activePlaybackEventIds,
  compact,
}) {
  return (
    <div className="falling-row" style={{ '--white-count': row.whiteCount }}>
      <div className="lane-grid">
        <div className="white-lane-layer">
          {row.whiteKeys.map((key) => <span key={key.note} className="lane white-lane" />)}
        </div>
        <div className="black-lane-layer" aria-hidden="true">
          {row.blackKeys.map((key) => (
            <span
              key={key.note}
              className="lane black-lane"
              style={{
                left: `${key.position.centerPercent}%`,
                width: `${key.position.widthPercent}%`,
              }}
            />
          ))}
        </div>
      </div>
      <div className="timing-line"><span>PRESS LINE</span></div>
      <span className="falling-row-name" aria-hidden="true">{row.label}</span>

      {visibleNotes.map((event) => {
        let parsed;
        try { parsed = parseNote(event.note); } catch { return null; }
        const position = layout.getPosition(parsed.midi);
        if (!position || position.rowId !== row.id) return null;

        const duration = event.visualDuration ?? event.duration ?? 0.2;
        const untilPress = event.time - currentTime;
        const strikeIsActive = activePlaybackEventIds.has(event.id);
        const geometry = fallingNoteGeometry({
          eventTime: event.time,
          currentTime,
          duration,
          leadTime,
          compact,
          strikeIsActive,
        });
        const normalizedNote = parsed.name;
        const isHot = strikeIsActive || Math.abs(untilPress) < 0.12 || activeNotes.has(normalizedNote) || activeNotes.has(event.note);
        const widthScale = event.widthScale || 1;
        const laneOffset = event.laneOffsetPercent || 0;
        const displayName = noteToDisplayName(parsed.midi, true);

        return (
          <div
            key={event.id}
            className={`falling-note ${position.isBlack ? 'black-falling-note' : 'white-falling-note'} ${handClassFor(event)} ${isHot ? 'hot' : ''}`}
            data-event-id={event.id}
            data-note={normalizedNote}
            data-note-time={event.time}
            data-strike-contact={geometry.touching ? 'true' : 'false'}
            style={{
              left: `${clamp(position.centerPercent + laneOffset, 0, 100)}%`,
              top: `${geometry.top}%`,
              height: `${geometry.height}%`,
              width: `${position.widthPercent * (position.isBlack ? 0.62 : compact ? 0.54 : 0.68) * widthScale}%`,
            }}
          >
            <strong>{displayName}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default function FallingNotes({
  song,
  layout,
  currentTime,
  isPlaying,
  leadTime,
  activeNotes,
  activePlaybackEventIds = new Set(),
  performanceTier = 'full',
  compact = layout.isTwoStorey,
}) {
  const notes = song?.notes || [];
  const visibleNotes = [];
  const normalizedTier = normalizePerformanceTier(performanceTier, 'full');
  const maximumNotes = PERFORMANCE_TIERS[normalizedTier].maximumNotes;
  const firstRelevantIndex = findFirstRelevantNote(notes, currentTime - MAX_VISUAL_NOTE_SECONDS);
  const latestVisibleTime = currentTime + leadTime + 0.25;

  for (let index = firstRelevantIndex; index < notes.length; index += 1) {
    const event = notes[index];
    if (event.time > latestVisibleTime) break;
    const duration = event.visualDuration ?? event.duration ?? 0.2;
    const untilPress = event.time - currentTime;
    const visible = untilPress < leadTime + 0.25 && currentTime < event.time + duration + 0.7;
    if (visible) visibleNotes.push(event);
    if (visibleNotes.length >= maximumNotes) break;
  }

  return (
    <section
      className={`falling-stage ${layout.isTwoStorey ? 'two-storey' : 'single-storey'} ${compact ? 'compact-row' : ''}`}
      aria-label="Falling note timeline"
    >
      <div className="stage-stars" />
      <div className="falling-rows">
        {layout.rows.map((row) => (
          <FallingRow
            key={row.id}
            row={row}
            layout={layout}
            visibleNotes={visibleNotes}
            currentTime={currentTime}
            leadTime={leadTime}
            activeNotes={activeNotes}
            activePlaybackEventIds={activePlaybackEventIds}
            compact={compact}
          />
        ))}
      </div>
      {!isPlaying && <div className="paused-badge">Ready</div>}
    </section>
  );
}
