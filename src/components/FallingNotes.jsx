import { parseNote, clamp, noteToDisplayName } from '../engine/noteMath.js';
import { PERFORMANCE_TIERS, normalizePerformanceTier } from '../engine/devicePerformance.js';
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

function FallingRow({ row, layout, visibleNotes, currentTime, leadTime, activeNotes, compact }) {
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

      {visibleNotes.map((event) => {
        let parsed;
        try { parsed = parseNote(event.note); } catch { return null; }
        const position = layout.getPosition(parsed.midi);
        if (!position || position.rowId !== row.id) return null;

        const duration = event.visualDuration ?? event.duration ?? 0.2;
        const untilPress = event.time - currentTime;
        const pressLinePercent = 98;
        const travelPercent = 108;
        const bottomY = clamp(pressLinePercent - (untilPress / leadTime) * travelPercent, -22, 132);
        const heightScale = compact ? 0.58 : 1;
        const height = clamp((duration / leadTime) * travelPercent * heightScale, 2.4, compact ? 54 : 112);
        const topY = clamp(bottomY - height, -118, 132);
        const normalizedNote = parsed.name;
        const isHot = Math.abs(untilPress) < 0.12 || activeNotes.has(normalizedNote) || activeNotes.has(event.note);
        const widthScale = event.widthScale || 1;
        const laneOffset = event.laneOffsetPercent || 0;
        const displayName = noteToDisplayName(parsed.midi, true);

        return (
          <div
            key={event.id}
            className={`falling-note ${position.isBlack ? 'black-falling-note' : 'white-falling-note'} ${handClassFor(event)} ${isHot ? 'hot' : ''}`}
            style={{
              left: `${clamp(position.centerPercent + laneOffset, 0, 100)}%`,
              top: `${topY}%`,
              height: `${height}%`,
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
  performanceTier = 'full',
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
    <section className={`falling-stage ${layout.isTwoStorey ? 'two-storey' : 'single-storey'}`} aria-label="Falling note timeline">
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
            compact={layout.isTwoStorey}
          />
        ))}
      </div>
      {!isPlaying && <div className="paused-badge">Ready</div>}
    </section>
  );
}
