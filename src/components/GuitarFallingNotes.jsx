const STRING_NAMES = ['Low E', 'A', 'D', 'G', 'B', 'High E'];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function playableNotesForEvent(event) {
  if (Array.isArray(event.frets)) {
    return event.frets
      .map((fret, stringIndex) => ({ stringIndex, fret: Number(fret) }))
      .filter((item) => Number.isFinite(item.fret) && item.fret >= 0);
  }

  if (Number.isInteger(event.stringIndex) && Number.isFinite(Number(event.fret))) {
    return [{ stringIndex: event.stringIndex, fret: Number(event.fret) }];
  }

  return [];
}

export default function GuitarFallingNotes({
  lesson,
  currentTime,
  isPlaying,
  leadTime,
  activeEventId,
}) {
  const visible = [];

  for (const event of lesson.events || []) {
    const duration = Number(event.duration || 0.5);
    const until = event.time - currentTime;
    if (until < leadTime + 0.3 && currentTime < event.time + duration + 0.6) {
      visible.push(event);
    }
    if (visible.length > 220) break;
  }

  return (
    <section className="guitar-falling-stage" aria-label="Falling guitar tablature">
      <div className="stage-stars" />
      <div className="guitar-lane-labels">
        {STRING_NAMES.map((name) => <span key={name}>{name}</span>)}
      </div>
      <div className="guitar-lanes">
        {STRING_NAMES.map((name) => <span key={name} className="guitar-lane" />)}
      </div>
      <div className="guitar-press-line"><span>PLAY LINE</span></div>
      <div className="guitar-target-bridge" aria-hidden="true">
        {STRING_NAMES.map((name) => <span key={name}>{name.replace('Low ', '').replace('High ', '')}</span>)}
      </div>

      {visible.flatMap((event) => {
        const duration = Number(event.duration || 0.5);
        const until = event.time - currentTime;
        const pressLinePercent = 89;
        const travelPercent = 106;
        const bottom = clamp(pressLinePercent - (until / leadTime) * travelPercent, -20, 126);
        const height = clamp((duration / leadTime) * travelPercent, 4.5, 82);
        const top = clamp(bottom - height, -90, 126);

        return playableNotesForEvent(event).map(({ stringIndex, fret }) => (
          <div
            key={`${event.id}-${stringIndex}`}
            className={`guitar-falling-note string-${stringIndex} ${activeEventId === event.id ? 'hot' : ''}`}
            style={{
              left: `${(stringIndex / 6) * 100 + 1.2}%`,
              width: `${100 / 6 - 2.4}%`,
              top: `${top}%`,
              height: `${height}%`,
            }}
          >
            <strong>{fret}</strong>
            {event.chord && <small>{event.chord}</small>}
          </div>
        ));
      })}

      {!isPlaying && <div className="paused-badge">Ready</div>}
    </section>
  );
}
