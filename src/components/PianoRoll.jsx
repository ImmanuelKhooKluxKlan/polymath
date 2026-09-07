import FallingNotes from './FallingNotes.jsx';
import PianoKeyboard from './PianoKeyboard.jsx';

function layoutForRow(layout, row) {
  return {
    ...layout,
    parentMode: layout.mode,
    isTwoStorey: false,
    rows: [row],
    rangeLabel: `${row.startNote}-${row.endNote}`,
    getPosition(noteOrMidi) {
      return row.getPosition(noteOrMidi);
    },
  };
}

export default function PianoRoll({
  song,
  layout,
  currentTime,
  isPlaying,
  leadTime,
  activeNotes,
  activePlaybackEventIds,
  performanceTier,
  ...keyboardProps
}) {
  const compact = layout.isTwoStorey;

  return (
    <div className={`piano-roll ${compact ? 'two-storey' : 'single-storey'}`}>
      {layout.rows.map((row, index) => {
        const rowLayout = layoutForRow(layout, row);
        return (
          <section className="piano-roll-pair" data-piano-row={row.id} key={row.id}>
            <FallingNotes
              song={song}
              layout={rowLayout}
              currentTime={currentTime}
              isPlaying={isPlaying}
              leadTime={leadTime}
              activeNotes={activeNotes}
              activePlaybackEventIds={activePlaybackEventIds}
              performanceTier={performanceTier}
              compact={compact}
            />
            <div className="piano-scroll-wrap">
              <PianoKeyboard
                {...keyboardProps}
                layout={rowLayout}
                activeNotes={activeNotes}
                performanceTier={performanceTier}
                compact={compact}
                showModeLabel={index === 0}
                showPreparation={index === 0}
              />
            </div>
          </section>
        );
      })}
    </div>
  );
}
