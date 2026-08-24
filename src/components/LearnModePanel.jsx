import { formatLearningTime } from '../utils/learningSections.js';

export default function LearnModePanel({
  mode, onModeChange, sections, selectedIndex, onSelectSection, onPracticeSection,
  repeatSection, onRepeatChange, preferredSeconds, onPreferredSecondsChange,
  locked = false, onUpgrade,
}) {
  const selected = sections[selectedIndex] || sections[0];

  return (
    <section className={`learn-mode-panel ${mode === 'learn' ? 'is-learning' : ''}`}>
      <div className="mode-switch" role="group" aria-label="Teaching mode">
        <button type="button" className={mode === 'regular' ? 'active' : ''} onClick={() => onModeChange('regular')}>Chilling</button>
        <button
          type="button"
          className={mode === 'learn' ? 'active' : ''}
          onClick={() => locked ? onUpgrade?.() : onModeChange('learn')}
        >
          {locked ? 'Learn · Musician' : 'Learn'}
        </button>
      </div>

      {mode === 'learn' && !locked && selected && (
        <div className="learn-mode-content">
          <div className="learn-focus">
            <div className="learn-current">
              <span>Part {selectedIndex + 1} of {sections.length}</span>
              <strong>{selected.name}</strong>
              <small>{formatLearningTime(selected.start)}–{formatLearningTime(selected.end)} · {selected.duration.toFixed(1)} seconds</small>
            </div>
            <div className="learn-navigation">
              <button type="button" className="ghost learn-step" aria-label="Previous part" disabled={selectedIndex <= 0} onClick={() => onSelectSection(selectedIndex - 1)}>←</button>
              <button type="button" className="primary learn-play" onClick={() => onPracticeSection(selected)}>Practise this part</button>
              <button type="button" className="ghost learn-step" aria-label="Next part" disabled={selectedIndex >= sections.length - 1} onClick={() => onSelectSection(selectedIndex + 1)}>→</button>
            </div>
            <label className="learn-part-jump">Go to part
              <input
                type="number"
                min="1"
                max={sections.length}
                value={selectedIndex + 1}
                onChange={(event) => {
                  const part = Number(event.target.value);
                  if (Number.isInteger(part) && part >= 1 && part <= sections.length) onSelectSection(part - 1);
                }}
              />
            </label>
            <label className="learn-repeat"><input type="checkbox" checked={repeatSection} onChange={(event) => onRepeatChange(event.target.checked)} /> Loop</label>
          </div>

          <details className="learn-settings">
            <summary>Show all part details</summary>
            <div className="learn-settings-content">
              <label>Approximate part length
                <span><input type="number" min="5" max="60" step="1" value={preferredSeconds} onChange={(event) => onPreferredSecondsChange(Number(event.target.value))} /> seconds</span>
              </label>
              <div className="learning-sections" role="list" aria-label="Song practice sections">
                {sections.map((section, index) => (
                  <button type="button" role="listitem" key={section.id} className={index === selectedIndex ? 'active' : ''} onClick={() => onSelectSection(index)}>
                    <span>Part {index + 1}</span>
                    <strong>{section.name}</strong>
                    <small>{formatLearningTime(section.start)}–{formatLearningTime(section.end)} · {section.duration.toFixed(1)}s</small>
                    {section.exceedsRecommendation && <em>Phrase kept intact</em>}
                  </button>
                ))}
              </div>
            </div>
          </details>
        </div>
      )}
    </section>
  );
}
