export default function MusicChoiceDisclosure({
  id,
  title,
  summary = '',
  expanded,
  onToggle,
  children,
}) {
  return (
    <div className={`music-choice-disclosure ${expanded ? 'expanded' : ''}`}>
      <button
        type="button"
        className="music-choice-trigger"
        aria-expanded={expanded}
        aria-controls={id}
        onClick={onToggle}
      >
        <span className="music-choice-copy">
          <strong>{title}</strong>
          {summary && <small>{summary}</small>}
        </span>
        <span className="music-choice-chevron" aria-hidden="true">&#8964;</span>
      </button>

      {expanded && (
        <div id={id} className="music-choice-content">
          {children}
        </div>
      )}
    </div>
  );
}
