function normalizedProgress(value) {
  if (value === null || value === undefined || value === '') return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, numeric));
}

export default function TaskProgress({
  label = 'Working on it…',
  progress = null,
  detail = '',
  onCancel,
  cancelLabel = 'Stop',
  compact = false,
  className = '',
  ariaLabel = 'Task progress',
}) {
  const safeProgress = normalizedProgress(progress);
  const determinate = safeProgress !== null;
  const roundedProgress = determinate ? Math.round(safeProgress) : null;

  return (
    <div
      className={`task-progress ${compact ? 'is-compact' : ''} ${className}`.trim()}
      role="status"
      aria-live="polite"
    >
      <div className="task-progress-heading">
        <strong>{label}</strong>
        {determinate && <span>{roundedProgress}%</span>}
        {onCancel && <button type="button" onClick={onCancel}>{cancelLabel}</button>}
      </div>
      <div
        className={`task-progress-track ${determinate ? '' : 'is-indeterminate'}`.trim()}
        role="progressbar"
        aria-label={ariaLabel}
        aria-valuemin={determinate ? 0 : undefined}
        aria-valuemax={determinate ? 100 : undefined}
        aria-valuenow={determinate ? roundedProgress : undefined}
        aria-valuetext={determinate ? `${roundedProgress}% complete` : 'In progress'}
      >
        <span style={determinate ? { width: `${safeProgress}%` } : undefined} />
      </div>
      {detail && <small>{detail}</small>}
    </div>
  );
}
