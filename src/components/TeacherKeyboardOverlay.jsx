import { noteToDisplayName } from '../engine/noteMath.js';

function positionedNotes(target, row) {
  return (target?.notes || [])
    .map((note) => ({ ...note, position: row.getPosition(note.midi) }))
    .filter((note) => note.position?.rowId === row.id);
}

function HandAndArm({ side, target, row, teacher, visible, showAtRest }) {
  const notes = positionedNotes(target, row);
  if (!visible || (!notes.length && !showAtRest)) return null;

  const defaultPercent = side === 'left' ? 38 : 62;
  const centerPercent = notes.length
    ? notes.reduce((sum, note) => sum + note.position.centerPercent, 0) / notes.length
    : defaultPercent;
  const centerX = centerPercent * 10;
  const shoulderX = side === 'left' ? 458 : 542;
  const fingertipY = target?.isPressing ? 105 : 94;
  const isActive = notes.length > 0;
  const verticalReach = 176;
  const reach = Math.hypot(centerX - shoulderX, verticalReach);
  const angle = Math.atan2(shoulderX - centerX, verticalReach) * (180 / Math.PI);
  const armWidth = 82;
  const armX = centerX - (armWidth / 2);
  const armY = fingertipY - reach;
  const armMirror = side === 'left' ? `translate(${centerX * 2} 0) scale(-1 1)` : undefined;
  const shoulderMaskId = `teacher-arm-mask-${teacher.id}-${row.id}-${side}`;

  return (
    <g className={`main-teacher-limb main-teacher-limb-${side} ${target?.isPressing ? 'is-pressing' : ''} ${isActive ? 'has-target' : 'at-rest'}`}>
      <defs>
        <linearGradient id={`${shoulderMaskId}-fade`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="black" />
          <stop offset=".13" stopColor="white" />
          <stop offset="1" stopColor="white" />
        </linearGradient>
        <mask id={shoulderMaskId} maskUnits="objectBoundingBox" maskContentUnits="objectBoundingBox">
          <rect x="0" y="0" width="1" height="1" fill={`url(#${shoulderMaskId}-fade)`} />
        </mask>
      </defs>
      <g transform={`rotate(${angle} ${centerX} ${fingertipY})`}>
        <g transform={armMirror}>
          <image
            className="main-teacher-human-arm"
            href={teacher.armImage}
            x={armX}
            y={armY}
            width={armWidth}
            height={reach}
            preserveAspectRatio="none"
            mask={`url(#${shoulderMaskId})`}
          />
        </g>
      </g>
      {notes.map((note) => {
        const endX = note.position.centerPercent * 10;
        const endY = fingertipY;
        return (
          <g key={`${note.id}-${note.finger}`} className="main-teacher-fingertip">
            <circle cx={endX} cy={endY} r="12" />
            <text x={endX} y={endY + 4} textAnchor="middle">{note.finger}</text>
            <title>{`${teacher.name}: ${side} finger ${note.finger} on ${noteToDisplayName(note.midi, true)}`}</title>
          </g>
        );
      })}
    </g>
  );
}

export function TeacherKeyboardPresence({ teacher, isPlaying }) {
  return (
    <div className="main-teacher-presence" aria-label={`${teacher.name} is seated behind the main piano`}>
      <div className="main-teacher-aura" />
      <img src={teacher.image} alt={`${teacher.name}, virtual piano teacher`} draggable="false" />
      <span className={isPlaying ? 'main-teacher-status is-live' : 'main-teacher-status'}>
        <i />{isPlaying ? `${teacher.name} is playing` : `${teacher.name} is ready`}
      </span>
    </div>
  );
}

export function TeacherRowHands({ teacher, targets, row, handMode = 'both', showAtRest = false }) {
  return (
    <svg
      className="main-teacher-hands-overlay"
      viewBox="0 0 1000 150"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <HandAndArm
        side="left"
        target={targets.left}
        row={row}
        teacher={teacher}
        visible={handMode === 'both' || handMode === 'left'}
        showAtRest={showAtRest}
      />
      <HandAndArm
        side="right"
        target={targets.right}
        row={row}
        teacher={teacher}
        visible={handMode === 'both' || handMode === 'right'}
        showAtRest={showAtRest}
      />
    </svg>
  );
}
