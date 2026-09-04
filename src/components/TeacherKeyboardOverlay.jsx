import { useEffect, useRef } from 'react';
import { teacherRowHandPlacement } from '../engine/teacherHands.js';

function boundedTravelDuration(previousCenter, nextCenter) {
  if (!Number.isFinite(previousCenter)) return 165;
  const distance = Math.abs(nextCenter - previousCenter);
  return Math.round(Math.max(135, Math.min(300, 135 + distance * 5.5)));
}

const PRESSED_POSE_FOR_BASE_IMAGE = Object.freeze({
  '/teachers/pianist-hands-overhead-v1.webp': '/teachers/pianist-hands-pressed-v2.webp',
  '/teachers/pianist-hands-overhead-male-v1.webp': '/teachers/pianist-hands-pressed-male-v2.webp',
  '/teachers/pianist-hands-overhead-dark-v1.webp': '/teachers/pianist-hands-pressed-dark-v2.webp',
});

function PhotographicHand({ side, target, row, teacher, visible, showAtRest }) {
  const placement = visible ? teacherRowHandPlacement(target, row, side, showAtRest) : null;
  const previousCenter = useRef(placement?.centerPercent);
  const travelDuration = placement
    ? boundedTravelDuration(previousCenter.current, placement.centerPercent)
    : 165;

  useEffect(() => {
    if (placement) previousCenter.current = placement.centerPercent;
  }, [placement?.centerPercent]);

  if (!placement) return null;

  const image = teacher?.handCameraImage || '/teachers/pianist-hands-overhead-v1.webp';
  const pressedImage = teacher?.pressedHandCameraImage
    || PRESSED_POSE_FOR_BASE_IMAGE[image]
    || '';
  const sourceViewBox = side === 'left' ? '0 0 768 900' : '768 0 768 900';
  const state = target?.isPressing ? 'is-pressing' : target?.isUpcoming ? 'is-upcoming' : 'at-rest';

  return (
    <svg
      className={`teacher-main-hand-photo teacher-main-hand-${side} ${state}`}
      style={{
        '--teacher-hand-center': `${placement.centerPercent}%`,
        '--teacher-hand-width': `${placement.widthPercent}%`,
        '--teacher-hand-narrow': placement.horizontalScale,
        '--teacher-finger-depth': `${placement.fingerDepthPercent}%`,
        '--teacher-hand-tilt': `${placement.wristTiltDegrees}deg`,
        '--teacher-hand-flex': placement.verticalFlex,
        '--teacher-hand-lift': `${placement.approachLiftPixels}px`,
        '--teacher-hand-press': `${placement.pressDepthPixels}px`,
        '--teacher-hand-travel-ms': `${travelDuration}ms`,
      }}
      viewBox={sourceViewBox}
      preserveAspectRatio="none"
      focusable="false"
      aria-hidden="true"
    >
      <image
        className="teacher-hand-pose teacher-hand-pose-relaxed"
        href={image}
        x="0"
        y="0"
        width="1536"
        height="1024"
        transform="rotate(180 768 512)"
        preserveAspectRatio="none"
      />
      {pressedImage && pressedImage !== image && (
        <image
          className="teacher-hand-pose teacher-hand-pose-pressed"
          href={pressedImage}
          x="0"
          y="0"
          width="1536"
          height="1024"
          transform="rotate(180 768 512)"
          preserveAspectRatio="none"
        />
      )}
    </svg>
  );
}

export function TeacherRowHands({
  teacher,
  targets,
  row,
  handMode = 'both',
  showAtRest = false,
}) {
  if (!teacher || !targets || !row) return null;
  return (
    <div className="teacher-row-hands" aria-hidden="true">
      <PhotographicHand
        side="left"
        target={targets.left}
        row={row}
        teacher={teacher}
        visible={handMode === 'both' || handMode === 'left'}
        showAtRest={showAtRest}
      />
      <PhotographicHand
        side="right"
        target={targets.right}
        row={row}
        teacher={teacher}
        visible={handMode === 'both' || handMode === 'right'}
        showAtRest={showAtRest}
      />
    </div>
  );
}
