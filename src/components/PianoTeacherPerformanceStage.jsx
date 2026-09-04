import { useEffect, useMemo, useState } from 'react';
import {
  TEACHER_KEYBOARD_GEOMETRY,
  separateTeacherHandCentres,
  teacherHandCenter,
  teacherKeyX,
} from '../engine/teacherPerformanceRig.js';

function safeId(value) {
  return String(value || 'teacher').replace(/[^a-z0-9_-]/gi, '-');
}

function boundedPercent(value, fallback, minimum = 0, maximum = 100) {
  const number = Number(value);
  return Math.max(minimum, Math.min(maximum, Number.isFinite(number) ? number : fallback));
}

function TeacherSpeechMouth({ teacher, speechCue }) {
  const image = teacher?.stageImage || teacher?.image;
  if (!image) return null;
  const supplied = teacher?.mouthAnchor || {};
  const width = boundedPercent(supplied.width, 4, 1.5, 12);
  const height = boundedPercent(supplied.height, 3, 1.2, 9);
  const x = boundedPercent(supplied.x, 50, width / 2, 100 - width / 2);
  const y = boundedPercent(supplied.y, 22, height / 2, 100 - height / 2);
  const left = x - width / 2;
  const top = y - height / 2;
  return (
    <span
      className="teacher-speech-mouth-window"
      data-speaking={speechCue?.active ? 'true' : 'false'}
      data-viseme={speechCue?.viseme || 'rest'}
      style={{
        left: `${left}%`,
        top: `${top}%`,
        width: `${width}%`,
        height: `${height}%`,
        '--mouth-interior-opacity': Number(speechCue?.intensity || 0) * 0.58,
      }}
      aria-hidden="true"
    >
      <img
        className="teacher-speech-mouth-source"
        src={image}
        alt=""
        draggable="false"
        style={{
          width: `${10000 / width}%`,
          height: `${10000 / height}%`,
          left: `${(-left * 100) / width}%`,
          top: `${(-top * 100) / height}%`,
        }}
      />
      <i className="teacher-speech-mouth-interior" />
    </span>
  );
}

function useSmoothedCentres(targets) {
  const rawLeft = teacherHandCenter(targets?.left, 'left');
  const rawRight = teacherHandCenter(targets?.right, 'right');
  const bothHandsActive = Boolean(targets?.left?.notes?.length && targets?.right?.notes?.length);
  const separated = bothHandsActive
    ? separateTeacherHandCentres(rawLeft, rawRight)
    : { left: rawLeft, right: rawRight };
  const desiredLeft = separated.left;
  const desiredRight = separated.right;
  const [centres, setCentres] = useState({ left: desiredLeft, right: desiredRight });

  useEffect(() => {
    let frame = 0;
    let latest = { ...centres };
    const move = () => {
      latest = {
        left: latest.left + (desiredLeft - latest.left) * 0.18,
        right: latest.right + (desiredRight - latest.right) * 0.18,
      };
      setCentres(latest);
      if (Math.abs(latest.left - desiredLeft) > 0.45 || Math.abs(latest.right - desiredRight) > 0.45) {
        frame = window.requestAnimationFrame(move);
      }
    };
    frame = window.requestAnimationFrame(move);
    return () => window.cancelAnimationFrame(frame);
  }, [desiredLeft, desiredRight]);

  return centres;
}

function PerformanceKeyboard({ targets }) {
  const activeMidis = new Set([
    ...(targets?.left?.notes || []),
    ...(targets?.right?.notes || []),
  ].map((note) => Number(note.midi)));
  const whiteKeys = TEACHER_KEYBOARD_GEOMETRY.filter((key) => !key.black);
  const blackKeys = TEACHER_KEYBOARD_GEOMETRY.filter((key) => key.black);
  return (
    <g className="teacher-performance-keyboard">
      <rect className="performance-piano-case" x="36" y="478" width="1128" height="172" rx="24" />
      <rect className="performance-piano-fallboard" x="52" y="483" width="1096" height="31" rx="7" />
      {whiteKeys.map((key) => (
        <rect key={key.midi} className={`performance-white-key ${activeMidis.has(key.midi) ? 'has-target' : ''}`} x={key.x} y={key.y} width={key.width - 1} height={key.height} rx="1.7" />
      ))}
      {blackKeys.map((key) => (
        <rect key={key.midi} className={`performance-black-key ${activeMidis.has(key.midi) ? 'has-target' : ''}`} x={key.x} y={key.y} width={key.width} height={key.height} rx="3" />
      ))}
      <rect className="performance-piano-front" x="42" y="614" width="1116" height="36" rx="10" />
    </g>
  );
}

function HandCamera({ teacher, targets, centres }) {
  const id = safeId(teacher?.id);
  const image = teacher?.handCameraImage || '/teachers/pianist-hands-overhead-v1.webp';
  const leftShift = Math.max(-270, Math.min(250, centres.left - 437));
  const rightShift = Math.max(-250, Math.min(270, centres.right - 782));
  const noteMarkers = useMemo(() => [
    ...(targets?.left?.notes || []).map((note) => ({ ...note, side: 'left' })),
    ...(targets?.right?.notes || []).map((note) => ({ ...note, side: 'right' })),
  ], [targets]);

  return (
    <div className="teacher-hand-camera">
      <div className="teacher-hand-camera-label"><span>Hands · song-synced</span><small>Exact MIDI timing on the lesson keys</small></div>
      <svg viewBox="0 180 1200 500" role="img" aria-label={`${teacher.name}'s overhead hands moving across the lesson keyboard`}>
        <defs>
          <clipPath id={`${id}-left-hand`} clipPathUnits="objectBoundingBox"><rect x="0" y="0" width=".5" height="1" /></clipPath>
          <clipPath id={`${id}-right-hand`} clipPathUnits="objectBoundingBox"><rect x=".5" y="0" width=".5" height="1" /></clipPath>
          <filter id={`${id}-hand-shadow`} x="-30%" y="-30%" width="160%" height="180%">
            <feDropShadow dx="0" dy="10" stdDeviation="8" floodColor="#02030f" floodOpacity=".58" />
          </filter>
        </defs>
        <PerformanceKeyboard targets={targets} />
        <g className="teacher-photo-hands" filter={`url(#${id}-hand-shadow)`}>
          <image
            className={targets?.left?.isPressing ? 'teacher-photo-hand is-pressing' : 'teacher-photo-hand'}
            href={image}
            x="220"
            y="90"
            width="760"
            height="507"
            clipPath={`url(#${id}-left-hand)`}
            transform={`translate(${leftShift} ${targets?.left?.isPressing ? 7 : 0})`}
            preserveAspectRatio="none"
          />
          <image
            className={targets?.right?.isPressing ? 'teacher-photo-hand is-pressing' : 'teacher-photo-hand'}
            href={image}
            x="220"
            y="90"
            width="760"
            height="507"
            clipPath={`url(#${id}-right-hand)`}
            transform={`translate(${rightShift} ${targets?.right?.isPressing ? 7 : 0})`}
            preserveAspectRatio="none"
          />
        </g>
        <g className="teacher-finger-guides">
          {noteMarkers.map((note) => (
            <g key={`${note.side}-${note.midi}-${note.finger}`} transform={`translate(${teacherKeyX(note.midi)} 642)`}>
              <rect x="-23" y="-14" width="46" height="24" rx="12" />
              <text textAnchor="middle" y="3">{note.note}</text>
              <title>{`${note.side} finger ${note.finger} on ${note.note}`}</title>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

export default function PianoTeacherPerformanceStage({
  teacher,
  targets,
  isPlaying = false,
  performanceTier = 'balanced',
  speechCue = null,
}) {
  const centres = useSmoothedCentres(targets);
  const summary = [...(targets?.left?.notes || []), ...(targets?.right?.notes || [])]
    .map((note) => note.note)
    .join(', ');
  const fullStagePhoto = Boolean(teacher?.stageImage);
  const isSpeaking = Boolean(speechCue?.active);

  return (
    <article
      className={`teacher-performance-stage virtual-teacher-live-stage performance-${performanceTier} ${isSpeaking ? 'is-speaking' : ''} ${isPlaying ? 'is-demonstrating' : ''}`}
      data-teacher-id={teacher.id}
      aria-label={`${teacher.name} seated at the piano with synchronized voice and hands`}
    >
      <div className="teacher-performance-scene">
        <img
          className={`teacher-performance-person ${fullStagePhoto ? 'is-stage-photo' : 'is-cutout'}`}
          src={teacher?.stageImage || teacher?.image}
          alt={`${teacher.name}, your selected virtual piano teacher seated at the piano`}
          draggable="false"
        />
        <TeacherSpeechMouth teacher={teacher} speechCue={speechCue} />
        {!fullStagePhoto && <div className="teacher-performance-cutout-piano" aria-hidden="true" />}
        <div className={isPlaying || isSpeaking ? 'teacher-performance-live is-playing' : 'teacher-performance-live'}>
          <i />
          <span>
            <strong>{isPlaying ? `${teacher.name} is demonstrating` : isSpeaking ? `${teacher.name} is speaking` : `${teacher.name} is ready`}</strong>
            <small>{summary || (isSpeaking ? 'Mouth following live speech boundaries' : 'Relaxed shoulders · level forearms · neutral wrists')}</small>
          </span>
        </div>
        <div className="teacher-performance-sync" aria-label="Teacher animation synchronization status">
          <span className={isSpeaking ? 'is-active' : ''}>Voice-synced lips</span>
          <span className={isPlaying ? 'is-active' : ''}>MIDI-synced hands</span>
        </div>
      </div>
      <HandCamera teacher={teacher} targets={targets} centres={centres} />
    </article>
  );
}
