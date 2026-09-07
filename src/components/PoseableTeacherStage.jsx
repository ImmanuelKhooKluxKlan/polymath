import { useEffect, useRef, useState } from 'react';
import {
  clampTeacherDepth,
  clampTeacherOffset,
  teacherPoseById,
  TEACHER_POSES,
} from '../engine/teacherAvatarControls.js';
import ArticulatedTeacherCanvas from './ArticulatedTeacherCanvas.jsx';

const INITIAL_POSITION = Object.freeze({ x: 0, y: 0 });

function TeacherJoystick({ onMove, onReset }) {
  const padRef = useRef(null);
  const pointerRef = useRef(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  function updateFromPointer(event) {
    const pad = padRef.current;
    if (!pad) return;
    const bounds = pad.getBoundingClientRect();
    const centerX = bounds.left + bounds.width / 2;
    const centerY = bounds.top + bounds.height / 2;
    const radius = Math.max(1, bounds.width * 0.31);
    let x = event.clientX - centerX;
    let y = event.clientY - centerY;
    const length = Math.hypot(x, y);
    if (length > radius) {
      x = (x / length) * radius;
      y = (y / length) * radius;
    }
    setKnob({ x, y });
    const previous = pointerRef.current?.last || { x: 0, y: 0 };
    pointerRef.current.last = { x, y };
    onMove({
      horizontal: (x - previous.x) / radius,
      depth: -(y - previous.y) / radius,
    });
  }

  function begin(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerRef.current = { pointerId: event.pointerId, last: { x: 0, y: 0 } };
    updateFromPointer(event);
  }

  function move(event) {
    if (pointerRef.current?.pointerId !== event.pointerId) return;
    updateFromPointer(event);
  }

  function end(event) {
    if (pointerRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    pointerRef.current = null;
    setKnob({ x: 0, y: 0 });
  }

  return (
    <div className="teacher-joystick-control">
      <span>Move / depth</span>
      <div
        ref={padRef}
        className="teacher-joystick"
        onPointerDown={begin}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
        aria-label="Drag joystick left or right to move. Drag forward or back to change depth."
      >
        <button type="button" className="teacher-joystick-front" aria-label="Move teacher forward" onClick={() => onMove({ depth: 0.08 })}>Front</button>
        <button type="button" className="teacher-joystick-left" aria-label="Move teacher left" onClick={() => onMove({ horizontal: -0.12 })}>Left</button>
        <button type="button" className="teacher-joystick-centre" aria-label="Reset teacher position and depth" onClick={onReset}>Reset</button>
        <button type="button" className="teacher-joystick-right" aria-label="Move teacher right" onClick={() => onMove({ horizontal: 0.12 })}>Right</button>
        <button type="button" className="teacher-joystick-back" aria-label="Move teacher back" onClick={() => onMove({ depth: -0.08 })}>Back</button>
        <i className="teacher-joystick-knob" aria-hidden="true" style={{ '--joystick-x': `${knob.x}px`, '--joystick-y': `${knob.y}px` }} />
      </div>
    </div>
  );
}

export default function PoseableTeacherStage({ teacher, targetSummary, performanceTier = 'balanced' }) {
  const stageRef = useRef(null);
  const [poseId, setPoseId] = useState('ready');
  const [motionKey, setMotionKey] = useState(0);
  const [position, setPosition] = useState(INITIAL_POSITION);
  const [depth, setDepth] = useState(1);
  const pose = teacherPoseById(poseId);

  useEffect(() => {
    setPoseId('ready');
    setPosition(INITIAL_POSITION);
    setDepth(1);
  }, [teacher.id]);

  function selectPose(nextPoseId) {
    setPoseId(nextPoseId);
    setMotionKey((current) => current + 1);
  }

  function moveWithJoystick({ horizontal = 0, depth: depthChange = 0 }) {
    const stageWidth = stageRef.current?.getBoundingClientRect().width || 600;
    setPosition((current) => clampTeacherOffset({
      ...current,
      x: current.x + horizontal * stageWidth * 0.34,
    }, {
      maximumX: stageWidth * 0.36,
      maximumY: stageRef.current?.getBoundingClientRect().height * 0.27 || 150,
    }));
    setDepth((current) => clampTeacherDepth(current + depthChange));
  }

  function resetTeacher() {
    setPoseId('ready');
    setMotionKey((current) => current + 1);
    setPosition(INITIAL_POSITION);
    setDepth(1);
  }

  return (
    <article className="poseable-teacher-card">
      <div
        ref={stageRef}
        className="poseable-teacher-stage"
        aria-label={`${teacher.name} interactive teacher stage`}
      >
        <div className="poseable-teacher-stage-glow" />
        <div className="poseable-teacher-instruction">Touch a limb and drag</div>
        <ArticulatedTeacherCanvas
          teacher={teacher}
          poseId={pose.id}
          motionKey={motionKey}
          position={position}
          depth={depth}
          performanceTier={performanceTier}
        />
        <div className="poseable-teacher-caption">
          <strong>{teacher.name}</strong>
          <span>{teacher.description}</span>
          <small>{targetSummary}</small>
        </div>
      </div>

      <div className="teacher-motion-controls" aria-label={`Move and pose ${teacher.name}`}>
        <div className="teacher-pose-buttons" role="group" aria-label="Teacher pose">
          {TEACHER_POSES.map((candidate) => (
            <button
              type="button"
              key={candidate.id}
              className={candidate.id === pose.id ? 'is-selected' : ''}
              aria-pressed={candidate.id === pose.id}
              onClick={() => selectPose(candidate.id)}
            >
              {candidate.label}
            </button>
          ))}
        </div>
        <TeacherJoystick onMove={moveWithJoystick} onReset={resetTeacher} />
      </div>
    </article>
  );
}
