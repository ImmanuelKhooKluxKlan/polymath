import { useEffect, useRef, useState } from 'react';
import {
  clampTeacherOffset,
  normalizeTeacherArmAngle,
  teacherPoseById,
  TEACHER_POSES,
} from '../engine/teacherAvatarControls.js';

const INITIAL_ARMS = Object.freeze({ left: -8, right: 8 });

export default function PoseableTeacherStage({ teacher, targetSummary }) {
  const stageRef = useRef(null);
  const dragRef = useRef(null);
  const [poseId, setPoseId] = useState('ready');
  const [motionKey, setMotionKey] = useState(0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [arms, setArms] = useState(INITIAL_ARMS);
  const [customArms, setCustomArms] = useState(false);
  const pose = teacherPoseById(poseId);

  useEffect(() => {
    setPoseId('ready');
    setOffset({ x: 0, y: 0 });
    setArms(INITIAL_ARMS);
    setCustomArms(false);
  }, [teacher.id]);

  function selectPose(nextPoseId) {
    setPoseId(nextPoseId);
    setMotionKey((current) => current + 1);
    if (nextPoseId === 'stretch') {
      setArms({ left: -92, right: 92 });
      setCustomArms(true);
    }
  }

  function beginDrag(event) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      offset,
    };
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    const stage = stageRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !stage) return;
    const bounds = stage.getBoundingClientRect();
    setOffset(clampTeacherOffset({
      x: drag.offset.x + event.clientX - drag.clientX,
      y: drag.offset.y + event.clientY - drag.clientY,
    }, {
      maximumX: Math.max(0, bounds.width * 0.34),
      maximumY: Math.max(0, bounds.height * 0.18),
    }));
  }

  function endDrag(event) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  function updateArm(side, value) {
    setCustomArms(true);
    setArms((current) => ({ ...current, [side]: normalizeTeacherArmAngle(value) }));
  }

  function resetTeacher() {
    setPoseId('ready');
    setMotionKey((current) => current + 1);
    setOffset({ x: 0, y: 0 });
    setArms(INITIAL_ARMS);
    setCustomArms(false);
  }

  return (
    <article className="poseable-teacher-card">
      <div
        ref={stageRef}
        className="poseable-teacher-stage"
        aria-label={`${teacher.name} interactive teacher stage`}
      >
        <div className="poseable-teacher-stage-glow" />
        <div className="poseable-teacher-instruction">Drag {teacher.name} to move</div>
        <div
          key={`${teacher.id}-${pose.id}-${motionKey}`}
          className={`poseable-teacher-rig is-pose-${pose.id} ${customArms ? 'has-custom-arms' : ''}`}
          style={{
            '--teacher-x': `${offset.x}px`,
            '--teacher-y': `${offset.y}px`,
            '--teacher-rotation': `${pose.rotation}deg`,
            '--teacher-scale': pose.scale,
          }}
          onPointerDown={beginDrag}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          <img
            className="poseable-teacher-body"
            src={teacher.image}
            alt={`${teacher.name}, interactive virtual piano teacher`}
            draggable="false"
          />
          <img
            className="poseable-teacher-arm poseable-teacher-arm-left"
            src={teacher.armImage}
            alt=""
            draggable="false"
            style={{ '--arm-rotation': `${arms.left}deg` }}
          />
          <img
            className="poseable-teacher-arm poseable-teacher-arm-right"
            src={teacher.armImage}
            alt=""
            draggable="false"
            style={{ '--arm-rotation': `${arms.right}deg` }}
          />
        </div>
        <div className="poseable-teacher-caption">
          <strong>{teacher.name}</strong>
          <span>{teacher.description}</span>
          <small>{targetSummary}</small>
        </div>
      </div>

      <div className="teacher-motion-controls" aria-label={`Move ${teacher.name}`}>
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
        <div className="teacher-arm-sliders">
          <label>
            <span>Left arm</span>
            <input
              type="range"
              min="-110"
              max="110"
              value={arms.left}
              onChange={(event) => updateArm('left', event.target.value)}
            />
          </label>
          <label>
            <span>Right arm</span>
            <input
              type="range"
              min="-110"
              max="110"
              value={arms.right}
              onChange={(event) => updateArm('right', event.target.value)}
            />
          </label>
          <button type="button" className="teacher-reset-pose" onClick={resetTeacher}>Reset</button>
        </div>
      </div>
    </article>
  );
}
