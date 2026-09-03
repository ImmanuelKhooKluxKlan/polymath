import { useEffect, useRef, useState } from 'react';
import {
  clampTeacherOffset,
  normalizeTeacherArmAngle,
  teacherBodyPartById,
  teacherPoseById,
  TEACHER_BODY_PARTS,
  TEACHER_POSES,
} from '../engine/teacherAvatarControls.js';

const INITIAL_ARMS = Object.freeze({ left: -8, right: 8 });

function initialPartOffsets() {
  return {
    head: { x: 0, y: 0 },
    torso: { x: 0, y: 0 },
    lower: { x: 0, y: 0 },
  };
}

function sideForPart(partId) {
  if (partId === 'leftArm') return 'left';
  if (partId === 'rightArm') return 'right';
  return null;
}

export default function PoseableTeacherStage({ teacher, targetSummary }) {
  const stageRef = useRef(null);
  const dragRef = useRef(null);
  const [poseId, setPoseId] = useState('ready');
  const [motionKey, setMotionKey] = useState(0);
  const [selectedPart, setSelectedPart] = useState('torso');
  const [partOffsets, setPartOffsets] = useState(initialPartOffsets);
  const [arms, setArms] = useState(INITIAL_ARMS);
  const [customArms, setCustomArms] = useState(false);
  const pose = teacherPoseById(poseId);
  const selectedPartLabel = teacherBodyPartById(selectedPart).label;

  useEffect(() => {
    setPoseId('ready');
    setSelectedPart('torso');
    setPartOffsets(initialPartOffsets());
    setArms(INITIAL_ARMS);
    setCustomArms(false);
  }, [teacher.id]);

  function selectPart(partId) {
    setSelectedPart(partId);
    if (sideForPart(partId)) setCustomArms(true);
  }

  function selectPose(nextPoseId) {
    setPoseId(nextPoseId);
    setMotionKey((current) => current + 1);
    if (nextPoseId === 'stretch') {
      setArms({ left: -92, right: 92 });
      setCustomArms(true);
    }
  }

  function beginPartDrag(partId, event) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    selectPart(partId);
    const side = sideForPart(partId);
    dragRef.current = {
      pointerId: event.pointerId,
      partId,
      clientX: event.clientX,
      clientY: event.clientY,
      offset: side ? null : partOffsets[partId],
      armAngle: side ? arms[side] : null,
    };
  }

  function movePartDrag(event) {
    const drag = dragRef.current;
    const stage = stageRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !stage) return;
    const deltaX = event.clientX - drag.clientX;
    const deltaY = event.clientY - drag.clientY;
    const side = sideForPart(drag.partId);

    if (side) {
      setArms((current) => ({
        ...current,
        [side]: normalizeTeacherArmAngle(drag.armAngle + deltaX * 0.72 - deltaY * 0.18),
      }));
      return;
    }

    const bounds = stage.getBoundingClientRect();
    const nextOffset = clampTeacherOffset({
      x: drag.offset.x + deltaX,
      y: drag.offset.y + deltaY,
    }, {
      maximumX: Math.max(0, bounds.width * 0.22),
      maximumY: Math.max(0, bounds.height * 0.16),
    });
    setPartOffsets((current) => ({ ...current, [drag.partId]: nextOffset }));
  }

  function endPartDrag(event) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  function partPointerHandlers(partId) {
    return {
      onPointerDown: (event) => beginPartDrag(partId, event),
      onPointerMove: movePartDrag,
      onPointerUp: endPartDrag,
      onPointerCancel: endPartDrag,
    };
  }

  function bodyPartStyle(partId) {
    const offset = partOffsets[partId];
    return {
      '--part-x': `${offset.x}px`,
      '--part-y': `${offset.y}px`,
    };
  }

  function resetTeacher() {
    setPoseId('ready');
    setMotionKey((current) => current + 1);
    setSelectedPart('torso');
    setPartOffsets(initialPartOffsets());
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
        <div className="poseable-teacher-instruction">
          Selected: {selectedPartLabel} · drag the highlighted part
        </div>
        <div
          key={`${teacher.id}-${pose.id}-${motionKey}`}
          className={`poseable-teacher-rig is-pose-${pose.id} ${customArms ? 'has-custom-arms' : ''}`}
          style={{
            '--teacher-rotation': `${pose.rotation}deg`,
            '--teacher-scale': pose.scale,
          }}
          role="img"
          aria-label={`${teacher.name}, interactive virtual piano teacher`}
        >
          <img
            className={`poseable-teacher-body-part poseable-teacher-lower ${selectedPart === 'lower' ? 'is-selected' : ''}`}
            src={teacher.image}
            alt=""
            draggable="false"
            style={bodyPartStyle('lower')}
            {...partPointerHandlers('lower')}
          />
          <img
            className={`poseable-teacher-body-part poseable-teacher-torso ${selectedPart === 'torso' ? 'is-selected' : ''}`}
            src={teacher.image}
            alt=""
            draggable="false"
            style={bodyPartStyle('torso')}
            {...partPointerHandlers('torso')}
          />
          <img
            className={`poseable-teacher-body-part poseable-teacher-head ${selectedPart === 'head' ? 'is-selected' : ''}`}
            src={teacher.image}
            alt=""
            draggable="false"
            style={bodyPartStyle('head')}
            {...partPointerHandlers('head')}
          />
          <img
            className={`poseable-teacher-arm poseable-teacher-arm-left ${selectedPart === 'leftArm' ? 'is-selected' : ''}`}
            src={teacher.armImage}
            alt=""
            draggable="false"
            style={{ '--arm-rotation': `${arms.left}deg` }}
            {...partPointerHandlers('leftArm')}
          />
          <img
            className={`poseable-teacher-arm poseable-teacher-arm-right ${selectedPart === 'rightArm' ? 'is-selected' : ''}`}
            src={teacher.armImage}
            alt=""
            draggable="false"
            style={{ '--arm-rotation': `${arms.right}deg` }}
            {...partPointerHandlers('rightArm')}
          />
          {!sideForPart(selectedPart) && (
            <span
              className={`poseable-teacher-part-highlight is-${selectedPart}`}
              style={bodyPartStyle(selectedPart)}
              aria-hidden="true"
            />
          )}
        </div>
        <div className="poseable-teacher-caption">
          <strong>{teacher.name}</strong>
          <span>{teacher.description}</span>
          <small>{targetSummary}</small>
        </div>
      </div>

      <div className="teacher-motion-controls" aria-label={`Pose ${teacher.name}`}>
        <div className="teacher-part-buttons" role="group" aria-label="Select a body part to move">
          {TEACHER_BODY_PARTS.map((part) => (
            <button
              type="button"
              key={part.id}
              className={part.id === selectedPart ? 'is-selected' : ''}
              aria-pressed={part.id === selectedPart}
              onClick={() => selectPart(part.id)}
            >
              {part.label}
            </button>
          ))}
        </div>
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
        <button type="button" className="teacher-reset-pose" onClick={resetTeacher}>Reset all parts</button>
      </div>
    </article>
  );
}
