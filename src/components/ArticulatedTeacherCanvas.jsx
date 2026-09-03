import { useEffect, useRef, useState } from 'react';

export default function ArticulatedTeacherCanvas({ teacher, poseId, motionKey, position, depth, performanceTier = 'balanced' }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const runtimeRef = useRef(null);
  const latestControlsRef = useRef({ poseId, motionKey, position, depth });
  const [renderState, setRenderState] = useState({ mode: 'loading', message: 'Preparing 3D teacher' });
  const [activeJoint, setActiveJoint] = useState('');

  useEffect(() => {
    latestControlsRef.current = { poseId, motionKey, position, depth };
    runtimeRef.current?.setPose(poseId, motionKey);
  }, [poseId, motionKey]);

  useEffect(() => {
    latestControlsRef.current = { poseId, motionKey, position, depth };
    runtimeRef.current?.setTransform({ x: position.x, depth });
  }, [position, depth, poseId, motionKey]);

  useEffect(() => {
    let cancelled = false;
    let runtime = null;
    setRenderState({ mode: 'loading', message: 'Preparing 3D teacher' });
    setActiveJoint('');
    import('../engine/teacherThreeRuntime.js')
      .then(({ createTeacherThreeRuntime }) => {
        if (cancelled) return null;
        return createTeacherThreeRuntime({
          canvas: canvasRef.current,
          container: containerRef.current,
          teacher,
          performanceTier,
          onStatus: (status) => {
            if (!cancelled) setRenderState(status);
          },
          onJointChange: (joint) => {
            if (!cancelled) setActiveJoint(joint);
          },
        });
      })
      .then((createdRuntime) => {
        if (!createdRuntime) return;
        if (cancelled) {
          createdRuntime.dispose();
          return;
        }
        runtime = createdRuntime;
        runtimeRef.current = createdRuntime;
        const controls = latestControlsRef.current;
        createdRuntime.setTransform({ x: controls.position.x, depth: controls.depth });
        createdRuntime.setPose(controls.poseId, controls.motionKey);
      })
      .catch((error) => {
        console.error('Articulated teacher renderer failed:', error);
        if (!cancelled) setRenderState({ mode: 'fallback', message: 'Portrait fallback' });
      });

    return () => {
      cancelled = true;
      runtime?.dispose();
      if (runtimeRef.current === runtime) runtimeRef.current = null;
    };
  }, [teacher, performanceTier]);

  return (
    <div ref={containerRef} className={`articulated-teacher-view is-${renderState.mode}`}>
      <canvas
        ref={canvasRef}
        className="articulated-teacher-canvas"
        role="img"
        aria-label={`${teacher.name} articulated 3D teacher. Touch a limb and drag to move its joint.`}
      />
      {renderState.mode === 'fallback' && (
        <img className="articulated-teacher-fallback" src={teacher.image} alt={`${teacher.name}, virtual piano teacher`} draggable="false" />
      )}
      <span className="articulated-teacher-status" role="status" aria-live="polite">{activeJoint ? `Moving ${activeJoint}` : renderState.message}</span>
    </div>
  );
}
