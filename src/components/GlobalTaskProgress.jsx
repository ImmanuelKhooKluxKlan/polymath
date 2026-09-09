import { useEffect, useState } from 'react';
import { GLOBAL_TASK_ACTIVITY_EVENT } from '../services/api.js';

export default function GlobalTaskProgress() {
  const [activeTasks, setActiveTasks] = useState(0);

  useEffect(() => {
    const update = (event) => {
      const delta = Number(event?.detail?.delta) || 0;
      setActiveTasks((current) => Math.max(0, current + delta));
    };
    window.addEventListener(GLOBAL_TASK_ACTIVITY_EVENT, update);
    return () => window.removeEventListener(GLOBAL_TASK_ACTIVITY_EVENT, update);
  }, []);

  if (!activeTasks) return null;
  return (
    <div
      className="global-task-progress"
      role="progressbar"
      aria-label="Saving your changes"
      aria-valuetext="In progress"
    >
      <span />
    </div>
  );
}
