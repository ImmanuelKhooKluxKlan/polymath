import { useEffect, useState } from 'react';
import TaskProgress from './TaskProgress.jsx';
import { routeLoadingDetail, routeLoadingProgress } from '../utils/routeLoading.js';

export default function RouteLoadingProgress({
  label = 'Opening this section...',
  initialElapsedMs = 0,
}) {
  const [elapsedMs, setElapsedMs] = useState(Math.max(0, Number(initialElapsedMs) || 0));

  useEffect(() => {
    const startedAt = Date.now() - Math.max(0, Number(initialElapsedMs) || 0);
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt);
    }, 750);
    return () => window.clearInterval(timer);
  }, [initialElapsedMs]);

  return (
    <TaskProgress
      compact
      className="route-loading-progress"
      label={label}
      progress={routeLoadingProgress(elapsedMs)}
      detail={routeLoadingDetail(elapsedMs)}
      ariaLabel="Section loading progress"
    />
  );
}
