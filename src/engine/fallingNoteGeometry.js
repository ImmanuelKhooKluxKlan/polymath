import { clamp } from './noteMath.js';

export const FALLING_NOTE_STRIKE_PERCENT = 100;
export const FALLING_NOTE_TRAVEL_PERCENT = 108;
export const FALLING_NOTE_STRIKE_SYNC_SECONDS = 0.1;

/**
 * Keep the falling bar and the illuminated key on one visual clock.
 *
 * The audio scheduler can illuminate a key between two animation frames.
 * During that brief interval currentTime is slightly behind the scheduled
 * note. Snap only that scheduled event to the contact boundary so an early
 * manual press cannot move an unrelated falling bar.
 */
export function synchronizedUntilPress(
  untilPress,
  strikeIsActive = false,
  maximumEarlySeconds = FALLING_NOTE_STRIKE_SYNC_SECONDS,
) {
  const remaining = Number(untilPress) || 0;
  if (
    strikeIsActive
    && remaining > 0
    && remaining <= Math.max(0, Number(maximumEarlySeconds) || 0)
  ) {
    return 0;
  }
  return remaining;
}

export function fallingNoteGeometry({
  eventTime,
  currentTime,
  duration,
  leadTime,
  compact = false,
  strikeIsActive = false,
}) {
  const safeLeadTime = Math.max(0.05, Number(leadTime) || 0.05);
  const untilPress = Number(eventTime) - Number(currentTime);
  const visualUntilPress = synchronizedUntilPress(untilPress, strikeIsActive);
  const bottom = clamp(
    FALLING_NOTE_STRIKE_PERCENT
      - (visualUntilPress / safeLeadTime) * FALLING_NOTE_TRAVEL_PERCENT,
    -22,
    132,
  );
  const heightScale = compact ? 0.58 : 1;
  const height = clamp(
    (Math.max(0.035, Number(duration) || 0.2) / safeLeadTime)
      * FALLING_NOTE_TRAVEL_PERCENT
      * heightScale,
    2.4,
    compact ? 54 : 112,
  );
  const top = clamp(bottom - height, -118, 132);

  return {
    top,
    height,
    bottom,
    untilPress,
    visualUntilPress,
    touching: Math.abs(bottom - FALLING_NOTE_STRIKE_PERCENT) < 0.000001,
  };
}
