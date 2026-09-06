import { useMemo, useState } from 'react';
import { buildLearningWin, shareLearningWin } from '../engine/learningWin.js';
import { trackProductEvent } from '../services/productAnalytics.js';

export default function LearningWinShare({ report, song, songKey, level, momentum }) {
  const [status, setStatus] = useState('idle');
  const win = useMemo(() => buildLearningWin({
    report,
    song,
    songKey,
    level,
    momentum,
    baseUrl: window.location.href,
  }), [level, momentum, report, song, songKey]);

  async function share() {
    if (status === 'sharing') return;
    setStatus('sharing');
    const result = await shareLearningWin(win);
    setStatus(result);
    if (result === 'shared' || result === 'copied') {
      trackProductEvent('learning_win_shared', {
        outcome: result,
        score: win.score,
        level: level?.id || level?.stage || '',
      });
    }
  }

  const statusText = status === 'copied'
    ? 'Challenge link copied.'
    : status === 'shared'
      ? 'Challenge shared.'
      : status === 'cancelled'
        ? 'Sharing cancelled.'
        : '';

  return (
    <section className="learn-win-share" aria-label="Share this piano result">
      <div>
        <span>Share the proof</span>
        <strong>Challenge a friend to beat {win.score}.</strong>
        <small>Your score is shared—not your recording.</small>
      </div>
      <button type="button" className="ghost" disabled={status === 'sharing'} onClick={share}>
        {status === 'sharing' ? 'Preparing card…' : status === 'copied' ? 'Copied' : 'Share my win'}
      </button>
      <span className="learn-share-status" role="status" aria-live="polite">{statusText}</span>
    </section>
  );
}
