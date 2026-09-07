import { useMemo } from 'react';
import { buildLearningMomentum } from '../engine/learningMomentum.js';
import { apiAssetUrl } from '../services/api.js';
import LearningWinShare from './LearningWinShare.jsx';

function resultHeadline(score) {
  if (score >= 90) return 'Stage-ready';
  if (score >= 75) return 'Strong performance';
  if (score >= 55) return 'You found the song';
  return 'First attempt complete';
}

export default function ArtistChallengeExperience({
  campaign,
  campaignStatus,
  campaignError,
  song,
  activeRange,
  preparationStatus,
  preparationProgress,
  preparationStage,
  onPrepare,
  onListen,
  onStartAttempt,
  attemptStatus,
  report,
  progress,
  challengeScore,
  onUpgrade,
  onExplore,
}) {
  const momentum = useMemo(() => buildLearningMomentum(progress), [progress]);
  const busy = ['preparing', 'running', 'paused'].includes(attemptStatus);
  const adminPreview = campaign?.adminPreview === true;

  if (campaignStatus === 'loading') {
    return (
      <section className="artist-challenge is-loading" aria-live="polite">
        <div className="campaign-loading-mark" aria-hidden="true"><i /><i /><i /></div>
        <strong>Preparing the artist challenge…</strong>
        <span>Loading the verified playable notes.</span>
      </section>
    );
  }

  if (campaignStatus === 'error' || !campaign) {
    return (
      <section className="artist-challenge is-error" role="alert">
        <p className="eyebrow">Challenge unavailable</p>
        <h1>This link is not ready to play.</h1>
        <p>{campaignError || 'The campaign may be paused, scheduled, or no longer live.'}</p>
        <button type="button" className="primary" onClick={onExplore}>Explore Polymath</button>
      </section>
    );
  }

  return (
    <section className="artist-challenge" aria-label={`${campaign.title} artist challenge`}>
      <header className="artist-challenge-hero">
        <div className="artist-challenge-cover">
          {campaign.coverUrl
            ? <img src={apiAssetUrl(campaign.coverUrl)} alt={`${campaign.title} cover`} />
            : <span aria-hidden="true">♪</span>}
        </div>
        <div className="artist-challenge-copy">
          <div className="artist-challenge-proof">
            {adminPreview && <span>Private preview</span>}
            <span>{campaign.verification?.humanVerified ? 'Human verified' : 'Needs pianist QA'}</span>
            <span>{campaign.verification?.qaScore || 0}/100 QA</span>
            <span>{campaign.preview?.durationSeconds || 20}s</span>
          </div>
          <p className="eyebrow">{adminPreview ? 'Administrator quality check' : 'Official artist challenge'}</p>
          <h1>{campaign.hook}</h1>
          <p className="artist-challenge-song"><strong>{campaign.title}</strong> by {campaign.artist}</p>
          {campaign.description && <p>{campaign.description}</p>}
          {campaign.artistUrl && (
            <a href={campaign.artistUrl} target="_blank" rel="noreferrer">Visit {campaign.artist}</a>
          )}
        </div>
      </header>

      {challengeScore !== null && challengeScore !== undefined && (
        <div className="artist-challenge-target">
          <span>Friend challenge</span>
          <strong>Beat {challengeScore}/100</strong>
        </div>
      )}

      {!report ? (
        <div className="artist-challenge-action">
          <div className={`learn-readiness ${preparationStatus === 'ready' ? 'is-ready' : ''}`}>
            <i aria-hidden="true" />
            <div>
              <span>{preparationStatus === 'ready' ? 'Piano ready' : 'One tap to prepare'}</span>
              <strong>{preparationStatus === 'ready' ? 'Play when the falling notes touch the keys' : preparationStage}</strong>
              {['calibrating', 'loading'].includes(preparationStatus) && (
                <progress max="100" value={preparationProgress} aria-label="Piano preparation progress" />
              )}
            </div>
            {preparationStatus !== 'ready' && !['calibrating', 'loading'].includes(preparationStatus) && (
              <button type="button" className="primary" onClick={onPrepare}>Prepare piano</button>
            )}
          </div>

          {busy ? (
            <div className="learn-attempt-live" role="status">
              <i /><span><strong>{attemptStatus === 'preparing' ? 'Getting ready' : attemptStatus === 'paused' ? 'Attempt paused' : 'Listening to you'}</strong><small>Keep playing until the challenge ends.</small></span>
            </div>
          ) : (
            <div className="artist-challenge-buttons">
              <button
                type="button"
                className="primary"
                disabled={preparationStatus !== 'ready' || !activeRange}
                onClick={() => onStartAttempt(activeRange)}
              >
                Play the challenge
              </button>
              <button
                type="button"
                className="ghost"
                disabled={preparationStatus !== 'ready' || !activeRange}
                onClick={() => onListen(activeRange)}
              >
                Hear it first
              </button>
            </div>
          )}
          <small>{adminPreview
            ? 'Private draft preview. Attempts are not added to campaign analytics.'
            : 'No sign-in. No card. Your recording stays on this device.'}</small>
        </div>
      ) : (
        <div className="artist-challenge-result">
          <div className="learn-score-ring" style={{ '--score': report.score }}>
            <strong>{report.score}</strong><span>out of 100</span>
          </div>
          <div className="artist-challenge-result-copy">
            <p className="eyebrow">{resultHeadline(report.score)}</p>
            <h2>{report.headline}</h2>
            <p>{report.matchedCount}/{report.expectedCount} notes matched. {report.nextAction}</p>
          </div>
          {!adminPreview && (
            <LearningWinShare
              report={report}
              song={song}
              songKey={song?.libraryId}
              level={{ id: 'artist-challenge', label: 'Artist challenge' }}
              momentum={momentum}
              campaign={campaign}
            />
          )}
          <div className="artist-challenge-buttons">
            {adminPreview
              ? <button type="button" className="primary" onClick={onExplore}>Return to campaign admin</button>
              : <button type="button" className="primary" onClick={onUpgrade}>Keep learning with Musician</button>}
            <button type="button" className="ghost" onClick={() => onStartAttempt(activeRange)}>Try again</button>
          </div>
        </div>
      )}

      <button type="button" className="artist-challenge-exit" onClick={onExplore}>
        {adminPreview ? 'Return to campaign admin' : 'Explore Polymath'}
      </button>
    </section>
  );
}
