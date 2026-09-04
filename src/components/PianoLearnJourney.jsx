import { useEffect, useMemo, useRef, useState } from 'react';
import {
  PIANO_LEARNING_LEVELS,
  LEARNING_SESSION_GOALS,
  recommendedLearningLevel,
} from '../engine/learningCoach.js';
import { formatLearningTime } from '../utils/learningSections.js';

const JOURNEY_STEPS = [
  ['music', 'Music'],
  ['level', 'Level'],
  ['session', 'Session'],
  ['play', 'Play'],
  ['review', 'Review'],
];

function songArtist(song) {
  const artist = String(song?.artist || song?.composer || '').trim();
  return artist && !['Unknown composer', 'CSV/JSON import', 'MIDI import'].includes(artist)
    ? artist
    : 'Your piano arrangement';
}

function rangeLabel(range) {
  if (!range) return 'Choose a section';
  return `${formatLearningTime(range.start)}–${formatLearningTime(range.end)}`;
}

function Metric({ metric }) {
  return (
    <div className={`learn-review-metric ${metric.available ? '' : 'is-unavailable'}`}>
      <span>{metric.label}</span>
      <strong>{metric.available ? `${metric.score}%` : 'Not measured'}</strong>
      <small>{metric.detail}</small>
    </div>
  );
}

export default function PianoLearnJourney({
  mode,
  locked = false,
  onUpgrade,
  onModeChange,
  song,
  songKey,
  levelId,
  onLevelChange,
  sessionId,
  onSessionChange,
  sections,
  selectedIndex,
  onSelectSection,
  activeRange,
  repeatSection,
  onRepeatChange,
  handMode,
  onHandModeChange,
  onChooseMusic,
  onPrepare,
  preparationStatus,
  preparationProgress,
  preparationStage,
  midi,
  onConnectMidi,
  onListen,
  onStartAttempt,
  attemptStatus,
  report,
  progress,
  onOpenTeacher,
  onFindTeacher,
  onOpenBand,
  onFocusPlayer,
}) {
  const [step, setStep] = useState(0);
  const panelRef = useRef(null);
  const previousSongRef = useRef(song?.libraryId || song?.title);
  const currentLevel = PIANO_LEARNING_LEVELS.find((level) => level.id === levelId) || PIANO_LEARNING_LEVELS[1];
  const currentSession = LEARNING_SESSION_GOALS.find((goal) => goal.id === sessionId) || LEARNING_SESSION_GOALS[1];
  const selectedSection = sections[selectedIndex] || sections[0];
  const songProgress = progress?.songs?.[songKey] || null;
  const recommendedLevelId = recommendedLearningLevel(songProgress);
  const stepTitle = [
    'Start with music you care about',
    'Make it achievable today',
    'Choose the size of today’s win',
    'Listen once—or play it yourself',
    report ? report.headline : 'Your review will appear here',
  ][step];
  const noteCount = useMemo(() => Number(song?.notes?.length || 0).toLocaleString(), [song?.notes?.length]);

  useEffect(() => {
    const songIdentity = song?.libraryId || song?.title;
    if (previousSongRef.current && songIdentity !== previousSongRef.current && mode === 'learn') {
      setStep(1);
      window.setTimeout(() => panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }
    previousSongRef.current = songIdentity;
  }, [song?.libraryId, song?.title, mode]);

  useEffect(() => {
    if (!report?.createdAt) return;
    setStep(4);
    window.setTimeout(() => panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
  }, [report?.createdAt]);

  function chooseStep(index) {
    if (index === 4 && !report) return;
    setStep(index);
  }

  function switchMode(nextMode) {
    if (nextMode === 'learn' && locked) {
      onUpgrade?.();
      return;
    }
    onModeChange(nextMode);
    if (nextMode === 'learn') setStep(0);
  }

  return (
    <section ref={panelRef} className={`piano-learn-journey ${mode === 'learn' ? 'is-active' : ''}`} aria-labelledby="learn-journey-title">
      <div className="mode-switch learn-journey-mode" role="group" aria-label="Piano mode">
        <button type="button" className={mode === 'regular' ? 'active' : ''} onClick={() => switchMode('regular')}>Chilling</button>
        <button type="button" className={mode === 'learn' ? 'active' : ''} onClick={() => switchMode('learn')}>
          {locked ? 'Learn · Musician' : 'Learn'}
        </button>
      </div>

      {mode === 'learn' && !locked && (
        <div className="learn-journey-body">
          <header className="learn-journey-header">
            <div>
              <p className="eyebrow">Learn journey</p>
              <h2 id="learn-journey-title">{stepTitle}</h2>
            </div>
            <span className="learn-journey-position">{step + 1} / {JOURNEY_STEPS.length}</span>
          </header>

          <nav className="learn-journey-steps" aria-label="Learning journey progress">
            {JOURNEY_STEPS.map(([id, label], index) => (
              <button
                type="button"
                key={id}
                className={`${index === step ? 'is-current' : ''} ${index < step ? 'is-complete' : ''}`}
                aria-current={index === step ? 'step' : undefined}
                aria-label={`Step ${index + 1}: ${label}`}
                disabled={index === 4 && !report}
                onClick={() => chooseStep(index)}
              >
                <i>{index < step ? '✓' : index + 1}</i>
                <span>{label}</span>
              </button>
            ))}
          </nav>

          <div className="learn-journey-stage" key={JOURNEY_STEPS[step][0]}>
            {step === 0 && (
              <div className="learn-music-step">
                <div className="learn-song-focus">
                  <span className="learn-song-art" aria-hidden="true"><i /><b>♪</b></span>
                  <div>
                    <span>Selected song</span>
                    <strong>{song?.title || 'Choose a song'}</strong>
                    <small>{songArtist(song)} · {noteCount} source notes</small>
                  </div>
                </div>
                <div className="learn-primary-actions">
                  <button type="button" className="primary" onClick={() => setStep(1)}>Learn this song</button>
                  <button type="button" className="ghost" onClick={() => onChooseMusic?.('available')}>Choose another</button>
                </div>
                <details className="learn-disclosure">
                  <summary>Use my own music</summary>
                  <div><p>Upload a ready-to-play sheet, MIDI or supported audio source. Your current song stays selected until a replacement is ready.</p><button type="button" onClick={() => onChooseMusic?.('upload')}>Open music upload</button></div>
                </details>
              </div>
            )}

            {step === 1 && (
              <div className="learn-level-step">
                <div className="learn-choice-grid learn-level-grid" role="radiogroup" aria-label="Arrangement difficulty">
                  {PIANO_LEARNING_LEVELS.map((level) => (
                    <button
                      type="button"
                      role="radio"
                      aria-checked={level.id === levelId}
                      key={level.id}
                      className={level.id === levelId ? 'is-selected' : ''}
                      onClick={() => onLevelChange(level.id)}
                    >
                      <span>{level.shortLabel}{level.id === recommendedLevelId ? ' · Recommended' : ''}</span>
                      <strong>{level.label}</strong>
                      <small>{level.summary}</small>
                    </button>
                  ))}
                </div>
                <details className="learn-disclosure">
                  <summary>What changes in {currentLevel.shortLabel}?</summary>
                  <div><p>{currentLevel.detail}</p><p>Starting tempo: {Math.round(currentLevel.speed * 100)}%. You can change it below the keyboard at any time.</p></div>
                </details>
                <div className="learn-primary-actions"><button type="button" className="primary" onClick={() => setStep(2)}>Continue</button></div>
              </div>
            )}

            {step === 2 && (
              <div className="learn-session-step">
                <div className="learn-choice-grid learn-session-grid" role="radiogroup" aria-label="Practice session goal">
                  {LEARNING_SESSION_GOALS.map((goal) => (
                    <button
                      type="button"
                      role="radio"
                      aria-checked={goal.id === sessionId}
                      key={goal.id}
                      className={goal.id === sessionId ? 'is-selected' : ''}
                      onClick={() => onSessionChange(goal.id)}
                    >
                      <strong>{goal.label}</strong>
                      <small>{goal.summary}</small>
                    </button>
                  ))}
                </div>
                <div className="learn-session-focus">
                  <span>{currentSession.id === 'full' ? 'Today’s run' : `Part ${selectedIndex + 1} of ${sections.length}`}</span>
                  <strong>{currentSession.id === 'full' ? 'Full song' : selectedSection?.name}</strong>
                  <small>{rangeLabel(activeRange)}</small>
                </div>
                {currentSession.id !== 'full' && (
                  <details className="learn-disclosure">
                    <summary>Choose a different part</summary>
                    <div className="learn-part-list" role="list">
                      {sections.map((section, index) => (
                        <button type="button" role="listitem" key={section.id} className={index === selectedIndex ? 'is-selected' : ''} onClick={() => onSelectSection(index)}>
                          <span>Part {index + 1}</span><strong>{section.name}</strong><small>{rangeLabel(section)}</small>
                        </button>
                      ))}
                    </div>
                  </details>
                )}
                <div className="learn-primary-actions"><button type="button" className="primary" onClick={() => { setStep(3); window.setTimeout(() => onFocusPlayer?.(), 80); }}>Go to piano</button></div>
              </div>
            )}

            {step === 3 && (
              <div className="learn-play-step">
                <div className={`learn-readiness ${preparationStatus === 'ready' ? 'is-ready' : ''}`}>
                  <i aria-hidden="true" />
                  <div>
                    <span>{preparationStatus === 'ready' ? 'Instrument ready' : 'Prepare this device'}</span>
                    <strong>{preparationStatus === 'ready' ? `${currentLevel.label} · ${rangeLabel(activeRange)}` : preparationStage}</strong>
                    {['calibrating', 'loading'].includes(preparationStatus) && <progress max="100" value={preparationProgress} aria-label="Keyboard preparation progress" />}
                  </div>
                  {preparationStatus !== 'ready' && !['calibrating', 'loading'].includes(preparationStatus) && <button type="button" className="primary" onClick={onPrepare}>Prepare piano</button>}
                </div>

                <div className="learn-input-source">
                  <div><span>Playing input</span><strong>{midi?.status === 'connected' ? midi.name : 'Screen or computer keys'}</strong></div>
                  {midi?.supported && midi.status !== 'connected' && <button type="button" className="ghost" onClick={onConnectMidi} disabled={midi.status === 'connecting'}>{midi.status === 'connecting' ? 'Connecting…' : 'Connect MIDI'}</button>}
                  {midi?.status === 'connected' && <span className="learn-connected-badge">Velocity ready</span>}
                </div>
                {midi?.error && <p className="learn-inline-error" role="alert">{midi.error}</p>}

                {['preparing', 'running', 'paused'].includes(attemptStatus) ? (
                  <div className="learn-attempt-live" role="status"><i /><span><strong>{attemptStatus === 'preparing' ? 'Getting your attempt ready' : attemptStatus === 'paused' ? 'Attempt paused' : 'Listening to your attempt'}</strong><small>{attemptStatus === 'preparing' ? 'The timer starts only when the piano is ready.' : attemptStatus === 'paused' ? 'Press play below the keyboard when you are ready.' : 'Play when each falling note reaches the keys.'}</small></span></div>
                ) : (
                  <div className="learn-play-actions">
                    <button type="button" className="ghost" disabled={preparationStatus !== 'ready'} onClick={() => onListen(activeRange)}>Hear example</button>
                    <button type="button" className="primary" disabled={preparationStatus !== 'ready'} onClick={() => onStartAttempt(activeRange)}>Start my attempt</button>
                  </div>
                )}

                <details className="learn-disclosure">
                  <summary>Practice options</summary>
                  <div className="learn-practice-options">
                    <fieldset>
                      <legend>Hands</legend>
                      <div role="group" aria-label="Hands to practise">
                        {[['left', 'Left'], ['right', 'Right'], ['both', 'Both']].map(([value, label]) => <button type="button" key={value} className={handMode === value ? 'is-selected' : ''} onClick={() => onHandModeChange(value)}>{label}</button>)}
                      </div>
                    </fieldset>
                    <label><input type="checkbox" checked={repeatSection} onChange={(event) => onRepeatChange(event.target.checked)} /> Loop example playback</label>
                    <small>Touch and dynamics are measured only when a velocity-sensitive MIDI keyboard is connected. Polymath will not invent a score it cannot measure.</small>
                  </div>
                </details>

                <button type="button" className="learn-teacher-link" onClick={onOpenTeacher}>Need a demonstration? Open virtual teacher</button>
              </div>
            )}

            {step === 4 && (
              <div className="learn-review-step">
                {!report ? (
                  <div className="learn-empty-review"><span aria-hidden="true">♪</span><strong>Play one attempt first</strong><small>Your note, rhythm and hold feedback will appear here.</small><button type="button" className="primary" onClick={() => setStep(3)}>Go to play</button></div>
                ) : (
                  <>
                    <div className="learn-review-hero">
                      <div className="learn-score-ring" style={{ '--score': report.score }}><strong>{report.score}</strong><span>out of 100</span></div>
                      <div><span>Coach focus · {report.focus}</span><h3>{report.headline}</h3><p>{report.nextAction}</p></div>
                    </div>
                    <div className="learn-review-summary">
                      <span><strong>{report.matchedCount}/{report.expectedCount}</strong> notes found</span>
                      <span><strong>{report.strongest}</strong> strongest</span>
                      <span><strong>{songProgress?.bestScore || report.score}</strong> personal best</span>
                    </div>
                    <div className="learn-primary-actions">
                      <button type="button" className="primary" onClick={() => { setStep(3); onStartAttempt(activeRange); }}>Try this part again</button>
                      {selectedIndex < sections.length - 1 && sessionId !== 'full' && <button type="button" className="ghost" onClick={() => { onSelectSection(selectedIndex + 1); setStep(3); }}>Next part</button>}
                    </div>
                    <details className="learn-disclosure learn-full-review">
                      <summary>See full feedback</summary>
                      <div className="learn-review-metrics">
                        {Object.values(report.metrics).map((item) => <Metric key={item.label} metric={item} />)}
                      </div>
                    </details>
                    <details className="learn-disclosure">
                      <summary>Get help from people</summary>
                      <div className="learn-human-help"><p>Share progress with a teacher or practise alongside other musicians when you want human support.</p><button type="button" onClick={onFindTeacher}>Find a teacher</button><button type="button" onClick={onOpenBand}>Open Band</button></div>
                    </details>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
