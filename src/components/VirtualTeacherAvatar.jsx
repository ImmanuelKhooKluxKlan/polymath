import { useCallback, useEffect, useRef, useState } from 'react';
import TeacherSensesPanel from './TeacherSensesPanel.jsx';
import TeacherConversationPanel from './TeacherConversationPanel.jsx';
import TeacherProjectionControls from './TeacherProjectionControls.jsx';
import TeacherArControls from './TeacherArControls.jsx';
import { evaluatePerformanceEvent, selectObservedNotes } from '../engine/teacherCoachingEngine.js';

const PREVIEW_DURATION_MS = 4200;
const TEACHER_SESSION_KEY = 'polymath_virtual_teacher_unlocked';
const TEACHER_VISIBLE_SESSION_KEY = 'polymath_virtual_teacher_visible';
const TEACHER_STYLE_SESSION_KEY = 'polymath_virtual_teacher_style';

export default function VirtualTeacherAvatar({
  speaking = false,
  isPlaying = false,
  deviceClass = 'desktop',
  performanceTier = 'balanced',
  song,
  currentTime = 0,
}) {
  const [teacherUnlocked, setTeacherUnlocked] = useState(
    () => window.sessionStorage.getItem(TEACHER_SESSION_KEY) === '1'
  );
  const [unlockChoicesOpen, setUnlockChoicesOpen] = useState(false);
  const [showTeacherPhysically, setShowTeacherPhysically] = useState(
    () => window.sessionStorage.getItem(TEACHER_VISIBLE_SESSION_KEY) !== '0'
  );
  const [teacherStyle, setTeacherStyle] = useState(
    () => window.sessionStorage.getItem(TEACHER_STYLE_SESSION_KEY) === 'bikini' ? 'bikini' : 'regular'
  );
  const [previewSpeaking, setPreviewSpeaking] = useState(false);
  const [teacherSpeaking, setTeacherSpeaking] = useState(false);
  const [coachingObservations, setCoachingObservations] = useState([]);
  const sensesRef = useRef(null);
  const activeObservedRef = useRef(new Map());
  const feedbackCooldownRef = useRef(new Map());
  const isSpeaking = speaking || previewSpeaking || teacherSpeaking;

  useEffect(() => {
    if (!isPlaying) return;
    activeObservedRef.current.clear();
  }, [isPlaying]);

  const handleObservation = useCallback((observation) => {
    if (isPlaying) return;
    const capturedAt = Number(observation?.capturedAt) || performance.now();
    const observed = selectObservedNotes(observation).filter((entry) => entry.confidence >= 0.3);
    const currentMidis = new Set(observed.map((entry) => entry.midi));
    for (const entry of observed) {
      const active = activeObservedRef.current.get(entry.midi);
      if (active) {
        active.lastSeenAt = capturedAt;
        active.confidence = Math.max(active.confidence, entry.confidence);
        active.misses = 0;
      } else {
        activeObservedRef.current.set(entry.midi, {
          ...entry,
          startCapturedAt: capturedAt,
          lastSeenAt: capturedAt,
          startedSongTime: Number(observation?.songTime) || currentTime,
          misses: 0,
        });
      }
    }

    for (const [midi, active] of activeObservedRef.current) {
      if (currentMidis.has(midi)) continue;
      active.misses += 1;
      if (active.misses < 2) continue;
      activeObservedRef.current.delete(midi);
      const duration = Math.max(0.04, (active.lastSeenAt - active.startCapturedAt) / 1000);
      const result = evaluatePerformanceEvent(song, { ...active, midi, duration });
      if (!result) continue;
      const cooldownKey = `${result.type}:${result.note}`;
      const lastFeedback = feedbackCooldownRef.current.get(cooldownKey) || 0;
      if (result.type !== 'correct' && capturedAt - lastFeedback < 2200) continue;
      feedbackCooldownRef.current.set(cooldownKey, capturedAt);
      setCoachingObservations((current) => [...current.slice(-39), result]);
    }
  }, [currentTime, isPlaying, song]);

  useEffect(() => {
    if (!previewSpeaking) return undefined;

    const timer = window.setTimeout(() => setPreviewSpeaking(false), PREVIEW_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [previewSpeaking]);

  function unlockTeacher(showPhysically) {
    window.sessionStorage.setItem(TEACHER_SESSION_KEY, '1');
    window.sessionStorage.setItem(TEACHER_VISIBLE_SESSION_KEY, showPhysically ? '1' : '0');
    setShowTeacherPhysically(showPhysically);
    setUnlockChoicesOpen(false);
    setTeacherUnlocked(true);
  }

  function togglePhysicalTeacher() {
    const next = !showTeacherPhysically;
    window.sessionStorage.setItem(TEACHER_VISIBLE_SESSION_KEY, next ? '1' : '0');
    setShowTeacherPhysically(next);
  }

  function chooseTeacherStyle(style) {
    window.sessionStorage.setItem(TEACHER_STYLE_SESSION_KEY, style);
    setTeacherStyle(style);
  }

  function hideLocalTeacherForProjection() {
    window.sessionStorage.setItem(TEACHER_VISIBLE_SESSION_KEY, '0');
    setShowTeacherPhysically(false);
  }

  function hideTeacher() {
    window.sessionStorage.removeItem(TEACHER_SESSION_KEY);
    setPreviewSpeaking(false);
    setTeacherSpeaking(false);
    window.speechSynthesis?.cancel?.();
    setTeacherUnlocked(false);
  }

  if (!teacherUnlocked) {
    return (
      <section className="virtual-teacher-unlock" aria-label="Unlock virtual teacher">
        <span className="virtual-teacher-unlock-mark" aria-hidden="true">♪</span>
        <div>
          <p className="eyebrow">Optional learning guide</p>
          <h2>Virtual teacher</h2>
        </div>
        <button type="button" className="primary" onClick={() => setUnlockChoicesOpen((current) => !current)} aria-expanded={unlockChoicesOpen}>
          Unlock virtual teacher
        </button>
        {unlockChoicesOpen && (
          <div className="virtual-teacher-unlock-choices">
            <button type="button" className="primary" onClick={() => unlockTeacher(true)}>
              <strong>Show teacher</strong>
              <small>See the animated teacher beside your lesson.</small>
            </button>
            <button type="button" className="ghost" onClick={() => unlockTeacher(false)}>
              <strong>Voice only</strong>
              <small>Keep the avatar hidden. Her voice will connect later.</small>
            </button>
          </div>
        )}
      </section>
    );
  }

  return (
    <>
      <section className="virtual-teacher-toolbar" aria-label="Virtual teacher controls">
        <span>
          <i aria-hidden="true" />
          {showTeacherPhysically
            ? isPlaying ? 'Teacher is following' : 'Teacher is roaming'
            : 'Teacher active · avatar hidden'}
        </span>
        <div>
          <button type="button" className="ghost" onClick={togglePhysicalTeacher}>
            {showTeacherPhysically ? 'Avatar off' : 'Show avatar'}
          </button>
          {showTeacherPhysically && (
            <button
              type="button"
              className={`ghost virtual-teacher-preview ${isSpeaking ? 'active' : ''}`}
              onClick={() => setPreviewSpeaking((current) => !current)}
              aria-pressed={isSpeaking}
            >
              {isSpeaking ? 'Stop' : 'Preview'}
            </button>
          )}
          <button type="button" className="ghost virtual-teacher-hide" onClick={hideTeacher}>
            Close
          </button>
        </div>
      </section>

      {showTeacherPhysically && (
        <div className="virtual-teacher-style-picker" role="group" aria-label="Choose virtual teacher appearance">
          {[
            ['regular', 'Regular'],
            ['bikini', 'Bikini'],
          ].map(([style, label]) => (
            <button
              type="button"
              key={style}
              className={teacherStyle === style ? 'active' : ''}
              onClick={() => chooseTeacherStyle(style)}
              aria-pressed={teacherStyle === style}
              title={`${label} teacher`}
            >
              <span className={`virtual-teacher-style-icon is-${style}`} aria-hidden="true" />
              <small>{label}</small>
            </button>
          ))}
        </div>
      )}

      <TeacherProjectionControls
        style={teacherStyle}
        speaking={isSpeaking}
        caption={[...coachingObservations].reverse().find((entry) => entry.type !== 'correct')?.message || ''}
        onProjectionStarted={hideLocalTeacherForProjection}
      />

      <TeacherArControls
        style={teacherStyle}
        speaking={isSpeaking}
        caption={[...coachingObservations].reverse().find((entry) => entry.type !== 'correct')?.message || ''}
        onArStarted={hideLocalTeacherForProjection}
      />

      <TeacherSensesPanel
        ref={sensesRef}
        deviceClass={deviceClass}
        performanceTier={performanceTier}
        currentTime={currentTime}
        onObservation={handleObservation}
      />

      <TeacherConversationPanel
        song={song}
        currentTime={currentTime}
        observations={coachingObservations}
        sensesRef={sensesRef}
        onSpeakingChange={setTeacherSpeaking}
      />

      {showTeacherPhysically && (
        <aside
          className={`virtual-teacher-free style-${teacherStyle} ${isSpeaking ? 'is-speaking' : 'is-idle'}`}
          aria-hidden="true"
        >
          <div className="virtual-teacher-roam">
            <div className="virtual-teacher-stage">
              <div className="virtual-teacher-motion">
                <div className="virtual-teacher-frame virtual-teacher-idle" />
                <div className="virtual-teacher-frame virtual-teacher-speaking" />
              </div>
            </div>
            <div className="virtual-teacher-wave" aria-hidden="true">
              <i /><i /><i /><i /><i />
            </div>
          </div>
        </aside>
      )}
    </>
  );
}
