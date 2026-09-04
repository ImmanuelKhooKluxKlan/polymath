import { useState } from 'react';
import { TEACHER_PROFILES } from '../engine/teacherHands.js';
import VirtualLessonPanel from './VirtualLessonPanel.jsx';

function TeacherPortrait({ teacher }) {
  return (
    <span className="teacher-human-portrait" aria-hidden="true">
      <img
        src={teacher.image}
        alt=""
        loading="lazy"
        draggable="false"
        style={teacher.portraitPosition ? { objectPosition: teacher.portraitPosition } : undefined}
      />
    </span>
  );
}

function teacherMinimumAge(profile) {
  const configured = Number(profile?.minimumAge);
  if (Number.isFinite(configured) && configured > 0) return Math.min(99, Math.floor(configured));
  return profile?.requiresAdultConfirmation ? 18 : 0;
}

export default function PianoTeacherStudio({
  profiles = TEACHER_PROFILES,
  teacherId,
  onTeacherChange,
  showHands,
  onShowHandsChange,
  targets,
  isPlaying = false,
  practiceReport = null,
  lessonContext = null,
  user = null,
  setUser,
  onNavigate,
  onDemonstrate,
}) {
  const [confirmedAge, setConfirmedAge] = useState(() => {
    const stored = Number(window.localStorage.getItem('polymath-teacher-confirmed-age') || 0);
    const legacyAdult = window.localStorage.getItem('polymath-teacher-adult-confirmed') === 'true' ? 18 : 0;
    return Math.max(Number.isFinite(stored) ? stored : 0, legacyAdult);
  });
  const [pendingAgeTeacher, setPendingAgeTeacher] = useState(null);
  const teacher = profiles.find((profile) => profile.id === teacherId)
    || profiles.find((profile) => profile.id === 'aria')
    || profiles[0];

  function chooseTeacher(profile) {
    if (teacherMinimumAge(profile) > confirmedAge) {
      setPendingAgeTeacher(profile);
      return;
    }
    onTeacherChange(profile.id);
  }

  function confirmTeacherAge() {
    if (!pendingAgeTeacher) return;
    const minimumAge = teacherMinimumAge(pendingAgeTeacher);
    const nextConfirmedAge = Math.max(confirmedAge, minimumAge);
    window.localStorage.setItem('polymath-teacher-confirmed-age', String(nextConfirmedAge));
    if (nextConfirmedAge >= 18) window.localStorage.setItem('polymath-teacher-adult-confirmed', 'true');
    setConfirmedAge(nextConfirmedAge);
    onTeacherChange(pendingAgeTeacher.id);
    setPendingAgeTeacher(null);
  }

  return (
    <section className="piano-teacher-studio" aria-labelledby="piano-teacher-title">
      <header className="piano-teacher-header">
        <div>
          <p className="eyebrow">Teacher controls</p>
          <h3 id="piano-teacher-title">{teacher.name} plays your main piano</h3>
          <p>The hands above follow this lesson on the same keys as the falling notes.</p>
        </div>
        <button
          type="button"
          className={showHands ? 'teacher-hands-toggle is-on' : 'teacher-hands-toggle'}
          aria-pressed={showHands}
          onClick={() => onShowHandsChange(!showHands)}
        >
          {showHands ? 'Hide hands' : 'Show hands'}
        </button>
      </header>

      <div className="piano-teacher-workspace human-teacher-workspace">
        {showHands ? (
          <div className="teacher-main-piano-status" role="status">
            <TeacherPortrait teacher={teacher} />
            <span>
              <strong>{isPlaying ? `${teacher.name} is demonstrating on the main piano` : `${teacher.name} is ready on the main piano`}</strong>
              <small>{[...(targets?.left?.notes || []), ...(targets?.right?.notes || [])].map((note) => note.note).join(', ') || 'Press Play to follow the teacher hands above.'}</small>
            </span>
          </div>
        ) : (
          <button type="button" className="teacher-stage-placeholder" onClick={() => onShowHandsChange(true)}>
            <TeacherPortrait teacher={teacher} />
            <span>
              <strong>Show {teacher.name}'s hands</strong>
              <small>The hands will appear directly on the main piano above.</small>
            </span>
          </button>
        )}

        <details className="teacher-studio-disclosure">
          <summary>Private voice session · Choose time and style</summary>
          <VirtualLessonPanel
            user={user}
            setUser={setUser}
            teacher={teacher}
            onTeacherChange={onTeacherChange}
            lessonContext={lessonContext}
            observations={{
              practiceReport,
              upcomingKeys: [...(targets?.left?.notes || []), ...(targets?.right?.notes || [])]
                .sort((left, right) => Number(left.time || 0) - Number(right.time || 0))
                .slice(0, 14)
                .map((target) => ({
                  note: target.note,
                  hand: target.hand,
                  time: target.time,
                  duration: target.duration,
                })),
            }}
            onDemonstrate={onDemonstrate}
            onNavigate={onNavigate}
          />
        </details>

        <details className="teacher-studio-disclosure">
          <summary>Choose another teacher</summary>
          <div className="teacher-roster" role="group" aria-label="Choose a virtual piano teacher">
            {profiles.map((profile) => (
              <button
                type="button"
                key={profile.id}
                className={profile.id === teacher.id ? 'teacher-choice is-selected' : 'teacher-choice'}
                aria-pressed={profile.id === teacher.id}
                onClick={() => chooseTeacher(profile)}
              >
                <TeacherPortrait teacher={profile} />
                <span>
                  <strong>{profile.name}</strong>
                  <small>
                    {profile.title}
                    {teacherMinimumAge(profile) > 0 ? ` · ${teacherMinimumAge(profile)}+` : ''}
                    {profile.effectivePricePer30MinutesMcoins !== null && profile.effectivePricePer30MinutesMcoins !== undefined
                      ? ` · ${Number(profile.effectivePricePer30MinutesMcoins).toFixed(2)} Mcoins / 30 min`
                      : ''}
                  </small>
                </span>
              </button>
            ))}
          </div>
        </details>
      </div>

      {pendingAgeTeacher && (
        <div className="teacher-age-gate" role="dialog" aria-modal="true" aria-labelledby="teacher-age-gate-title">
          <div>
            <TeacherPortrait teacher={pendingAgeTeacher} />
            <span>
              <strong id="teacher-age-gate-title">Confirm you are {teacherMinimumAge(pendingAgeTeacher)}+</strong>
              <small>{pendingAgeTeacher.name} has an administrator-set age requirement. This confirmation is stored on this device.</small>
            </span>
          </div>
          <div className="teacher-age-gate-actions">
            <button type="button" onClick={() => setPendingAgeTeacher(null)}>Cancel</button>
            <button type="button" className="primary" onClick={confirmTeacherAge}>I am {teacherMinimumAge(pendingAgeTeacher)} or older</button>
          </div>
        </div>
      )}
    </section>
  );
}
