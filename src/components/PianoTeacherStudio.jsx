import { useEffect, useState } from 'react';
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
  targets,
  isPlaying = false,
  practiceReport = null,
  practiceOutcome = null,
  masteryProfile = null,
  coachPlan = null,
  lessonContext = null,
  user = null,
  setUser,
  onNavigate,
  onDemonstrate,
  performanceTier = 'balanced',
}) {
  const [confirmedAge, setConfirmedAge] = useState(() => {
    const stored = Number(window.localStorage.getItem('polymath-teacher-confirmed-age') || 0);
    const legacyAdult = window.localStorage.getItem('polymath-teacher-adult-confirmed') === 'true' ? 18 : 0;
    return Math.max(Number.isFinite(stored) ? stored : 0, legacyAdult);
  });
  const [pendingAgeTeacher, setPendingAgeTeacher] = useState(null);
  const [lessonLock, setLessonLock] = useState(null);
  const selectedTeacher = profiles.find((profile) => profile.id === teacherId)
    || profiles.find((profile) => profile.id === 'aria')
    || profiles[0];
  const lockedProfile = lessonLock?.teacher?.id
    ? profiles.find((profile) => profile.id === lessonLock.teacher.id)
      || TEACHER_PROFILES.find((profile) => profile.id === lessonLock.teacher.id)
      || selectedTeacher
    : null;
  const teacher = lessonLock?.teacher
    ? { ...lockedProfile, ...lessonLock.teacher }
    : selectedTeacher;

  useEffect(() => {
    const lockedTeacherId = lessonLock?.teacher?.id;
    if (lockedTeacherId && lockedTeacherId !== teacherId) onTeacherChange(lockedTeacherId);
  }, [lessonLock?.teacher?.id, onTeacherChange, teacherId]);

  function chooseTeacher(profile) {
    if (lessonLock) return;
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
          <h3 id="piano-teacher-title">{teacher.name} guides your main piano</h3>
          <p>The next keys light up directly. Nothing covers your keyboard.</p>
        </div>
      </header>

      <div className="piano-teacher-workspace human-teacher-workspace">
        <div className="teacher-main-piano-status" role="status">
          <TeacherPortrait teacher={teacher} />
          <span>
            <strong>{isPlaying ? `${teacher.name} is guiding the coloured keys` : `${teacher.name} is ready to guide you`}</strong>
            <small>{[...(targets?.left?.notes || []), ...(targets?.right?.notes || [])].map((note) => note.note).join(', ') || 'Press Play and follow the keys that light up above.'}</small>
          </span>
        </div>

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
              practiceOutcome,
              masteryProfile: masteryProfile ? {
                overall: masteryProfile.overall,
                attempts: masteryProfile.attempts,
                measuredSkillCount: masteryProfile.measuredSkillCount,
                skills: masteryProfile.skills?.map((skill) => ({
                  id: skill.id,
                  label: skill.label,
                  score: skill.score,
                  observations: skill.observations,
                  confidence: skill.confidence,
                  trend: skill.trend,
                })),
              } : null,
              coachPlan: coachPlan ? {
                source: coachPlan.source,
                skillId: coachPlan.skillId,
                skillLabel: coachPlan.skillLabel,
                title: coachPlan.title,
                reason: coachPlan.reason,
                instruction: coachPlan.instruction,
                successRule: coachPlan.successRule,
                speedPercent: coachPlan.speedPercent,
                confidence: coachPlan.confidence,
              } : null,
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
            onSessionLockChange={setLessonLock}
            handTargets={targets}
            teacherIsPlaying={isPlaying}
            performanceTier={performanceTier}
          />
        </details>

        {lessonLock ? (
          <div className="teacher-session-lock" role="status" data-locked-teacher-id={teacher.id}>
            <TeacherPortrait teacher={teacher} />
            <span>
              <strong>Locked to {teacher.name}</strong>
              <small>You paid for {teacher.name}, so this teacher stays with you until the session ends.</small>
            </span>
            <span aria-hidden="true">🔒</span>
          </div>
        ) : (
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
        )}
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
