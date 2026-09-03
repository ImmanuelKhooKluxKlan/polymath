import { useState } from 'react';
import { teacherReply, TEACHER_PROFILES } from '../engine/teacherHands.js';
import PoseableTeacherStage from './PoseableTeacherStage.jsx';

function TeacherPortrait({ teacher }) {
  return (
    <span className="teacher-human-portrait" aria-hidden="true">
      <img src={teacher.image} alt="" loading="lazy" draggable="false" />
    </span>
  );
}

function targetSentence(targets, showHands) {
  if (!showHands) return 'Choose a teacher, then place their hands on the main keyboard.';
  const notes = [...(targets?.left?.notes || []), ...(targets?.right?.notes || [])];
  if (!notes.length) return 'The teacher’s hands are resting until the next note.';
  return notes.map((note) => `${note.hand} ${note.note} · finger ${note.finger}`).join('  |  ');
}

export default function PianoTeacherStudio({
  teacherId,
  onTeacherChange,
  showHands,
  onShowHandsChange,
  targets,
}) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [adultConfirmed, setAdultConfirmed] = useState(
    () => window.localStorage.getItem('polymath-teacher-adult-confirmed') === 'true',
  );
  const [pendingAdultTeacher, setPendingAdultTeacher] = useState(null);
  const teacher = TEACHER_PROFILES.find((profile) => profile.id === teacherId)
    || TEACHER_PROFILES.find((profile) => profile.id === 'anakin')
    || TEACHER_PROFILES[0];

  function chooseTeacher(profile) {
    if (profile.requiresAdultConfirmation && !adultConfirmed) {
      setPendingAdultTeacher(profile);
      return;
    }
    onTeacherChange(profile.id);
    setMessages([{ from: 'teacher', text: `${profile.name} selected. I’m ready at the main piano.` }]);
  }

  function confirmAdultTeacher() {
    if (!pendingAdultTeacher) return;
    window.localStorage.setItem('polymath-teacher-adult-confirmed', 'true');
    setAdultConfirmed(true);
    onTeacherChange(pendingAdultTeacher.id);
    setMessages([{ from: 'teacher', text: `${pendingAdultTeacher.name} selected. I’m ready at the main piano.` }]);
    setPendingAdultTeacher(null);
  }

  function submitChat(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setMessages((current) => [
      ...current,
      { from: 'student', text: message },
      { from: 'teacher', text: teacherReply(teacher, message, targets) },
    ].slice(-6));
    setDraft('');
  }

  return (
    <section className="piano-teacher-studio" aria-labelledby="piano-teacher-title">
      <header className="piano-teacher-header">
        <div>
          <p className="eyebrow">Virtual piano teacher</p>
          <h3 id="piano-teacher-title">Learn with {teacher.name}</h3>
          <p>{teacher.title} · {teacher.voice}</p>
        </div>
        <button
          type="button"
          className={showHands ? 'teacher-hands-toggle is-on' : 'teacher-hands-toggle'}
          aria-pressed={showHands}
          onClick={() => onShowHandsChange(!showHands)}
        >
          {showHands ? 'Remove teacher from piano' : 'Show teacher on main piano'}
        </button>
      </header>

      <div className="piano-teacher-workspace human-teacher-workspace">
        <PoseableTeacherStage teacher={teacher} targetSummary={targetSentence(targets, showHands)} />

        <div className="teacher-chat-panel">
          <div className="teacher-chat-heading">
            <TeacherPortrait teacher={teacher} />
            <span><strong>Chat with {teacher.name}</strong><small>Lesson-aware text coach</small></span>
          </div>
          <div className="teacher-chat-messages" aria-live="polite">
            {!messages.length && <p className="teacher-chat-empty">Ask which hand, finger, note, or speed to practise.</p>}
            {messages.map((message, index) => (
              <p key={`${message.from}-${index}`} className={`teacher-message teacher-message-${message.from}`}>{message.text}</p>
            ))}
          </div>
          <form className="teacher-chat-form" onSubmit={submitChat}>
            <label className="sr-only" htmlFor="piano-teacher-message">Message {teacher.name}</label>
            <input id="piano-teacher-message" value={draft} onChange={(event) => setDraft(event.target.value)} maxLength="280" placeholder={`Ask ${teacher.name}…`} />
            <button type="submit" className="primary">Send</button>
          </form>
        </div>
      </div>

      <div className="teacher-roster" role="group" aria-label="Choose a virtual piano teacher">
        {TEACHER_PROFILES.map((profile) => (
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
              <small>{profile.title}{profile.requiresAdultConfirmation && !adultConfirmed ? ' · 18+' : ''}</small>
            </span>
          </button>
        ))}
      </div>

      {pendingAdultTeacher && (
        <div className="teacher-age-gate" role="dialog" aria-modal="true" aria-labelledby="teacher-age-gate-title">
          <div>
            <TeacherPortrait teacher={pendingAdultTeacher} />
            <span>
              <strong id="teacher-age-gate-title">Confirm you are 18+</strong>
              <small>Nova is an adult-only optional character. This confirmation is stored on this device.</small>
            </span>
          </div>
          <div className="teacher-age-gate-actions">
            <button type="button" onClick={() => setPendingAdultTeacher(null)}>Cancel</button>
            <button type="button" className="primary" onClick={confirmAdultTeacher}>I am 18 or older</button>
          </div>
        </div>
      )}
    </section>
  );
}
