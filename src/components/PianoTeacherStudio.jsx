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
  const teacher = TEACHER_PROFILES.find((profile) => profile.id === teacherId) || TEACHER_PROFILES[0];

  function chooseTeacher(profile) {
    onTeacherChange(profile.id);
    setMessages([{ from: 'teacher', text: `${profile.name} selected. I’m ready at the main piano.` }]);
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
            <span><strong>{profile.name}</strong><small>{profile.title}</small></span>
          </button>
        ))}
      </div>
    </section>
  );
}
