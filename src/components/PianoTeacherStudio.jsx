import { useEffect, useMemo, useState } from 'react';
import {
  buildTeacherHandTargets,
  prepareTeacherHandTimeline,
  teacherReply,
  TEACHER_PROFILES,
} from '../engine/teacherHands.js';

const WHITE_KEYS = Array.from({ length: 52 }, (_, index) => index);
const BLACK_KEY_PATTERN = [0, 1, 3, 4, 5];
const BLACK_KEYS = Array.from({ length: 7 }, (_, octave) => (
  BLACK_KEY_PATTERN.map((offset) => octave * 7 + offset)
)).flat().filter((index) => index < 51);

function TeacherPortrait({ teacher }) {
  return (
    <span
      className={`teacher-portrait teacher-portrait-${teacher.look}`}
      style={{
        '--teacher-skin': teacher.palette.skin,
        '--teacher-hair': teacher.palette.hair,
        '--teacher-primary': teacher.palette.primary,
      }}
      aria-hidden="true"
    >
      <span className="teacher-portrait-hair" />
      <span className="teacher-portrait-face"><i /><i /></span>
      <span className="teacher-portrait-outfit" />
    </span>
  );
}

function FingerHand({ side, x, palette, target, visible }) {
  const activeFingers = new Set(target.notes.map((note) => note.finger));
  return (
    <g
      className={`teacher-svg-hand teacher-svg-hand-${side} ${target.isPressing ? 'is-pressing' : ''} ${visible ? 'is-visible' : 'is-resting'}`}
      transform={`translate(${x} 333)`}
      aria-hidden="true"
    >
      <ellipse cx="0" cy="0" rx="26" ry="14" fill={palette.skin} />
      {[1, 2, 3, 4, 5].map((finger, index) => {
        const direction = side === 'left' ? -1 : 1;
        const localX = direction * (index - 2) * 8;
        const length = 17 + (finger === 3 ? 5 : finger === 2 || finger === 4 ? 3 : 0);
        return (
          <g key={finger} className={activeFingers.has(finger) ? 'is-active-finger' : ''}>
            <line x1={localX} y1="-4" x2={localX + direction * 2} y2={-length} stroke={palette.skin} strokeWidth="7" strokeLinecap="round" />
            {activeFingers.has(finger) && <circle cx={localX + direction * 2} cy={-length} r="5.5" />}
          </g>
        );
      })}
    </g>
  );
}

function Outfit({ teacher }) {
  const { look, palette } = teacher;
  if (look === 'athletic') {
    return (
      <>
        <path d="M438 194 Q500 174 562 194 L548 252 Q500 269 452 252Z" fill={palette.primary} />
        <path d="M451 215 Q475 194 500 220 Q525 194 549 215" fill="none" stroke={palette.secondary} strokeWidth="5" />
        <path d="M455 270 Q500 258 545 270 L554 322 L446 322Z" fill="#27213f" />
        <path d="M468 270 L460 318 M532 270 L540 318" stroke={palette.secondary} strokeWidth="4" opacity=".65" />
      </>
    );
  }
  if (look === 'athletic-male') {
    return (
      <>
        <path d="M442 190 Q500 168 558 190 L548 294 Q500 316 452 294Z" fill={palette.skin} />
        <path d="M500 206 V286 M472 222 Q500 238 528 222 M474 250 Q500 266 526 250 M477 278 Q500 290 523 278" fill="none" stroke="#95694f" strokeWidth="4" opacity=".7" />
        <path d="M448 286 Q500 272 552 286 L555 324 L445 324Z" fill={palette.primary} />
      </>
    );
  }
  if (look === 'blue-dress') {
    return (
      <path d="M442 191 Q500 174 558 191 Q548 231 570 318 L430 318 Q452 231 442 191Z" fill={palette.primary} />
    );
  }
  return (
    <>
      <path d="M438 191 Q500 171 562 191 L552 307 L448 307Z" fill={palette.primary} />
      <path d="M472 190 L500 234 L528 190" fill={palette.secondary} opacity=".88" />
      <path d="M500 234 V306" stroke="#221a31" strokeWidth="7" />
    </>
  );
}

function TeacherFigure({ teacher, targets, showHands, isPlaying }) {
  const leftX = showHands ? 100 + targets.left.centerPercent * 8 : 420;
  const rightX = showHands ? 100 + targets.right.centerPercent * 8 : 580;
  const leftShoulder = [454, 208];
  const rightShoulder = [546, 208];
  const status = showHands && isPlaying ? 'Demonstrating' : showHands ? 'Hands ready' : 'Teacher seated';

  return (
    <div className={`teacher-performance-stage ${showHands ? 'hands-unlocked' : ''}`}>
      <svg viewBox="0 0 1000 440" role="img" aria-label={`${teacher.name} seated at a piano. ${status}.`}>
        <defs>
          <linearGradient id="teacher-stage-sky" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#171538" />
            <stop offset="1" stopColor="#30266a" />
          </linearGradient>
          <linearGradient id="teacher-piano-case" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#23264c" />
            <stop offset="1" stopColor="#080a19" />
          </linearGradient>
        </defs>
        <rect width="1000" height="440" rx="34" fill="url(#teacher-stage-sky)" />
        <circle cx="120" cy="75" r="2" fill="#d7cfff" /><circle cx="850" cy="95" r="3" fill="#cfbfff" />
        <circle cx="770" cy="52" r="2" fill="#fff" /><circle cx="262" cy="118" r="2.5" fill="#fff" />

        <g className={`seated-teacher seated-teacher-${teacher.look}`}>
          <rect x="425" y="290" width="150" height="24" rx="10" fill="#18162a" />
          <path d="M447 308 L422 401 M553 308 L578 401" stroke="#161321" strokeWidth="14" strokeLinecap="round" />
          <path d="M467 292 L455 374 L424 414 M533 292 L545 374 L576 414" fill="none" stroke={teacher.look === 'blue-dress' ? teacher.palette.primary : '#27253f'} strokeWidth="24" strokeLinecap="round" strokeLinejoin="round" />
          <Outfit teacher={teacher} />
          <rect x="484" y="161" width="32" height="33" rx="13" fill={teacher.palette.skin} />
          <ellipse cx="500" cy="132" rx="48" ry="55" fill={teacher.palette.skin} />
          {teacher.look === 'master' ? (
            <path d="M457 119 Q500 70 543 119 Q531 82 500 78 Q469 82 457 119Z" fill={teacher.palette.hair} opacity=".35" />
          ) : (
            <path d="M450 137 Q441 73 500 66 Q559 75 550 137 Q539 103 515 96 Q485 91 452 123Z" fill={teacher.palette.hair} />
          )}
          {(teacher.look === 'athletic' || teacher.look === 'blue-dress') && (
            <path d="M453 116 Q437 177 457 222 M547 116 Q563 177 543 222" fill="none" stroke={teacher.palette.hair} strokeWidth="22" strokeLinecap="round" />
          )}
          {teacher.look === 'athletic-male' && <path d="M455 115 Q477 73 531 85 L548 112 Q514 97 475 119Z" fill={teacher.palette.hair} />}
          <path d="M477 128 Q486 121 494 128 M506 128 Q514 121 523 128" fill="none" stroke="#292136" strokeWidth="4" strokeLinecap="round" />
          <path d="M486 154 Q500 163 514 154" fill="none" stroke="#984e61" strokeWidth="3" strokeLinecap="round" />
          <circle cx="486" cy="128" r="2.5" fill="#171424" /><circle cx="514" cy="128" r="2.5" fill="#171424" />

          <path d={`M${leftShoulder[0]} ${leftShoulder[1]} C420 244 ${leftX - 42} 270 ${leftX} 333`} fill="none" stroke={teacher.palette.skin} strokeWidth="22" strokeLinecap="round" />
          <path d={`M${rightShoulder[0]} ${rightShoulder[1]} C580 244 ${rightX + 42} 270 ${rightX} 333`} fill="none" stroke={teacher.palette.skin} strokeWidth="22" strokeLinecap="round" />
        </g>

        <g className="teacher-mini-piano">
          <rect x="72" y="315" width="856" height="116" rx="16" fill="url(#teacher-piano-case)" />
          <rect x="100" y="330" width="800" height="84" rx="6" fill="#eceef8" />
          {WHITE_KEYS.map((index) => (
            <line key={index} x1={100 + (index * 800 / 52)} y1="330" x2={100 + (index * 800 / 52)} y2="414" stroke="#9fa4ba" strokeWidth="1" />
          ))}
          {BLACK_KEYS.map((index) => (
            <rect key={index} x={100 + ((index + 0.72) * 800 / 52)} y="330" width="8.6" height="49" rx="2" fill="#101224" />
          ))}
          {showHands && [...targets.left.notes, ...targets.right.notes].map((note) => (
            <g key={`${note.hand}-${note.id}`} className={note.time <= 0 ? '' : 'teacher-target-marker'}>
              <circle cx={100 + note.percent * 8} cy="397" r="10" fill={note.hand === 'left' ? '#835cff' : '#df4b9f'} />
              <text x={100 + note.percent * 8} y="401" textAnchor="middle" fontSize="10" fontWeight="800" fill="white">{note.finger}</text>
            </g>
          ))}
          <FingerHand side="left" x={leftX} palette={teacher.palette} target={targets.left} visible={showHands} />
          <FingerHand side="right" x={rightX} palette={teacher.palette} target={targets.right} visible={showHands} />
        </g>
      </svg>
      <span className={`teacher-live-status ${isPlaying && showHands ? 'is-live' : ''}`}><i />{status}</span>
    </div>
  );
}

function targetSentence(targets, showHands) {
  if (!showHands) return 'Unlock the teacher’s hands to see fingering.';
  const notes = [...targets.left.notes, ...targets.right.notes];
  if (!notes.length) return 'Hands are resting until the next note.';
  return notes.map((note) => `${note.hand} ${note.note} · finger ${note.finger}`).join('  |  ');
}

export default function PianoTeacherStudio({ song, currentTime, isPlaying, handMode }) {
  const [teacherId, setTeacherId] = useState(() => window.localStorage.getItem('polymath-piano-teacher') || 'padme');
  const [showHands, setShowHands] = useState(() => window.localStorage.getItem('polymath-teacher-hands') === 'true');
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const teacher = TEACHER_PROFILES.find((profile) => profile.id === teacherId) || TEACHER_PROFILES[0];
  const timeline = useMemo(() => prepareTeacherHandTimeline(song?.notes || []), [song]);
  const targets = useMemo(
    () => buildTeacherHandTargets(timeline, currentTime, { handMode }),
    [timeline, currentTime, handMode],
  );

  useEffect(() => {
    window.localStorage.setItem('polymath-piano-teacher', teacher.id);
  }, [teacher.id]);

  useEffect(() => {
    window.localStorage.setItem('polymath-teacher-hands', String(showHands));
  }, [showHands]);

  function chooseTeacher(profile) {
    setTeacherId(profile.id);
    setMessages([{ from: 'teacher', text: `${profile.name} selected. I’m ready to demonstrate this lesson.` }]);
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
          onClick={() => setShowHands((current) => !current)}
        >
          {showHands ? 'Hide teacher’s hands' : 'Unlock teacher’s hands'}
        </button>
      </header>

      <div className="piano-teacher-workspace">
        <div>
          <TeacherFigure teacher={teacher} targets={targets} showHands={showHands} isPlaying={isPlaying} />
          <p className="teacher-hand-readout" aria-live="polite">{targetSentence(targets, showHands)}</p>
        </div>

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
