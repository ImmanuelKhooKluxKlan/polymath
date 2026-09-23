import { useEffect, useMemo, useState } from 'react';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import { apiRequest } from '../services/api.js';
import ReputationScore from '../components/ReputationScore.jsx';

const LEVELS = [
  ['beginner', 'Beginner'],
  ['intermediate', 'Intermediate'],
  ['advanced', 'Advanced'],
];

const LESSON_MODES = [
  ['online', 'Online'],
  ['in-person', 'In person'],
];

const EMPTY_FORM = {
  headline: '',
  bio: '',
  instruments: ['piano'],
  levels: ['beginner'],
  lessonModes: ['online'],
  location: '',
  languages: 'English',
  availability: '',
  hourlyRateMcoins: 0,
  published: true,
};

const DEFAULT_MARKETPLACE = Object.freeze({
  directoryEnabled: true,
  applicationsEnabled: true,
  reviewsEnabled: true,
  minimumHourlyRateMcoins: 0,
  maximumHourlyRateMcoins: 100000,
  platformFeePercent: 25,
  teacherKeepsPercent: 75,
  withdrawalFeePercent: 25,
  notice: '',
  checkoutAvailable: false,
});

const DEFAULT_LESSON_CONFIG = Object.freeze({
  durationStepMinutes: 10,
  callCostPerStepMcoins: 0.2,
  breakoutRoomCostMcoins: 0.5,
  minimumTransferMcoins: 30,
});

const SAVED_INVITE_KEY = 'polymath_human_lesson_invite';

function localDateTimeValue(offsetMinutes = 30) {
  const date = new Date(Date.now() + (offsetMinutes * 60_000));
  date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
  return date.toISOString().slice(0, 16);
}

function percentLabel(value) {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function initials(name) {
  return String(name || 'Music teacher')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase();
}

function titleCase(value) {
  return String(value || '').replace(/(^|[\s-])\S/g, (letter) => letter.toUpperCase());
}

function TeacherAvatar({ teacher, size = 'normal' }) {
  return teacher?.avatarUrl
    ? <img className={`teacher-avatar ${size}`} src={teacher.avatarUrl} alt={`${teacher.name} profile`} />
    : <span className={`teacher-avatar teacher-avatar-fallback ${size}`} aria-hidden="true">{initials(teacher?.name)}</span>;
}

function TeacherStars({ value = 0, onChange = null, label = 'Rating' }) {
  const rounded = Math.round(Number(value) || 0);
  return (
    <div className={`teacher-stars ${onChange ? 'interactive' : ''}`} aria-label={`${label}: ${Number(value || 0).toFixed(1)} out of 5`}>
      {[1, 2, 3, 4, 5].map((star) => onChange ? (
        <button
          key={star}
          type="button"
          className={star <= rounded ? 'filled' : ''}
          aria-label={`${star} star${star === 1 ? '' : 's'}`}
          onClick={() => onChange(star)}
        >&#9733;</button>
      ) : <span key={star} className={star <= rounded ? 'filled' : ''} aria-hidden="true">&#9733;</span>)}
    </div>
  );
}

function toggleChoice(values, value) {
  return values.includes(value) ? values.filter((entry) => entry !== value) : [...values, value];
}

function formFromTeacher(teacher) {
  if (!teacher) return { ...EMPTY_FORM };
  return {
    headline: teacher.headline || '',
    bio: teacher.bio || '',
    instruments: teacher.instruments || ['piano'],
    levels: teacher.levels || ['beginner'],
    lessonModes: teacher.lessonModes || ['online'],
    location: teacher.location || '',
    languages: (teacher.languages || []).map(titleCase).join(', '),
    availability: teacher.availability || '',
    hourlyRateMcoins: Number(teacher.hourlyRateMcoins || 0),
    published: teacher.published !== false,
  };
}

export default function TeacherMarketplacePage({ user, setUser, onNavigate }) {
  const [teachers, setTeachers] = useState([]);
  const [ownTeacher, setOwnTeacher] = useState(null);
  const [marketplace, setMarketplace] = useState(DEFAULT_MARKETPLACE);
  const [filters, setFilters] = useState({ query: '', instrument: '', lessonMode: '', level: '' });
  const [showTeacherForm, setShowTeacherForm] = useState(false);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [openReviews, setOpenReviews] = useState('');
  const [reviewsByTeacher, setReviewsByTeacher] = useState({});
  const [reviewDrafts, setReviewDrafts] = useState({});
  const [status, setStatus] = useState('');
  const [saving, setSaving] = useState(false);
  const [lessonTools, setLessonTools] = useState('');
  const [lessons, setLessons] = useState([]);
  const [lessonConfig, setLessonConfig] = useState(DEFAULT_LESSON_CONFIG);
  const [contacts, setContacts] = useState([]);
  const [lessonDraft, setLessonDraft] = useState({
    title: 'Private music lesson', durationMinutes: 60, scheduledFor: localDateTimeValue(),
  });
  const [joinDraft, setJoinDraft] = useState({ meetingId: '', accessCode: '' });
  const [inviteTargets, setInviteTargets] = useState({});
  const [breakoutNames, setBreakoutNames] = useState({});
  const [lessonSaving, setLessonSaving] = useState(false);

  async function loadTeachers() {
    try {
      const data = await apiRequest('/api/teachers');
      setTeachers(data.teachers || []);
      setMarketplace((current) => ({ ...current, ...(data.marketplace || {}) }));
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function loadOwnTeacher() {
    if (!user) {
      setOwnTeacher(null);
      setForm({ ...EMPTY_FORM });
      return;
    }
    try {
      const data = await apiRequest('/api/teachers/me');
      setOwnTeacher(data.teacher || null);
      setForm(formFromTeacher(data.teacher));
      setMarketplace((current) => ({ ...current, ...(data.marketplace || {}) }));
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function loadLessonWorkspace() {
    if (!user) {
      setLessons([]);
      setContacts([]);
      return;
    }
    try {
      const [lessonData, threadData] = await Promise.all([
        apiRequest('/api/human-lessons'),
        apiRequest('/api/messages/threads'),
      ]);
      setLessons(lessonData.meetings || []);
      setLessonConfig({ ...DEFAULT_LESSON_CONFIG, ...(lessonData.config || {}) });
      setContacts((threadData.threads || []).map((thread) => thread.otherUser).filter(Boolean));
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadTeachers(); }, [user?.user_id]);
  useEffect(() => { loadOwnTeacher(); }, [user?.user_id]);
  useEffect(() => { loadLessonWorkspace(); }, [user?.user_id]);

  const filteredTeachers = useMemo(() => teachers.filter((teacher) => {
    const text = [
      teacher.name,
      teacher.headline,
      teacher.bio,
      teacher.location,
      ...(teacher.instruments || []),
      ...(teacher.languages || []),
    ].join(' ').toLowerCase();
    return (!filters.query || text.includes(filters.query.toLowerCase()))
      && (!filters.instrument || teacher.instruments.includes(filters.instrument))
      && (!filters.lessonMode || teacher.lessonModes.includes(filters.lessonMode))
      && (!filters.level || teacher.levels.includes(filters.level));
  }), [teachers, filters]);

  function openTeacherEditor() {
    if (!user) {
      onNavigate('account', { next: 'find-teacher' });
      return;
    }
    if (!ownTeacher && !marketplace.applicationsEnabled) {
      setStatus('New teacher profiles are temporarily paused by the administrator.');
      return;
    }
    setForm(formFromTeacher(ownTeacher));
    setShowTeacherForm(true);
    window.setTimeout(() => document.getElementById('teacher-profile-editor')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0);
  }

  async function saveTeacherProfile(event) {
    event.preventDefault();
    setSaving(true);
    setStatus('Saving teacher profile...');
    try {
      const data = await apiRequest('/api/teachers/me', {
        method: 'PUT',
        body: JSON.stringify({
          ...form,
          hourlyRateMcoins: Number(form.hourlyRateMcoins || 0),
          languages: form.languages.split(',').map((language) => language.trim()).filter(Boolean),
        }),
      });
      setOwnTeacher(data.teacher);
      setForm(formFromTeacher(data.teacher));
      setShowTeacherForm(false);
      setStatus(data.teacher.published ? 'Your teacher profile is live.' : 'Your teacher profile is hidden.');
      await loadTeachers();
    } catch (error) {
      setStatus(error.message);
    } finally {
      setSaving(false);
    }
  }

  function startChat(teacher) {
    if (!user) {
      onNavigate('account', { next: 'find-teacher' });
      return;
    }
    onNavigate('messages', { userId: teacher.user_id, name: teacher.name, context: 'teacher' });
  }

  async function toggleReviews(teacher) {
    if (openReviews === teacher.id) {
      setOpenReviews('');
      return;
    }
    setOpenReviews(teacher.id);
    try {
      const data = await apiRequest(`/api/teachers/${teacher.id}/reviews`);
      setReviewsByTeacher((current) => ({ ...current, [teacher.id]: data.reviews || [] }));
      if (typeof data.reviewsEnabled === 'boolean') {
        setMarketplace((current) => ({ ...current, reviewsEnabled: data.reviewsEnabled }));
      }
      const mine = data.reviews?.find((review) => review.mine);
      if (mine) {
        setReviewDrafts((current) => ({
          ...current,
          [teacher.id]: { rating: mine.rating, comment: mine.comment },
        }));
      }
    } catch (error) {
      setStatus(error.message);
    }
  }

  function updateReviewDraft(teacherId, changes) {
    setReviewDrafts((current) => ({
      ...current,
      [teacherId]: { rating: 5, comment: '', ...current[teacherId], ...changes },
    }));
  }

  async function submitReview(event, teacher) {
    event.preventDefault();
    if (!user) {
      onNavigate('account', { next: 'find-teacher' });
      return;
    }
    const draft = { rating: 5, comment: '', ...reviewDrafts[teacher.id] };
    try {
      const data = await apiRequest(`/api/teachers/${teacher.id}/reviews`, {
        method: 'POST',
        body: JSON.stringify(draft),
      });
      setReviewsByTeacher((current) => {
        const reviews = current[teacher.id] || [];
        const next = reviews.some((review) => review.id === data.review.id)
          ? reviews.map((review) => review.id === data.review.id ? data.review : review)
          : [data.review, ...reviews];
        return { ...current, [teacher.id]: next };
      });
      setTeachers((current) => current.map((item) => item.id === teacher.id
        ? { ...item, reviewSummary: data.summary }
        : item));
      setStatus('Your teacher review is public.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  function openLessonTools(mode) {
    if (!user) {
      onNavigate('account', { next: 'find-teacher' });
      return;
    }
    setLessonTools((current) => current === mode ? '' : mode);
    window.setTimeout(() => document.getElementById('human-lesson-workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0);
  }

  async function createLesson(event) {
    event.preventDefault();
    setLessonSaving(true);
    setStatus('Creating private lesson…');
    try {
      const data = await apiRequest('/api/human-lessons', {
        method: 'POST',
        body: JSON.stringify({
          ...lessonDraft,
          durationMinutes: Number(lessonDraft.durationMinutes),
          scheduledFor: new Date(lessonDraft.scheduledFor).toISOString(),
        }),
      });
      setLessons((current) => [data.meeting, ...current]);
      setStatus('Lesson ready. Copy the credentials or send them through a private chat.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLessonSaving(false);
    }
  }

  function enterLesson(meetingId, accessCode = '', { autoJoin = false, isHost = false } = {}) {
    window.sessionStorage.setItem(SAVED_INVITE_KEY, JSON.stringify({ meetingId, accessCode, autoJoin, isHost }));
    onNavigate('lesson-room', { meetingId });
  }

  function joinLesson(event) {
    event.preventDefault();
    enterLesson(
      joinDraft.meetingId.trim().toUpperCase(),
      joinDraft.accessCode.trim().toUpperCase(),
      { autoJoin: true },
    );
  }

  async function sendLessonInvite(meeting) {
    const toUserId = inviteTargets[meeting.id];
    if (!toUserId) {
      setStatus('Choose a private-chat contact first.');
      return;
    }
    setLessonSaving(true);
    try {
      await apiRequest(`/api/human-lessons/${encodeURIComponent(meeting.meetingId)}/invitations`, {
        method: 'POST', body: JSON.stringify({ toUserId }),
      });
      const contact = contacts.find((item) => item.user_id === toUserId);
      setStatus(`Private invitation sent to ${contact?.name || 'your student'}.`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLessonSaving(false);
    }
  }

  async function createBreakout(meeting) {
    setLessonSaving(true);
    try {
      const data = await apiRequest(`/api/human-lessons/${encodeURIComponent(meeting.meetingId)}/breakouts`, {
        method: 'POST', body: JSON.stringify({ name: breakoutNames[meeting.id] || '' }),
      });
      setLessons((current) => current.map((item) => item.id === meeting.id ? data.meeting : item));
      setUser?.(data.user);
      setBreakoutNames((current) => ({ ...current, [meeting.id]: '' }));
      setStatus(`${data.breakout.roomName} created for ${data.chargedMcoins.toFixed(2)} Mcoins.`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLessonSaving(false);
    }
  }

  const lessonQuote = Number((
    (Number(lessonDraft.durationMinutes || 0) / lessonConfig.durationStepMinutes)
    * lessonConfig.callCostPerStepMcoins
  ).toFixed(2));

  return (
    <section className="page-shell teacher-marketplace-page">
      <header className="teacher-marketplace-heading">
        <div>
          <p className="eyebrow">Learn with a person</p>
          <h1>Find a teacher</h1>
          <p>Choose a teacher, chat privately, and learn at your pace.</p>
        </div>
        <div className="teacher-heading-actions">
          <button className="ghost" type="button" onClick={() => openLessonTools('join')}>Join lesson</button>
          {ownTeacher && <button className="primary" type="button" onClick={() => openLessonTools('host')}>Host lesson</button>}
          <button className={ownTeacher ? 'ghost' : 'primary'} type="button" disabled={!ownTeacher && !marketplace.applicationsEnabled} onClick={openTeacherEditor}>
            {ownTeacher ? 'Edit profile' : marketplace.applicationsEnabled ? 'Teach on Polymath' : 'Teacher signup paused'}
          </button>
        </div>
      </header>

      {lessonTools && (
        <section id="human-lesson-workspace" className="human-lesson-workspace">
          <header>
            <div>
              <p className="eyebrow">Private video lessons</p>
              <h2>{lessonTools === 'host' ? 'Host a lesson' : 'Join a lesson'}</h2>
              <p>{lessonTools === 'host' ? 'Create the room, then invite students privately.' : 'Use the private meeting ID and password your teacher sent.'}</p>
            </div>
            <button className="ghost" type="button" onClick={() => setLessonTools('')}>Close</button>
          </header>

          {lessonTools === 'join' && (
            <div className="lesson-join-layout">
              <form className="lesson-join-form" onSubmit={joinLesson}>
                <label className="field">Meeting ID<input value={joinDraft.meetingId} onChange={(event) => setJoinDraft({ ...joinDraft, meetingId: event.target.value.toUpperCase() })} placeholder="PM-ABCD-2345" autoCapitalize="characters" required /></label>
                <label className="field">Password<input value={joinDraft.accessCode} onChange={(event) => setJoinDraft({ ...joinDraft, accessCode: event.target.value.toUpperCase() })} placeholder="ABCD-2345" autoCapitalize="characters" required /></label>
                <button className="primary" type="submit">Enter private lesson</button>
              </form>
              <div className="invited-lesson-list">
                <strong>Your invitations</strong>
                {lessons.filter((meeting) => !meeting.isHost).map((meeting) => (
                  <article key={meeting.id}>
                    <span><b>{meeting.title}</b><small>{new Date(meeting.scheduledFor).toLocaleString()} · {meeting.durationMinutes} min</small></span>
                    <button className="ghost" type="button" onClick={() => onNavigate('messages', { userId: meeting.host?.user_id, name: meeting.host?.name || 'Teacher', context: 'teacher' })}>Open invite</button>
                  </article>
                ))}
                {!lessons.some((meeting) => !meeting.isHost) && <p className="muted">Meeting invitations appear in your private chats.</p>}
              </div>
            </div>
          )}

          {lessonTools === 'host' && ownTeacher && (
            <>
              <form className="lesson-host-form" onSubmit={createLesson}>
                <label className="field">Lesson name<input maxLength="100" value={lessonDraft.title} onChange={(event) => setLessonDraft({ ...lessonDraft, title: event.target.value })} required /></label>
                <label className="field">Starts<input type="datetime-local" value={lessonDraft.scheduledFor} onChange={(event) => setLessonDraft({ ...lessonDraft, scheduledFor: event.target.value })} required /></label>
                <label className="field">Minutes
                  <input type="number" min={lessonConfig.durationStepMinutes} max="720" step={lessonConfig.durationStepMinutes} value={lessonDraft.durationMinutes} onChange={(event) => setLessonDraft({ ...lessonDraft, durationMinutes: event.target.value })} required />
                  <small>Use 10-minute intervals.</small>
                </label>
                <div className="lesson-duration-presets" aria-label="Common lesson durations">
                  {[30, 60, 90].map((minutes) => <button key={minutes} className={Number(lessonDraft.durationMinutes) === minutes ? 'selected' : ''} type="button" onClick={() => setLessonDraft({ ...lessonDraft, durationMinutes: minutes })}>{minutes} min</button>)}
                </div>
                <div className="lesson-cost-quote"><span>Call fee when you start</span><strong>{lessonQuote.toFixed(2)} Mcoins</strong><small>{lessonConfig.callCostPerStepMcoins} Mcoin per {lessonConfig.durationStepMinutes} minutes</small></div>
                <button className="primary" type="submit" disabled={lessonSaving}>{lessonSaving ? 'Creating…' : 'Create private room'}</button>
              </form>

              <div className="hosted-lesson-list">
                {lessons.filter((meeting) => meeting.isHost).map((meeting) => (
                  <details className="hosted-lesson-card" key={meeting.id} open={meeting.status === 'active'}>
                    <summary><span><strong>{meeting.title}</strong><small>{new Date(meeting.scheduledFor).toLocaleString()} · {meeting.durationMinutes} min</small></span><b>{titleCase(meeting.status)}</b></summary>
                    <div className="lesson-credential-grid">
                      <span><small>Meeting ID</small><strong>{meeting.meetingId}</strong></span>
                      <span><small>Password</small><strong>{meeting.accessCode}</strong></span>
                    </div>
                    <div className="hosted-lesson-actions">
                      <button className="primary" type="button" disabled={['ended', 'cancelled'].includes(meeting.status)} onClick={() => enterLesson(meeting.meetingId, '', { autoJoin: true, isHost: true })}>Start / rejoin</button>
                      <select aria-label={`Student to invite to ${meeting.title}`} value={inviteTargets[meeting.id] || ''} onChange={(event) => setInviteTargets({ ...inviteTargets, [meeting.id]: event.target.value })}>
                        <option value="">Choose private-chat contact</option>
                        {contacts.map((contact) => <option key={contact.user_id} value={contact.user_id}>{contact.name || 'Polymath member'}</option>)}
                      </select>
                      <button className="ghost" type="button" disabled={lessonSaving || !contacts.length} onClick={() => sendLessonInvite(meeting)}>Send private invite</button>
                    </div>
                    {!contacts.length && <small className="lesson-contact-hint">Start a private chat with a student first, then they appear here.</small>}

                    <section className="breakout-manager">
                      <header><span><strong>Breakout rooms</strong><small>{lessonConfig.breakoutRoomCostMcoins} Mcoin each</small></span></header>
                      {(meeting.breakouts || []).map((breakout) => (
                        <article key={breakout.id}>
                          <span><b>{breakout.roomName}</b><small>{breakout.meetingId} · {breakout.accessCode}</small></span>
                          <select aria-label={`Student for ${breakout.roomName}`} value={inviteTargets[breakout.id] || ''} onChange={(event) => setInviteTargets({ ...inviteTargets, [breakout.id]: event.target.value })}>
                            <option value="">Choose student</option>
                            {contacts.map((contact) => <option key={contact.user_id} value={contact.user_id}>{contact.name || 'Polymath member'}</option>)}
                          </select>
                          <button className="ghost" type="button" disabled={lessonSaving || !contacts.length} onClick={() => sendLessonInvite(breakout)}>Invite</button>
                        </article>
                      ))}
                      <div className="breakout-create-row">
                        <input maxLength="60" aria-label="New breakout room name" placeholder={`Breakout ${(meeting.breakouts || []).length + 1}`} value={breakoutNames[meeting.id] || ''} onChange={(event) => setBreakoutNames({ ...breakoutNames, [meeting.id]: event.target.value })} />
                        <button className="ghost" type="button" disabled={lessonSaving || ['ended', 'cancelled'].includes(meeting.status)} onClick={() => createBreakout(meeting)}>Add · {lessonConfig.breakoutRoomCostMcoins} Mcoin</button>
                      </div>
                    </section>
                  </details>
                ))}
                {!lessons.some((meeting) => meeting.isHost) && <p className="muted">Your created lessons will appear here.</p>}
              </div>
            </>
          )}
        </section>
      )}

      {showTeacherForm && (
        <form id="teacher-profile-editor" className="teacher-profile-editor" onSubmit={saveTeacherProfile}>
          <header>
            <div><p className="eyebrow">Teacher profile</p><h2>{ownTeacher ? 'Update your profile' : 'Introduce yourself'}</h2></div>
            <button className="ghost" type="button" onClick={() => setShowTeacherForm(false)}>Close</button>
          </header>

          <div className="teacher-form-grid">
            <label className="field">Teaching headline
              <input maxLength="100" placeholder="Patient piano teacher for beginners" value={form.headline} onChange={(event) => setForm({ ...form, headline: event.target.value })} required />
            </label>
            <label className="field">Hourly rate in Mcoins
              <input type="number" min={marketplace.minimumHourlyRateMcoins} max={marketplace.maximumHourlyRateMcoins || 1000000000} step="0.01" value={form.hourlyRateMcoins} onChange={(event) => setForm({ ...form, hourlyRateMcoins: event.target.value })} />
              <small>{marketplace.maximumHourlyRateMcoins > 0
                ? `${marketplace.minimumHourlyRateMcoins.toLocaleString()}–${marketplace.maximumHourlyRateMcoins.toLocaleString()} Mcoins allowed`
                : `${marketplace.minimumHourlyRateMcoins.toLocaleString()} Mcoins minimum; no maximum`}</small>
            </label>
            <label className="field">Location
              <input maxLength="100" placeholder="Singapore or online only" value={form.location} onChange={(event) => setForm({ ...form, location: event.target.value })} />
            </label>
            <label className="field">Languages
              <input maxLength="200" placeholder="English, Mandarin" value={form.languages} onChange={(event) => setForm({ ...form, languages: event.target.value })} />
            </label>
          </div>

          <label className="field">About your teaching
            <textarea rows="4" maxLength="1200" placeholder="How do you help students learn?" value={form.bio} onChange={(event) => setForm({ ...form, bio: event.target.value })} required />
          </label>

          <fieldset className="teacher-choice-group">
            <legend>Instruments</legend>
            <div>
              {INSTRUMENTS.map((instrument) => (
                <button key={instrument.id} type="button" className={form.instruments.includes(instrument.id) ? 'selected' : ''} aria-pressed={form.instruments.includes(instrument.id)} onClick={() => setForm({ ...form, instruments: toggleChoice(form.instruments, instrument.id) })}>{instrument.shortLabel}</button>
              ))}
            </div>
          </fieldset>

          <div className="teacher-form-grid compact">
            <fieldset className="teacher-choice-group">
              <legend>Student levels</legend>
              <div>{LEVELS.map(([value, label]) => <button key={value} type="button" className={form.levels.includes(value) ? 'selected' : ''} aria-pressed={form.levels.includes(value)} onClick={() => setForm({ ...form, levels: toggleChoice(form.levels, value) })}>{label}</button>)}</div>
            </fieldset>
            <fieldset className="teacher-choice-group">
              <legend>Lesson type</legend>
              <div>{LESSON_MODES.map(([value, label]) => <button key={value} type="button" className={form.lessonModes.includes(value) ? 'selected' : ''} aria-pressed={form.lessonModes.includes(value)} onClick={() => setForm({ ...form, lessonModes: toggleChoice(form.lessonModes, value) })}>{label}</button>)}</div>
            </fieldset>
          </div>

          <label className="field">Availability
            <input maxLength="200" placeholder="Weeknights and Saturday mornings" value={form.availability} onChange={(event) => setForm({ ...form, availability: event.target.value })} />
          </label>
          <label className="teacher-publish-toggle"><input type="checkbox" checked={form.published} onChange={(event) => setForm({ ...form, published: event.target.checked })} /><span>Show my profile to students</span></label>
          <p className="teacher-editor-fee-note">Polymath platform fee: <strong>{percentLabel(marketplace.platformFeePercent)}%</strong> of future lesson payments processed through Polymath. You keep <strong>{percentLabel(marketplace.teacherKeepsPercent)}%</strong> before any cash-out fee.</p>
          <button className="primary" type="submit" disabled={saving}>{saving ? 'Saving...' : 'Save teacher profile'}</button>
        </form>
      )}

      <div className="teacher-filter-bar">
        <input aria-label="Search teachers" placeholder="Search teachers" value={filters.query} onChange={(event) => setFilters({ ...filters, query: event.target.value })} />
        <select aria-label="Filter by instrument" value={filters.instrument} onChange={(event) => setFilters({ ...filters, instrument: event.target.value })}>
          <option value="">All instruments</option>
          {INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}
        </select>
        <select aria-label="Filter by lesson type" value={filters.lessonMode} onChange={(event) => setFilters({ ...filters, lessonMode: event.target.value })}>
          <option value="">Online or in person</option>
          {LESSON_MODES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <select aria-label="Filter by student level" value={filters.level} onChange={(event) => setFilters({ ...filters, level: event.target.value })}>
          <option value="">All levels</option>
          {LEVELS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      {marketplace.directoryEnabled && <div className="teacher-grid">
        {filteredTeachers.map((teacher) => {
          const reviews = reviewsByTeacher[teacher.id] || [];
          const draft = { rating: 5, comment: '', ...reviewDrafts[teacher.id] };
          const reviewsOpen = openReviews === teacher.id;
          return (
            <article className={`teacher-card ${reviewsOpen ? 'reviews-open' : ''}`} key={teacher.id}>
              <header className="teacher-card-identity">
                <TeacherAvatar teacher={teacher} size="large" />
                <div>
                  <h2>{teacher.name}</h2>
                  <p>{teacher.headline}</p>
                  <button className="teacher-rating-summary" type="button" onClick={() => toggleReviews(teacher)}>
                    <TeacherStars value={teacher.reviewSummary.averageRating} label={`${teacher.name} average rating`} />
                    <span>{teacher.reviewSummary.reviewCount ? `${teacher.reviewSummary.averageRating} (${teacher.reviewSummary.reviewCount})` : 'New teacher'}</span>
                  </button>
                </div>
              </header>

              <ReputationScore ranking={teacher.ranking} audienceLabel="students" />

              <div className="teacher-tags">
                {teacher.instruments.map((instrument) => <span key={instrument}>{INSTRUMENT_BY_ID[instrument]?.shortLabel || titleCase(instrument)}</span>)}
                {teacher.lessonModes.map((mode) => <span key={mode}>{titleCase(mode)}</span>)}
              </div>
              <p className="teacher-bio">{teacher.bio}</p>
              <dl className="teacher-details">
                {teacher.levels?.length > 0 && <div><dt>Teaches</dt><dd>{teacher.levels.map(titleCase).join(', ')}</dd></div>}
                {teacher.languages?.length > 0 && <div><dt>Languages</dt><dd>{teacher.languages.map(titleCase).join(', ')}</dd></div>}
                {teacher.location && <div><dt>Location</dt><dd>{teacher.location}</dd></div>}
                {teacher.availability && <div><dt>Available</dt><dd>{teacher.availability}</dd></div>}
              </dl>

              <footer className="teacher-card-footer">
                <strong>{teacher.hourlyRateMcoins > 0 ? `${teacher.hourlyRateMcoins.toLocaleString()} Mcoins / hour` : 'Ask for rate'}</strong>
                <div>
                  {teacher.isSelf
                    ? <button className="primary" type="button" onClick={openTeacherEditor}>Edit profile</button>
                    : <button className="primary" type="button" onClick={() => startChat(teacher)}>Private chat</button>}
                  <button className="ghost" type="button" onClick={() => toggleReviews(teacher)}>Reviews</button>
                </div>
              </footer>

              {reviewsOpen && (
                <section className="teacher-reviews">
                  {!teacher.isSelf && teacher.canReview && marketplace.reviewsEnabled && (
                    <form className="teacher-review-form" onSubmit={(event) => submitReview(event, teacher)}>
                      <strong>Your review</strong>
                      <TeacherStars value={draft.rating} onChange={(rating) => updateReviewDraft(teacher.id, { rating })} label="Your rating" />
                      <textarea rows="3" maxLength="1000" placeholder="How was your experience?" value={draft.comment} onChange={(event) => updateReviewDraft(teacher.id, { comment: event.target.value })} required />
                      <button className="primary" type="submit">Post review</button>
                    </form>
                  )}
                  {!teacher.isSelf && !teacher.canReview && marketplace.reviewsEnabled && (
                    <button className="teacher-review-unlock" type="button" onClick={() => startChat(teacher)}>Private chat with this teacher before reviewing.</button>
                  )}
                  {!teacher.isSelf && !marketplace.reviewsEnabled && (
                    <p className="teacher-no-reviews">New reviews are temporarily paused. Existing reviews remain visible.</p>
                  )}
                  <div className="teacher-review-list">
                    {reviews.map((review) => (
                      <article key={review.id}>
                        <div><TeacherAvatar teacher={review.author} size="small" /><span><strong>{review.author.name}</strong><small>Connected student</small></span><TeacherStars value={review.rating} label={`${review.author.name} rating`} /></div>
                        <p>{review.comment}</p>
                        <time dateTime={review.updatedAt || review.createdAt}>{new Date(review.updatedAt || review.createdAt).toLocaleDateString()}</time>
                      </article>
                    ))}
                    {!reviews.length && <p className="teacher-no-reviews">No reviews yet.</p>}
                  </div>
                </section>
              )}
            </article>
          );
        })}
      </div>}

      {!marketplace.directoryEnabled && (
        <div className="teacher-empty-state">
          <h2>Teacher directory paused</h2>
          <p>The administrator has temporarily hidden public teacher listings. Existing profiles and reviews are preserved.</p>
        </div>
      )}
      {marketplace.directoryEnabled && !filteredTeachers.length && (
        <div className="teacher-empty-state">
          <h2>No teachers found yet.</h2>
          <p>Try another filter or become the first teacher in this category.</p>
          <button className="primary" type="button" onClick={openTeacherEditor}>Teach on Polymath</button>
        </div>
      )}
      <aside className="teacher-marketplace-fee-notice" aria-label="Find a Teacher fees and tax notice">
        <header><p className="eyebrow">Fees and tax notice</p><h2>Clear before you choose</h2></header>
        <div>
          <span><small>Polymath teacher fee</small><strong>{percentLabel(marketplace.platformFeePercent)}%</strong></span>
          <span><small>Teacher keeps</small><strong>{percentLabel(marketplace.teacherKeepsPercent)}%</strong></span>
          <span><small>Later cash-out fee</small><strong>{percentLabel(marketplace.withdrawalFeePercent)}%</strong></span>
        </div>
        <p>The Polymath teacher fee applies only to lesson payments processed through Polymath. It is a platform fee, not a government tax. Government taxes or payment-processor charges may be separate.</p>
        <p>Virtual-room infrastructure costs {lessonConfig.callCostPerStepMcoins} Mcoin per {lessonConfig.durationStepMinutes} minutes and each breakout room costs {lessonConfig.breakoutRoomCostMcoins} Mcoin. Teachers are charged only when they start the main room; student-to-teacher transfers are separate.</p>
        {!marketplace.checkoutAvailable && <p>Teachers arrange their lesson price directly with students. Polymath does not automatically deduct the teacher’s advertised hourly rate.</p>}
        {marketplace.notice && <p className="teacher-marketplace-custom-notice">{marketplace.notice}</p>}
      </aside>
      {status && <p className="form-status floating-status">{status}</p>}
    </section>
  );
}
