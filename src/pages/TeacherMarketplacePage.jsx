import { useEffect, useMemo, useState } from 'react';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import { apiRequest } from '../services/api.js';

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

export default function TeacherMarketplacePage({ user, onNavigate }) {
  const [teachers, setTeachers] = useState([]);
  const [ownTeacher, setOwnTeacher] = useState(null);
  const [filters, setFilters] = useState({ query: '', instrument: '', lessonMode: '', level: '' });
  const [showTeacherForm, setShowTeacherForm] = useState(false);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [openReviews, setOpenReviews] = useState('');
  const [reviewsByTeacher, setReviewsByTeacher] = useState({});
  const [reviewDrafts, setReviewDrafts] = useState({});
  const [status, setStatus] = useState('');
  const [saving, setSaving] = useState(false);

  async function loadTeachers() {
    try {
      const data = await apiRequest('/api/teachers');
      setTeachers(data.teachers || []);
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
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadTeachers(); }, [user?.user_id]);
  useEffect(() => { loadOwnTeacher(); }, [user?.user_id]);

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

  return (
    <section className="page-shell teacher-marketplace-page">
      <header className="teacher-marketplace-heading">
        <div>
          <p className="eyebrow">Learn with a person</p>
          <h1>Find a teacher</h1>
          <p>Choose a teacher, chat privately, and learn at your pace.</p>
        </div>
        <button className="primary" type="button" onClick={openTeacherEditor}>
          {ownTeacher ? 'Edit teacher profile' : 'Teach on Polymath'}
        </button>
      </header>

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
              <input type="number" min="0" max="100000" step="0.5" value={form.hourlyRateMcoins} onChange={(event) => setForm({ ...form, hourlyRateMcoins: event.target.value })} />
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

      <div className="teacher-grid">
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
                  {!teacher.isSelf && teacher.canReview && (
                    <form className="teacher-review-form" onSubmit={(event) => submitReview(event, teacher)}>
                      <strong>Your review</strong>
                      <TeacherStars value={draft.rating} onChange={(rating) => updateReviewDraft(teacher.id, { rating })} label="Your rating" />
                      <textarea rows="3" maxLength="1000" placeholder="How was your experience?" value={draft.comment} onChange={(event) => updateReviewDraft(teacher.id, { comment: event.target.value })} required />
                      <button className="primary" type="submit">Post review</button>
                    </form>
                  )}
                  {!teacher.isSelf && !teacher.canReview && (
                    <button className="teacher-review-unlock" type="button" onClick={() => startChat(teacher)}>Private chat with this teacher before reviewing.</button>
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
      </div>

      {!filteredTeachers.length && (
        <div className="teacher-empty-state">
          <h2>No teachers found yet.</h2>
          <p>Try another filter or become the first teacher in this category.</p>
          <button className="primary" type="button" onClick={openTeacherEditor}>Teach on Polymath</button>
        </div>
      )}
      {status && <p className="form-status floating-status">{status}</p>}
    </section>
  );
}
