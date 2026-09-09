import { useEffect, useMemo, useRef, useState } from 'react';
import {
  applySongBlueprint,
  buildSongArrangement,
  CREATE_MUSIC_GENRES,
  CREATE_MUSIC_INSTRUMENTS,
  CREATE_MUSIC_MOODS,
  createBlankMusicProject,
  lyricsAsText,
  lyricsFromText,
} from '../engine/songCreationEngine.js';
import { pianoAudio } from '../engine/audioEngine.js';
import { ensembleAudio } from '../engine/ensembleEngine.js';
import { guitarAudio } from '../engine/guitarEngine.js';
import { apiRequest } from '../services/api.js';
import { downloadSongJson, downloadSongMidi } from '../utils/exporters.js';
import '../createMusic.css';

const STEPS = [
  ['idea', 'Idea'],
  ['sound', 'Sound'],
  ['lyrics', 'Lyrics'],
  ['arrange', 'Arrangement'],
  ['rehearse', 'Rehearse'],
];
const DRAFT_STORAGE_KEY = 'polymath-create-music-draft-v1';
const ACTIVE_JOB_KEY = 'polymath-create-music-job-v1';

function activeJobStorageKey(userId) {
  return `${ACTIVE_JOB_KEY}:${String(userId || 'guest')}`;
}

function validActiveJobId(value) {
  return /^resp_[A-Za-z0-9_-]{8,210}$/.test(String(value || '').trim());
}

function restoreActiveJobId(userId) {
  const storageKey = activeJobStorageKey(userId);
  const stored = window.localStorage.getItem(storageKey) || '';
  if (validActiveJobId(stored)) return stored;
  if (stored) window.localStorage.removeItem(storageKey);
  return '';
}

function loadDraft() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(DRAFT_STORAGE_KEY) || 'null');
    return saved?.brief && saved?.lyrics ? saved : createBlankMusicProject();
  } catch {
    return createBlankMusicProject();
  }
}

function safeFilename(value, extension) {
  const base = String(value || 'polymath-song')
    .replace(/[^a-z0-9-_]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 70) || 'polymath-song';
  return `${base}.${extension}`;
}

function downloadText(text, filename, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${String(Math.floor(safe % 60)).padStart(2, '0')}`;
}

function firstEventAtOrAfter(events, time) {
  let low = 0;
  let high = events.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (Number(events[middle]?.time || 0) < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function playArrangementEvent(event, track, delay) {
  const duration = Math.max(0.06, Number(event.duration || 0.35));
  if (track.instrument === 'piano') {
    const at = pianoAudio.getCurrentTime() + Math.max(0, delay);
    pianoAudio.playAt(event.note, event.velocity || 0.7, duration, at, {
      source: 'create-music',
      scoreRole: event.scoreRole || track.role,
    });
    return;
  }
  if (track.instrument === 'acoustic-guitar') {
    const at = guitarAudio.getCurrentTime() + Math.max(0, delay);
    guitarAudio.playEvent(event, at);
    return;
  }
  const at = ensembleAudio.getCurrentTime() + Math.max(0, delay);
  ensembleAudio.playAt(event.note, track.instrument, event.velocity || 0.7, duration, at);
}

function ArrangementPlayer({
  arrangement,
  guideVoice,
  setGuideVoice,
  onUseInPiano,
  canExport,
  canGuideVoice,
  onUnlock,
}) {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [enabledTracks, setEnabledTracks] = useState(() => new Set(arrangement.tracks.map((track) => track.id)));
  const playbackRef = useRef({ startedAt: 0, offset: 0, next: new Map(), spokenCue: -1 });
  const intervalRef = useRef(null);
  const frameRef = useRef(null);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setEnabledTracks(new Set(arrangement.tracks.map((track) => track.id)));
    pianoAudio.stopAll({ releaseSeconds: 0.03 });
    ensembleAudio.stopAll();
    guitarAudio.stopAll(0.03);
  }, [arrangement]);

  useEffect(() => () => {
    if (intervalRef.current) window.clearInterval(intervalRef.current);
    if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    window.speechSynthesis?.cancel();
    pianoAudio.stopAll({ releaseSeconds: 0.03 });
    ensembleAudio.stopAll();
    guitarAudio.stopAll(0.03);
  }, []);

  function stopSchedulers() {
    if (intervalRef.current) window.clearInterval(intervalRef.current);
    if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    intervalRef.current = null;
    frameRef.current = null;
  }

  function pause() {
    const elapsed = (performance.now() - playbackRef.current.startedAt) / 1000;
    playbackRef.current.offset = Math.min(arrangement.duration, playbackRef.current.offset + elapsed);
    setCurrentTime(playbackRef.current.offset);
    setPlaying(false);
    stopSchedulers();
    window.speechSynthesis?.cancel();
    pianoAudio.stopAll({ releaseSeconds: 0.035 });
    ensembleAudio.stopAll();
    guitarAudio.stopAll(0.035);
  }

  async function start() {
    if (playing) return pause();
    if (currentTime >= arrangement.duration - 0.05) playbackRef.current.offset = 0;
    else playbackRef.current.offset = currentTime;
    await pianoAudio.activate();
    ensembleAudio.ensure();
    if (arrangement.tracks.some((track) => track.instrument === 'acoustic-guitar')) {
      await guitarAudio.prepareForPlayback();
      if (guitarAudio.getDiagnostics().acousticSamples === 'unavailable') guitarAudio.setToneMode('clean');
    }
    playbackRef.current.startedAt = performance.now();
    playbackRef.current.spokenCue = firstEventAtOrAfter(arrangement.lyricCues, playbackRef.current.offset) - 1;
    playbackRef.current.next = new Map(arrangement.tracks.map((track) => [
      track.id,
      firstEventAtOrAfter(track.events, playbackRef.current.offset),
    ]));
    setPlaying(true);

    const schedule = () => {
      const songTime = playbackRef.current.offset + (performance.now() - playbackRef.current.startedAt) / 1000;
      const horizon = songTime + 0.2;
      arrangement.tracks.forEach((track) => {
        if (!enabledTracks.has(track.id)) return;
        let index = playbackRef.current.next.get(track.id) || 0;
        while (index < track.events.length && track.events[index].time <= horizon) {
          const event = track.events[index];
          if (event.time >= songTime - 0.04) playArrangementEvent(event, track, event.time - songTime);
          index += 1;
        }
        playbackRef.current.next.set(track.id, index);
      });
      if (guideVoice && window.speechSynthesis) {
        const nextCueIndex = playbackRef.current.spokenCue + 1;
        const cue = arrangement.lyricCues[nextCueIndex];
        if (cue && cue.start <= horizon && cue.start >= songTime - 0.08) {
          const utterance = new window.SpeechSynthesisUtterance(cue.text);
          utterance.rate = Math.max(0.65, Math.min(1.35, arrangement.bpm / 105));
          utterance.pitch = 1.08;
          utterance.volume = 0.38;
          window.speechSynthesis.speak(utterance);
          playbackRef.current.spokenCue = nextCueIndex;
        }
      }
    };

    const draw = () => {
      const songTime = playbackRef.current.offset + (performance.now() - playbackRef.current.startedAt) / 1000;
      if (songTime >= arrangement.duration) {
        stopSchedulers();
        setCurrentTime(arrangement.duration);
        setPlaying(false);
        playbackRef.current.offset = arrangement.duration;
        return;
      }
      setCurrentTime(songTime);
      frameRef.current = window.requestAnimationFrame(draw);
    };
    schedule();
    intervalRef.current = window.setInterval(schedule, 30);
    frameRef.current = window.requestAnimationFrame(draw);
  }

  function seek(value) {
    const next = Math.max(0, Math.min(arrangement.duration, Number(value) || 0));
    if (playing) pause();
    playbackRef.current.offset = next;
    setCurrentTime(next);
  }

  function toggleTrack(trackId) {
    setEnabledTracks((current) => {
      const next = new Set(current);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
    if (playing) pause();
  }

  const activeCue = arrangement.lyricCues.find((cue) => currentTime >= cue.start && currentTime < cue.end);
  const nextCue = arrangement.lyricCues.find((cue) => cue.start > currentTime);
  const activeSection = arrangement.sections.find((section) => currentTime >= section.start && currentTime < section.end);

  return (
    <section className='creation-player' aria-label='Karaoke arrangement player'>
      <header className='creation-player-heading'>
        <div>
          <span>{activeSection?.label || 'Ready'}</span>
          <h2>{arrangement.title}</h2>
        </div>
        <div className='creation-time'><strong>{formatTime(currentTime)}</strong><span>/ {formatTime(arrangement.duration)}</span></div>
      </header>

      <div className='karaoke-stage' aria-live='polite'>
        <small>{activeCue ? activeCue.section.replace(/-/g, ' ') : 'Count in when ready'}</small>
        <p className={activeCue ? 'active' : ''}>{activeCue?.text || 'Your lyrics will follow the arrangement here.'}</p>
        <span>{nextCue?.text || 'End of song'}</span>
      </div>

      <input
        className='creation-seek'
        type='range'
        min='0'
        max={arrangement.duration}
        step='0.02'
        value={currentTime}
        onChange={(event) => seek(event.target.value)}
        aria-label='Song position'
      />

      <div className='creation-transport'>
        <button type='button' className='primary' onClick={start}>{playing ? 'Pause' : 'Play karaoke'}</button>
        <button type='button' className='ghost' onClick={() => seek(0)}>Restart</button>
        <button
          type='button'
          className={guideVoice ? 'guide-toggle active' : 'guide-toggle'}
          aria-pressed={guideVoice}
          onClick={() => canGuideVoice ? setGuideVoice(!guideVoice) : onUnlock()}
        >
          {!canGuideVoice ? 'Unlock guide voice' : guideVoice ? 'Timing voice on' : 'Timing voice off'}
        </button>
      </div>

      <div className='creation-mixer' aria-label='Arrangement tracks'>
        {arrangement.tracks.map((track) => (
          <button
            type='button'
            key={track.id}
            className={enabledTracks.has(track.id) ? 'active' : ''}
            onClick={() => toggleTrack(track.id)}
            style={{ '--track-colour': track.colour }}
          >
            <i aria-hidden='true' />
            <span>{track.label}</span>
            <small>{enabledTracks.has(track.id) ? 'On' : 'Muted'}</small>
          </button>
        ))}
      </div>

      <div className='creation-export-row'>
        <button type='button' className='primary' onClick={onUseInPiano}>Open piano version</button>
        <button type='button' className='ghost' disabled={!canExport} onClick={() => downloadSongMidi(arrangement)}>Download MIDI</button>
        <button type='button' className='ghost' disabled={!canExport} onClick={() => downloadSongJson(arrangement)}>Download project JSON</button>
        {!canExport && <small>Exports unlock with a Create Music subscription.</small>}
      </div>
    </section>
  );
}

export default function CreateMusicPage({ user, onNavigate }) {
  const [project, setProject] = useState(loadDraft);
  const [step, setStep] = useState(0);
  const [lyricsText, setLyricsText] = useState(() => lyricsAsText(loadDraft()));
  const [capabilities, setCapabilities] = useState(null);
  const [activeJobId, setActiveJobId] = useState('');
  const [aiStatus, setAiStatus] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [guideVoice, setGuideVoice] = useState(false);
  const [cloudProjects, setCloudProjects] = useState([]);
  const [cloudProjectsOpen, setCloudProjectsOpen] = useState(false);
  const arrangement = useMemo(() => buildSongArrangement(project), [project]);
  const hasCreatorAccess = Boolean(user?.admin || user?.access?.createMusic);
  const hasAiAccess = Boolean(user?.admin || user?.access?.createMusicAi);
  const hasArrangementAccess = Boolean(user?.admin || user?.access?.createMusicArrangements);
  const hasGuideVoiceAccess = Boolean(user?.admin || user?.access?.createMusicGuideVoice);
  const canExport = Boolean(user?.admin || user?.access?.createMusicExports);

  useEffect(() => {
    window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(project));
  }, [project]);

  useEffect(() => {
    apiRequest('/api/music-creation/capabilities')
      .then(setCapabilities)
      .catch(() => setCapabilities({ configured: false, provider: 'Local song architect' }));
  }, []);

  useEffect(() => {
    if (!hasCreatorAccess) {
      setCloudProjects([]);
      return;
    }
    apiRequest('/api/music-creation/projects')
      .then((data) => setCloudProjects(Array.isArray(data.projects) ? data.projects : []))
      .catch(() => setCloudProjects([]));
  }, [hasCreatorAccess, user?.user_id]);

  useEffect(() => {
    setActiveJobId(user?.user_id ? restoreActiveJobId(user.user_id) : '');
  }, [user?.user_id]);

  useEffect(() => {
    if (!activeJobId || !user) return undefined;
    let cancelled = false;
    let timeout;
    let consecutiveFailures = 0;
    const poll = async () => {
      try {
        const data = await apiRequest(`/api/music-creation/jobs/${encodeURIComponent(activeJobId)}`);
        if (cancelled) return;
        consecutiveFailures = 0;
        setAiStatus(data.status === 'IN_QUEUE' ? 'Your song draft is queued…' : data.status === 'IN_PROGRESS' ? 'Writing and arranging your draft…' : '');
        if (data.status === 'COMPLETED' && data.blueprint) {
          setProject((current) => {
            const next = applySongBlueprint(current, data.blueprint);
            setLyricsText(lyricsAsText(next));
            return next;
          });
          setStep(2);
          setActiveJobId('');
          window.localStorage.removeItem(activeJobStorageKey(user.user_id));
          setAiStatus('Draft ready. Every lyric remains editable.');
          return;
        }
        if (data.finished) {
          setError(data.error || 'The AI draft did not finish correctly. Your manual draft is still safe.');
          setActiveJobId('');
          window.localStorage.removeItem(activeJobStorageKey(user.user_id));
          return;
        }
        timeout = window.setTimeout(poll, 1800);
      } catch (requestError) {
        if (!cancelled) {
          setError(requestError.message);
          consecutiveFailures += 1;
          if ([401, 403, 404].includes(requestError.status) || consecutiveFailures >= 6) {
            setActiveJobId('');
            window.localStorage.removeItem(activeJobStorageKey(user.user_id));
            if (![401, 403, 404].includes(requestError.status)) {
              setError('That draft could not be recovered. Your song idea is safe—press Draft lyrics with AI to retry.');
            }
          } else {
            timeout = window.setTimeout(poll, 3000);
          }
        }
      }
    };
    poll();
    return () => { cancelled = true; window.clearTimeout(timeout); };
  }, [activeJobId, user?.user_id]);

  function updateBrief(changes) {
    setProject((current) => ({
      ...current,
      brief: { ...current.brief, ...changes },
      updatedAt: new Date().toISOString(),
    }));
  }

  function toggleInstrument(instrumentId) {
    const current = new Set(project.brief.instruments || []);
    if (current.has(instrumentId)) current.delete(instrumentId);
    else if (current.size < 6) current.add(instrumentId);
    updateBrief({ instruments: [...current] });
  }

  function openCreatorPlans() {
    onNavigate('payment', { category: 'create-music' });
  }

  function goToStep(nextStep) {
    if (nextStep >= 3 && !hasArrangementAccess) {
      openCreatorPlans();
      return;
    }
    setStep(nextStep);
  }

  async function askAi(kind = 'draft') {
    if (!user) {
      onNavigate('account', { next: 'create-music' });
      return;
    }
    if (!hasAiAccess) {
      openCreatorPlans();
      return;
    }
    setError('');
    setAiStatus('Starting your private music assistant…');
    try {
      const data = await apiRequest('/api/music-creation/jobs', {
        method: 'POST',
        body: JSON.stringify({
          kind,
          brief: {
            ...project.brief,
            title: project.title,
            existingLyrics: lyricsText,
            revisionRequest: kind === 'revise' ? project.revisionRequest || '' : '',
          },
        }),
      });
      setActiveJobId(data.id);
      window.localStorage.setItem(activeJobStorageKey(user.user_id), data.id);
    } catch (requestError) {
      setError(requestError.message);
      setAiStatus('');
    }
  }

  async function cancelAi() {
    if (!activeJobId) return;
    try { await apiRequest(`/api/music-creation/jobs/${encodeURIComponent(activeJobId)}/cancel`, { method: 'POST' }); } catch { /* best effort */ }
    setActiveJobId('');
    window.localStorage.removeItem(activeJobStorageKey(user?.user_id));
    setAiStatus('Draft stopped. Your work is still here.');
  }

  function applyEditedLyrics() {
    setProject((current) => ({ ...current, lyrics: lyricsFromText(lyricsText), updatedAt: new Date().toISOString() }));
    setAiStatus('Lyrics updated. The melody has been rebuilt around your words.');
  }

  async function saveProject() {
    if (!user) {
      onNavigate('account', { next: 'create-music' });
      return;
    }
    if (!hasCreatorAccess) {
      openCreatorPlans();
      return;
    }
    setSaving(true);
    setError('');
    try {
      const data = await apiRequest(project.id
        ? `/api/music-creation/projects/${encodeURIComponent(project.id)}`
        : '/api/music-creation/projects', {
        method: project.id ? 'PUT' : 'POST',
        body: JSON.stringify({ project, arrangement }),
      });
      setProject(data.project);
      const list = await apiRequest('/api/music-creation/projects');
      setCloudProjects(Array.isArray(list.projects) ? list.projects : []);
      setAiStatus('Project saved to your account.');
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  }

  async function openCloudProject(projectId) {
    setSaving(true);
    setError('');
    try {
      const data = await apiRequest(`/api/music-creation/projects/${encodeURIComponent(projectId)}`);
      setProject(data.project);
      setLyricsText(lyricsAsText(data.project));
      setStep(0);
      setCloudProjectsOpen(false);
      setAiStatus('Cloud project opened.');
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  }

  function useInPiano() {
    window.sessionStorage.setItem('polymath-created-arrangement-v1', JSON.stringify(arrangement));
    onNavigate('studio', { created: '1' });
  }

  function resetProject() {
    if (!window.confirm('Start a fresh song? Download anything you want to keep first.')) return;
    const next = createBlankMusicProject();
    setProject(next);
    setLyricsText(lyricsAsText(next));
    setStep(0);
    setAiStatus('Fresh song ready.');
  }

  return (
    <section className='page-shell create-music-page'>
      <header className='create-music-hero'>
        <div>
          <p className='eyebrow'>Human-led songwriting</p>
          <h1>Create music you can actually perform.</h1>
          <p>Shape the idea, edit every word, build the backing arrangement, then rehearse it karaoke-style. AI assists; you remain the artist.</p>
        </div>
        <div className='create-music-hero-actions'>
          <span className={capabilities?.configured ? 'creation-ai-state ready' : 'creation-ai-state'}>
            {capabilities?.configured ? 'AI architect connected' : 'Manual studio ready'}
          </span>
          <button type='button' className='ghost' onClick={resetProject}>New song</button>
        </div>
      </header>

      <nav className='creation-steps' aria-label='Create music steps'>
        {STEPS.map(([id, label], index) => (
          <button
            type='button'
            key={id}
            className={step === index ? 'active' : index < step ? 'complete' : ''}
            onClick={() => goToStep(index)}
            aria-current={step === index ? 'step' : undefined}
          >
            <span>{index + 1}</span><strong>{label}</strong>
          </button>
        ))}
      </nav>

      {step === 0 && (
        <section className='creation-focus-card'>
          <div className='creation-question'>
            <span>1 of 5</span>
            <h2>What do you want this song to say?</h2>
            <p>Describe the story, message, person, or moment. Rough thoughts are enough.</p>
          </div>
          <label className='creation-large-field'>Song idea
            <textarea
              rows='7'
              maxLength='1800'
              placeholder='Example: I kept pretending I was fine after we moved apart, but one rainy train ride made me finally admit I miss you…'
              value={project.brief.idea}
              onChange={(event) => updateBrief({ idea: event.target.value })}
            />
            <small>{project.brief.idea.length}/1800</small>
          </label>
          <label className='field'>Working title
            <input value={project.title} placeholder='You can leave this blank' onChange={(event) => setProject({ ...project, title: event.target.value })} />
          </label>
          <div className='creation-reference-note'>
            <strong>Using a reference?</strong>
            <span>We borrow broad traits such as tempo, groove and energy—not another artist’s melody, lyrics or voice.</span>
          </div>
          <label className='field'>Optional reference
            <input
              value={project.brief.reference}
              placeholder='Song or artist reference'
              onChange={(event) => updateBrief({ reference: event.target.value })}
            />
          </label>
          <button type='button' className='primary creation-next' disabled={!project.brief.idea.trim()} onClick={() => setStep(1)}>Choose the sound</button>
        </section>
      )}

      {step === 1 && (
        <section className='creation-focus-card'>
          <div className='creation-question'>
            <span>2 of 5</span>
            <h2>How should it feel?</h2>
            <p>These decisions guide the arrangement and the way you will sing it.</p>
          </div>
          <div className='creation-choice-grid two'>
            <label className='field'>Genre<select value={project.brief.genre} onChange={(event) => updateBrief({ genre: event.target.value })}>{CREATE_MUSIC_GENRES.map((genre) => <option key={genre}>{genre}</option>)}</select></label>
            <label className='field'>Mood<select value={project.brief.mood} onChange={(event) => updateBrief({ mood: event.target.value })}>{CREATE_MUSIC_MOODS.map((mood) => <option key={mood}>{mood}</option>)}</select></label>
            <label className='field'>Tempo<input type='number' min='40' max='220' value={project.brief.bpm} onChange={(event) => updateBrief({ bpm: Number(event.target.value) })} /><small>{project.brief.bpm} BPM</small></label>
            <label className='field'>Key<select value={project.brief.key} onChange={(event) => updateBrief({ key: event.target.value })}>{['C major', 'D major', 'E major', 'F major', 'G major', 'A major', 'Bb major', 'C minor', 'D minor', 'E minor', 'F minor', 'G minor', 'A minor'].map((key) => <option key={key}>{key}</option>)}</select></label>
          </div>
          <div className='creation-energy' role='group' aria-label='Song energy'>
            {['low', 'medium', 'high'].map((energy) => <button type='button' key={energy} className={project.brief.energy === energy ? 'active' : ''} onClick={() => updateBrief({ energy })}>{energy}</button>)}
          </div>
          <fieldset className='creation-instruments'>
            <legend>Backing instruments</legend>
            <div>
              {CREATE_MUSIC_INSTRUMENTS.map((instrument) => (
                <button
                  type='button'
                  key={instrument.id}
                  className={project.brief.instruments.includes(instrument.id) ? 'active' : ''}
                  aria-pressed={project.brief.instruments.includes(instrument.id)}
                  onClick={() => toggleInstrument(instrument.id)}
                >{instrument.label}</button>
              ))}
            </div>
          </fieldset>
          <div className='creation-button-row'>
            <button type='button' className='ghost' onClick={() => setStep(0)}>Back</button>
            <button type='button' className='primary' disabled={Boolean(activeJobId)} onClick={() => askAi('draft')}>
              {hasAiAccess ? 'Draft lyrics with AI' : 'Unlock AI songwriter'}
            </button>
            <button type='button' className='ghost' onClick={() => setStep(2)}>Write lyrics myself</button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className='creation-focus-card creation-lyrics-card'>
          <div className='creation-question'>
            <span>3 of 5</span>
            <h2>Make every line yours.</h2>
            <p>Edit freely. Keep headings inside square brackets so Polymath understands each section.</p>
          </div>
          <textarea className='lyrics-editor' rows='24' value={lyricsText} onChange={(event) => setLyricsText(event.target.value)} spellCheck='true' />
          <label className='field'>What should AI change? <input value={project.revisionRequest || ''} placeholder='Example: make the chorus simpler and more hopeful' onChange={(event) => setProject({ ...project, revisionRequest: event.target.value })} /></label>
          <div className='creation-button-row'>
            <button type='button' className='ghost' onClick={() => setStep(1)}>Back</button>
            <button type='button' className='ghost' disabled={Boolean(activeJobId)} onClick={() => askAi('revise')}>Ask AI to revise</button>
            <button type='button' className='primary' onClick={() => {
              if (!hasArrangementAccess) {
                openCreatorPlans();
                return;
              }
              applyEditedLyrics();
              setStep(3);
            }}>{hasArrangementAccess ? 'Build the arrangement' : 'Unlock arrangements'}</button>
          </div>
        </section>
      )}

      {step === 3 && (
        <section className='creation-focus-card'>
          <div className='creation-question'>
            <span>4 of 5</span>
            <h2>Your playable arrangement is ready.</h2>
            <p>Polymath matched your lyric rhythm to an original melody, chords, bass and drums. Review the musical map before rehearsing.</p>
          </div>
          <div className='arrangement-summary-grid'>
            <article><small>Tempo</small><strong>{arrangement.bpm} BPM</strong></article>
            <article><small>Key</small><strong>{arrangement.key}</strong></article>
            <article><small>Length</small><strong>{formatTime(arrangement.duration)}</strong></article>
            <article><small>Tracks</small><strong>{arrangement.tracks.length}</strong></article>
          </div>
          <div className='arrangement-map'>
            {arrangement.sections.map((section) => (
              <div key={section.id} style={{ flexGrow: section.end - section.start }}>
                <strong>{section.label}</strong><small>{section.bars} bars</small>
              </div>
            ))}
          </div>
          <div className='chord-preview'><span>Chord map</span>{arrangement.chords.slice(0, 12).map((chord, index) => <b key={`${chord.time}-${index}`}>{chord.degree}</b>)}</div>
          <div className='creation-button-row'>
            <button type='button' className='ghost' onClick={() => setStep(2)}>Edit lyrics</button>
            <button type='button' className='primary' onClick={() => setStep(4)}>Rehearse the song</button>
          </div>
        </section>
      )}

      {step === 4 && (
        <section className='creation-rehearsal'>
          <ArrangementPlayer
            arrangement={arrangement}
            guideVoice={guideVoice}
            setGuideVoice={setGuideVoice}
            onUseInPiano={useInPiano}
            canExport={canExport}
            canGuideVoice={hasGuideVoiceAccess}
            onUnlock={openCreatorPlans}
          />
          <aside className='vocal-coach-card'>
            <p className='eyebrow'>Your singing plan</p>
            <h2>Learn it in four passes.</h2>
            <ol>
              {(project.vocalCoach?.practiceSteps || []).slice(0, 6).map((instruction) => <li key={instruction}>{instruction}</li>)}
            </ol>
            <div><strong>Comfortable range</strong><span>{project.vocalCoach?.comfortableRange || project.brief.vocalRange}</span></div>
          </aside>
        </section>
      )}

      {hasCreatorAccess && (
        <section className='creation-cloud-projects'>
          <button
            type='button'
            className='creation-cloud-toggle'
            aria-expanded={cloudProjectsOpen}
            onClick={() => setCloudProjectsOpen((open) => !open)}
          >
            <span><strong>My cloud projects</strong><small>{cloudProjects.length ? `${cloudProjects.length} saved` : 'No saved songs yet'}</small></span>
            <b>{cloudProjectsOpen ? '−' : '+'}</b>
          </button>
          {cloudProjectsOpen && (
            <div className='creation-cloud-list'>
              {cloudProjects.map((savedProject) => (
                <button type='button' key={savedProject.id} disabled={saving} onClick={() => openCloudProject(savedProject.id)}>
                  <span><strong>{savedProject.title}</strong><small>{[savedProject.genre, savedProject.key, savedProject.bpm ? `${savedProject.bpm} BPM` : ''].filter(Boolean).join(' · ')}</small></span>
                  <time dateTime={savedProject.updatedAt}>{new Date(savedProject.updatedAt).toLocaleDateString()}</time>
                </button>
              ))}
              {!cloudProjects.length && <p>Save this song to make it available on your other devices.</p>}
            </div>
          )}
        </section>
      )}

      <footer className='creation-project-footer'>
        <div>
          <strong>{project.title || 'Untitled song'}</strong>
          <span>Saved automatically on this device</span>
        </div>
        <button type='button' className='ghost' disabled={saving} onClick={saveProject}>{saving ? 'Saving…' : hasCreatorAccess ? 'Save to account' : 'Unlock cloud projects'}</button>
        <button type='button' className='ghost' onClick={() => downloadText(lyricsText, safeFilename(project.title, 'txt'))}>Download lyrics</button>
      </footer>

      {activeJobId && <div className='creation-job-status' role='status'><span className='creation-pulse' />{aiStatus || 'Creating your draft…'}<button type='button' onClick={cancelAi}>Stop</button></div>}
      {!activeJobId && aiStatus && <p className='form-status creation-status'>{aiStatus}</p>}
      {error && <p className='creation-error' role='alert'>{error}</p>}
    </section>
  );
}
