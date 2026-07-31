import { useEffect, useMemo, useRef, useState } from 'react';
import InstrumentTeacherSurface from '../components/InstrumentTeacherSurface.jsx';
import InstrumentIcon from '../components/InstrumentIcon.jsx';
import MusicUploadPanel from '../components/MusicUploadPanel.jsx';
import LearnModePanel from '../components/LearnModePanel.jsx';
import { ENSEMBLE_INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';
import { ensembleAudio } from '../engine/ensembleEngine.js';
import { getSongDuration, normalizeSong } from '../engine/scheduler.js';
import { parseUploadedSongFile } from '../utils/songParser.js';
import { apiRequest, fetchProtectedFile } from '../services/api.js';
import { analyzeLearningSections } from '../utils/learningSections.js';

const LOOKAHEAD_SECONDS = 0.16;
const SCHEDULER_INTERVAL_MS = 25;

const MELODIC_DEMO_NOTES = [
  { note: 'C4', time: 0, duration: 0.42, velocity: 0.78 },
  { note: 'E4', time: 0.5, duration: 0.42, velocity: 0.8 },
  { note: 'G4', time: 1, duration: 0.55, velocity: 0.84 },
  { note: 'A4', time: 1.65, duration: 0.42, velocity: 0.82 },
  { note: 'G4', time: 2.15, duration: 0.42, velocity: 0.78 },
  { note: 'E4', time: 2.65, duration: 0.42, velocity: 0.76 },
  { note: 'D4', time: 3.15, duration: 0.42, velocity: 0.78 },
  { note: 'C4', time: 3.65, duration: 0.9, velocity: 0.86 },
];

const DEMO_SONGS = {
  fiddle: {
    title: 'Fiddle Position Trainer', composer: 'Polymath Musician', bpm: 96,
    notes: MELODIC_DEMO_NOTES,
  },
  banjo: {
    title: 'Banjo Roll Trainer', composer: 'Polymath Musician', bpm: 112,
    notes: [
      { note: 'G3', time: 0, duration: 0.28, velocity: 0.82 },
      { note: 'B3', time: 0.28, duration: 0.28, velocity: 0.78 },
      { note: 'D4', time: 0.56, duration: 0.28, velocity: 0.84 },
      { note: 'G4', time: 0.84, duration: 0.28, velocity: 0.8 },
      { note: 'D4', time: 1.12, duration: 0.28, velocity: 0.78 },
      { note: 'B3', time: 1.4, duration: 0.28, velocity: 0.76 },
      { note: 'G3', time: 1.68, duration: 0.62, velocity: 0.88 },
      { note: 'D4', time: 2.4, duration: 0.28, velocity: 0.82 },
      { note: 'G4', time: 2.68, duration: 0.28, velocity: 0.86 },
      { note: 'B4', time: 2.96, duration: 0.7, velocity: 0.88 },
    ],
  },
  mandolin: {
    title: 'Mandolin Course Trainer', composer: 'Polymath Musician', bpm: 104,
    notes: MELODIC_DEMO_NOTES.map((note) => ({ ...note, note: note.note === 'C4' ? 'D4' : note.note })),
  },
  dobro: {
    title: 'Dobro Slide Position Trainer', composer: 'Polymath Musician', bpm: 82,
    notes: [
      { note: 'G3', time: 0, duration: 0.7, velocity: 0.78 },
      { note: 'B3', time: 0.8, duration: 0.7, velocity: 0.8 },
      { note: 'D4', time: 1.6, duration: 0.9, velocity: 0.84 },
      { note: 'E4', time: 2.65, duration: 0.75, velocity: 0.8 },
      { note: 'D4', time: 3.55, duration: 0.75, velocity: 0.78 },
      { note: 'B3', time: 4.45, duration: 0.75, velocity: 0.78 },
      { note: 'G3', time: 5.35, duration: 1.1, velocity: 0.86 },
    ],
  },
  'upright-bass': {
    title: 'Upright Bass Walking Trainer', composer: 'Polymath Musician', bpm: 92,
    notes: [
      { note: 'C2', time: 0, duration: 0.7, velocity: 0.88 },
      { note: 'E2', time: 0.75, duration: 0.7, velocity: 0.82 },
      { note: 'G2', time: 1.5, duration: 0.7, velocity: 0.86 },
      { note: 'A2', time: 2.25, duration: 0.7, velocity: 0.82 },
      { note: 'G2', time: 3, duration: 0.7, velocity: 0.84 },
      { note: 'E2', time: 3.75, duration: 0.7, velocity: 0.8 },
      { note: 'D2', time: 4.5, duration: 0.7, velocity: 0.82 },
      { note: 'C2', time: 5.25, duration: 1.1, velocity: 0.9 },
    ],
  },
  ukulele: {
    title: 'Ukulele Pop Trainer', composer: 'Polymath Musician', bpm: 108,
    notes: [
      { note: 'C4', time: 0, duration: 0.45, velocity: 0.82 },
      { note: 'E4', time: 0, duration: 0.45, velocity: 0.78 },
      { note: 'G4', time: 0, duration: 0.45, velocity: 0.8 },
      { note: 'A4', time: 0.55, duration: 0.45, velocity: 0.82 },
      { note: 'C5', time: 1.1, duration: 0.45, velocity: 0.84 },
      { note: 'E5', time: 1.65, duration: 0.65, velocity: 0.88 },
      { note: 'C5', time: 2.45, duration: 0.45, velocity: 0.82 },
      { note: 'A4', time: 3, duration: 0.45, velocity: 0.8 },
      { note: 'G4', time: 3.55, duration: 0.8, velocity: 0.86 },
    ],
  },
  'electric-guitar': {
    title: 'Electric Pop Riff Trainer', composer: 'Polymath Musician', bpm: 118,
    notes: [
      { note: 'E3', time: 0, duration: 0.3, velocity: 0.86 },
      { note: 'G3', time: 0.4, duration: 0.3, velocity: 0.82 },
      { note: 'A3', time: 0.8, duration: 0.55, velocity: 0.9 },
      { note: 'B3', time: 1.5, duration: 0.3, velocity: 0.84 },
      { note: 'D4', time: 1.9, duration: 0.3, velocity: 0.86 },
      { note: 'E4', time: 2.3, duration: 0.7, velocity: 0.92 },
      { note: 'D4', time: 3.15, duration: 0.3, velocity: 0.84 },
      { note: 'B3', time: 3.55, duration: 0.3, velocity: 0.82 },
      { note: 'A3', time: 3.95, duration: 0.85, velocity: 0.9 },
    ],
  },
  drums: {
    title: 'Electronic Pop Drum Trainer', composer: 'Polymath Musician', bpm: 120,
    notes: [
      { note: 'C2', time: 0, duration: 0.2, velocity: 0.95 },
      { note: 'F#2', time: 0, duration: 0.12, velocity: 0.68 },
      { note: 'F#2', time: 0.5, duration: 0.12, velocity: 0.64 },
      { note: 'D2', time: 1, duration: 0.2, velocity: 0.9 },
      { note: 'F#2', time: 1, duration: 0.12, velocity: 0.7 },
      { note: 'F#2', time: 1.5, duration: 0.12, velocity: 0.64 },
      { note: 'C2', time: 2, duration: 0.2, velocity: 0.95 },
      { note: 'F#2', time: 2, duration: 0.12, velocity: 0.68 },
      { note: 'C2', time: 2.5, duration: 0.2, velocity: 0.82 },
      { note: 'F#2', time: 2.5, duration: 0.12, velocity: 0.64 },
      { note: 'D2', time: 3, duration: 0.2, velocity: 0.92 },
      { note: 'C#3', time: 3, duration: 0.4, velocity: 0.84 },
      { note: 'A2', time: 3.5, duration: 0.2, velocity: 0.82 },
      { note: 'G2', time: 3.75, duration: 0.25, velocity: 0.88 },
    ],
  },
  synth: {
    title: 'Electronic Pop Synth Trainer', composer: 'Polymath Musician', bpm: 124,
    notes: [
      { note: 'C4', time: 0, duration: 0.4, velocity: 0.82 },
      { note: 'G4', time: 0.5, duration: 0.4, velocity: 0.84 },
      { note: 'A4', time: 1, duration: 0.4, velocity: 0.86 },
      { note: 'E4', time: 1.5, duration: 0.4, velocity: 0.8 },
      { note: 'F4', time: 2, duration: 0.4, velocity: 0.82 },
      { note: 'C5', time: 2.5, duration: 0.4, velocity: 0.88 },
      { note: 'G4', time: 3, duration: 0.4, velocity: 0.84 },
      { note: 'E4', time: 3.5, duration: 0.85, velocity: 0.86 },
    ],
  },
};

function demoSongFor(instrument) {
  return normalizeSong(DEMO_SONGS[instrument] || {
    ...DEMO_SONGS.fiddle,
    title: `${INSTRUMENT_BY_ID[instrument]?.label || 'Instrument'} Melody Trainer`,
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function findStartIndex(events, time) {
  let low = 0;
  let high = events.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (events[middle].time < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${minutes}:${String(remainder).padStart(2, '0')}`;
}

export default function EnsemblePage({ user, setUser, onNavigate }) {
  const [instrument, setInstrument] = useState('fiddle');
  const [song, setSong] = useState(() => demoSongFor('fiddle'));
  const [isCustomSong, setIsCustomSong] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [volume, setVolume] = useState(0.9);
  const [activeNotes, setActiveNotes] = useState(new Set());
  const [librarySongs, setLibrarySongs] = useState([]);
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [teachingMode, setTeachingMode] = useState('regular');
  const [preferredSectionSeconds, setPreferredSectionSeconds] = useState(15);
  const [selectedSectionIndex, setSelectedSectionIndex] = useState(0);
  const [repeatSection, setRepeatSection] = useState(true);
  const [practiceRange, setPracticeRange] = useState(null);

  const nextEventIndex = useRef(0);
  const startStamp = useRef(0);
  const pauseOffset = useRef(0);
  const scheduler = useRef(null);
  const animationFrame = useRef(null);
  const timers = useRef([]);
  const runId = useRef(0);

  const duration = useMemo(() => getSongDuration(song), [song]);
  const learningSections = useMemo(
    () => analyzeLearningSections(song.notes, duration, preferredSectionSeconds),
    [song, duration, preferredSectionSeconds],
  );
  const selectedInstrument = INSTRUMENT_BY_ID[instrument];
  const manualNotes = selectedInstrument?.manualNotes || [];
  const progress = duration > 0 ? clamp((currentTime / duration) * 100, 0, 100) : 0;
  const visibleNotes = useMemo(
    () => song.notes
      .filter((note) => note.time + note.duration >= currentTime - 0.2 && note.time <= currentTime + 4.2)
      .slice(0, 30),
    [song, currentTime],
  );
  const availableSongs = useMemo(() => [
    demoSongFor(instrument),
    ...librarySongs.filter((item) => item.instrument === instrument).map((item) => item.song),
  ], [instrument, librarySongs]);

  useEffect(() => {
    ensembleAudio.setMasterVolume(volume);
  }, [volume]);

  useEffect(() => {
    ensembleAudio.preloadInstrument(instrument);
  }, [instrument]);

  useEffect(() => {
    setSelectedSectionIndex(0);
    setPracticeRange(null);
  }, [song.title, instrument]);

  useEffect(() => {
    let cancelled = false;
    if (!user?.user_id) {
      setLibrarySongs([]);
      return undefined;
    }
    apiRequest('/api/library')
      .then(async ({ purchasedSongs }) => {
        const supported = purchasedSongs.filter((item) => ENSEMBLE_INSTRUMENTS.some((instrumentItem) => instrumentItem.id === item.instrument) && ['JSON', 'MIDI', 'MUSICXML'].includes(item.format));
        const loaded = await Promise.all(supported.map(async (item) => {
          const file = await fetchProtectedFile(`/api/listings/${item.id}/download`, item.filename || `${item.title}.${item.format.toLowerCase()}`);
          const parsed = normalizeSong(await parseUploadedSongFile(file));
          return { instrument: item.instrument, song: { ...parsed, title: item.title, composer: item.artist || parsed.composer } };
        }));
        if (!cancelled) setLibrarySongs(loaded);
      })
      .catch((error) => console.error('Purchased instrument songs could not be loaded:', error));
    return () => { cancelled = true; };
  }, [user?.user_id]);

  function clearTimers() {
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current = [];
  }

  function stopClocks() {
    if (scheduler.current) window.clearInterval(scheduler.current);
    if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
    scheduler.current = null;
    animationFrame.current = null;
    clearTimers();
  }

  function silence(position = currentTime) {
    runId.current += 1;
    stopClocks();
    ensembleAudio.stopAll(0.04);
    pauseOffset.current = clamp(Number(position) || 0, 0, duration);
    setActiveNotes(new Set());
  }

  function stopPlayback() {
    setIsPlaying(false);
    silence(0);
    pauseOffset.current = 0;
    nextEventIndex.current = 0;
    setCurrentTime(0);
  }

  async function startAt(position, speedOverride = speed) {
    const safePosition = clamp(Number(position) || 0, 0, duration);
    const safeSpeed = clamp(Number(speedOverride) || 1, 0.2, 1.75);
    runId.current += 1;
    ensembleAudio.ensure();
    nextEventIndex.current = findStartIndex(song.notes, Math.max(0, safePosition - 0.0005));
    pauseOffset.current = safePosition;
    startStamp.current = performance.now() - (safePosition * 1000) / safeSpeed;
    setCurrentTime(safePosition);
    setIsPlaying(true);
    setPlaybackEpoch((value) => value + 1);
  }

  async function togglePlayPause() {
    if (isPlaying) {
      silence(currentTime);
      setIsPlaying(false);
      return;
    }
    await startAt(pauseOffset.current);
  }

  async function seekTo(value) {
    const target = clamp(Number(value) || 0, 0, duration);
    const resume = isPlaying;
    silence(target);
    setIsPlaying(false);
    pauseOffset.current = target;
    setCurrentTime(target);
    if (resume) await startAt(target);
  }

  async function changeSpeed(value) {
    const nextSpeed = clamp(Number(value) || 1, 0.2, 1.75);
    const resume = isPlaying;
    const position = currentTime;
    if (resume) {
      silence(position);
      setIsPlaying(false);
    }
    setSpeed(nextSpeed);
    if (resume) await startAt(position, nextSpeed);
  }

  function changeInstrument(nextInstrument) {
    silence(0);
    setIsPlaying(false);
    setCurrentTime(0);
    pauseOffset.current = 0;
    setInstrument(nextInstrument);
    if (!isCustomSong) setSong(demoSongFor(nextInstrument));
  }

  useEffect(() => {
    if (!isPlaying) return undefined;
    const currentRun = runId.current;
    const playbackEnd = teachingMode === 'learn' && practiceRange
      ? Math.min(duration, practiceRange.end)
      : duration;

    function songTime(now = performance.now()) {
      return ((now - startStamp.current) / 1000) * speed;
    }

    function scheduleEvents() {
      if (runId.current !== currentRun) return;
      const nowSong = songTime();
      const audioNow = ensembleAudio.getCurrentTime();
      const lookAhead = nowSong + LOOKAHEAD_SECONDS * speed;

      while (nextEventIndex.current < song.notes.length && song.notes[nextEventIndex.current].time <= lookAhead && song.notes[nextEventIndex.current].time < playbackEnd) {
        const event = song.notes[nextEventIndex.current];
        nextEventIndex.current += 1;
        const delay = Math.max(0, (event.time - nowSong) / speed);
        const eventDuration = Math.max(0.05, Number(event.audioDuration ?? event.duration ?? 0.45) / speed);
        ensembleAudio.playAt(event.note, instrument, event.velocity, eventDuration, audioNow + delay);
        const startTimer = window.setTimeout(() => {
          if (runId.current !== currentRun) return;
          setActiveNotes((previous) => new Set(previous).add(event.note));
          const endTimer = window.setTimeout(() => {
            if (runId.current !== currentRun) return;
            setActiveNotes((previous) => {
              const next = new Set(previous);
              next.delete(event.note);
              return next;
            });
          }, eventDuration * 1000 + 80);
          timers.current.push(endTimer);
        }, delay * 1000);
        timers.current.push(startTimer);
      }

      if (nowSong >= playbackEnd) {
        if (teachingMode === 'learn' && practiceRange && repeatSection) {
          const restartAt = practiceRange.start;
          silence(restartAt);
          setIsPlaying(false);
          setCurrentTime(restartAt);
          window.setTimeout(() => startAt(restartAt), 35);
        } else {
          silence(practiceRange?.start || 0);
          setIsPlaying(false);
          setCurrentTime(playbackEnd);
        }
      }
    }

    function tick(now) {
      if (runId.current !== currentRun) return;
      const next = songTime(now);
      setCurrentTime(Math.min(next, playbackEnd));
      if (next >= playbackEnd) {
        return;
      }
      animationFrame.current = requestAnimationFrame(tick);
    }

    scheduleEvents();
    scheduler.current = window.setInterval(scheduleEvents, SCHEDULER_INTERVAL_MS);
    animationFrame.current = requestAnimationFrame(tick);
    return stopClocks;
  }, [isPlaying, song, speed, instrument, duration, playbackEpoch, teachingMode, practiceRange, repeatSection]);

  function selectLearningSection(index) {
    const safeIndex = clamp(index, 0, Math.max(0, learningSections.length - 1));
    const section = learningSections[safeIndex];
    if (!section) return;
    silence(section.start);
    setIsPlaying(false);
    setSelectedSectionIndex(safeIndex);
    setPracticeRange(section);
    setCurrentTime(section.start);
  }

  async function practiseLearningSection(section) {
    const index = learningSections.findIndex((candidate) => candidate.id === section.id);
    selectLearningSection(index < 0 ? 0 : index);
    setPracticeRange(section);
    await startAt(section.start);
  }

  async function loadReadySheet(file) {
    const isBrowserFile = file && typeof file.arrayBuffer === 'function' && typeof file.name === 'string';
    const parsed = isBrowserFile ? await parseUploadedSongFile(file) : file;
    if (!parsed.notes?.length) throw new Error('No playable note events were found in this ready-to-play sheet.');
    const normalized = normalizeSong(parsed);
    stopPlayback();
    setSong(normalized);
    setIsCustomSong(true);
    return normalized;
  }

  function playManualNote(note) {
    const durationSeconds = instrument === 'fiddle' || instrument === 'dobro' ? 1.05 : instrument === 'drums' ? 0.32 : 0.72;
    ensembleAudio.play(note, instrument, 0.86, durationSeconds);
    setActiveNotes((previous) => new Set(previous).add(note));
    const timer = window.setTimeout(() => {
      setActiveNotes((previous) => {
        const next = new Set(previous);
        next.delete(note);
        return next;
      });
    }, durationSeconds * 1000 + 100);
    timers.current.push(timer);
  }

  function restoreInstrumentDemo() {
    stopPlayback();
    setSong(demoSongFor(instrument));
    setIsCustomSong(false);
  }

  function chooseAvailableSong(title) {
    const selected = availableSongs.find((candidate) => candidate.title === title);
    if (!selected) return;
    stopPlayback();
    setSong(selected);
    setIsCustomSong(selected.title !== demoSongFor(instrument).title);
  }

  return (
    <section className="page-shell ensemble-page">
      <div className="page-heading split-heading">
        <div>
          <p className="eyebrow">Polymath Musician Visual Teacher</p>
          <h1>Choose an instrument and follow the target.</h1>
          <p>Every note now lights up on the physical playing surface—not only in a falling-note lane. Learn strings, frets, finger positions, slide positions, drum strikes, and electronic-pop keys by sight.</p>
        </div>
        <div className="teacher-trust-card">
          <strong>Visual-first learning</strong>
          <span>Play • watch • copy • repeat</span>
          <small>Ready-to-play JSON/MIDI and OpenAI PDF translation supported.</small>
        </div>
      </div>

      <div className="ensemble-instrument-grid" aria-label="Choose an instrument">
        {ENSEMBLE_INSTRUMENTS.map((item) => (
          <button key={item.id} type="button" className={instrument === item.id ? 'active' : ''} onClick={() => changeInstrument(item.id)}>
            <InstrumentIcon instrument={item.id} />
            <strong>{item.label}</strong>
            <small>{item.description}</small>
          </button>
        ))}
      </div>

      <LearnModePanel
        mode={teachingMode}
        onModeChange={(mode) => {
          stopPlayback();
          setTeachingMode(mode);
          if (mode === 'regular') setPracticeRange(null);
        }}
        sections={learningSections}
        selectedIndex={selectedSectionIndex}
        onSelectSection={selectLearningSection}
        onPracticeSection={practiseLearningSection}
        repeatSection={repeatSection}
        onRepeatChange={setRepeatSection}
        preferredSeconds={preferredSectionSeconds}
        onPreferredSecondsChange={(value) => setPreferredSectionSeconds(clamp(value || 15, 5, 60))}
      />

      <div className="ensemble-layout">
        <aside className="ensemble-control-card">
          <div>
            <p className="eyebrow">Selected teacher</p>
            <h2>{selectedInstrument.label}</h2>
            <p className="muted">{selectedInstrument.description}</p>
          </div>

          <label className="field">Available songs
            <select value={song.title} onChange={(event) => chooseAvailableSong(event.target.value)}>
              {!availableSongs.some((candidate) => candidate.title === song.title) && <option value={song.title}>{song.title} — uploaded</option>}
              {availableSongs.map((candidate) => <option key={candidate.title} value={candidate.title}>{candidate.title}</option>)}
            </select>
          </label>

          <div className="teacher-mode-badge">
            <span>LIVE</span>
            <strong>Physical target guidance</strong>
            <small>Glowing markers show exactly where the learner should press, pluck, bow, slide, strike, or tap.</small>
          </div>

          <label className="field">Volume: {Math.round(volume * 100)}%
            <input type="range" min="0.2" max="1.1" step="0.05" value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
          </label>
          <label className="field">Playback speed: {speed.toFixed(2)}×
            <input type="range" min="0.2" max="1.75" step="0.05" value={speed} onChange={(event) => changeSpeed(event.target.value)} />
          </label>

          <div>
            <div className="mini-heading-row">
              <strong>Quick practice targets</strong>
              {isCustomSong && <button type="button" className="text-button" onClick={restoreInstrumentDemo}>Restore demo</button>}
            </div>
            <div className="manual-note-grid">
              {manualNotes.map((note) => (
                <button key={note} type="button" className={activeNotes.has(note) ? 'active' : ''} onClick={() => playManualNote(note)}>{note}</button>
              ))}
            </div>
          </div>
        </aside>

        <div className="ensemble-stage-card">
          <header>
            <div>
              <p className="eyebrow">Now teaching</p>
              <h2>{song.title}</h2>
              <p className="muted">{song.composer || selectedInstrument.label} · {song.notes.length} playable targets</p>
            </div>
            <InstrumentIcon instrument={selectedInstrument.id} size="lg" />
          </header>

          <InstrumentTeacherSurface
            instrument={instrument}
            activeNotes={activeNotes}
            onPlay={playManualNote}
          />

          <div className="timing-lane-heading">
            <div>
              <p className="eyebrow">Timing lane</p>
              <strong>Use the lane for when; use the instrument for where.</strong>
            </div>
            <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
          </div>

          <div className="ensemble-note-stage compact-timing-lane">
            <div className="ensemble-time-line" style={{ left: `${progress}%` }} />
            {visibleNotes.map((note, index) => {
              const left = duration > 0 ? clamp((note.time / duration) * 100, 0, 100) : 0;
              const width = duration > 0 ? clamp(((note.duration || 0.4) / duration) * 100, 1.5, 18) : 4;
              return (
                <div
                  key={note.id || `${note.note}-${note.time}-${index}`}
                  className={`ensemble-note-pill ${activeNotes.has(note.note) ? 'active' : ''}`}
                  style={{ left: `${left}%`, width: `${width}%`, '--note-row': index % 4 }}
                  title={`${note.note} at ${note.time.toFixed(2)} seconds`}
                >
                  {note.note}
                </div>
              );
            })}
          </div>

          <div className="ensemble-transport">
            <button className="primary" type="button" onClick={togglePlayPause}>{isPlaying ? 'Pause lesson' : 'Play lesson'}</button>
            <button className="ghost" type="button" onClick={stopPlayback}>Restart</button>
            <span>{formatTime(currentTime)}</span>
            <input type="range" min="0" max={duration || 0} step="0.01" value={Math.min(currentTime, duration)} onChange={(event) => seekTo(event.target.value)} aria-label="Song timeline" />
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <aside className="ensemble-upload-card">
          <MusicUploadPanel
            compact
            user={user}
            setUser={setUser}
            onNavigate={onNavigate}
            instrument={instrument}
            onReadyFile={loadReadySheet}
            enableMedia
          />
        </aside>
      </div>
    </section>
  );
}
