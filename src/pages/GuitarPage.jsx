import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import GuitarFallingNotes from '../components/GuitarFallingNotes.jsx';
import GuitarTransport from '../components/GuitarTransport.jsx';
import MusicUploadPanel from '../components/MusicUploadPanel.jsx';
import MusicChoiceDisclosure from '../components/MusicChoiceDisclosure.jsx';
import LearnModePanel from '../components/LearnModePanel.jsx';
import { GUITAR_TONE_LABELS, guitarAudio } from '../engine/guitarEngine.js';
import { assignNotesToStrings } from '../engine/guitarVoicing.js';
import { parseUploadedSongFile } from '../utils/songParser.js';
import { apiRequest, fetchProtectedFile } from '../services/api.js';
import { analyzeLearningSections } from '../utils/learningSections.js';

const AUDIO_LOOKAHEAD_SECONDS = 0.14;
const SCHEDULER_INTERVAL_MS = 25;
const STRINGS = ['Low E', 'A', 'D', 'G', 'B', 'High E'];

const CHORDS = {
  C: [-1, 3, 2, 0, 1, 0],
  G: [3, 2, 0, 0, 0, 3],
  Am: [-1, 0, 2, 2, 1, 0],
  F: [1, 3, 3, 2, 1, 1],
  D: [-1, -1, 0, 2, 3, 2],
  Em: [0, 2, 2, 0, 0, 0],
  E: [0, 2, 2, 1, 0, 0],
  A: [-1, 0, 2, 2, 2, 0],
  Dm: [-1, -1, 0, 2, 3, 1],
  Bm: [-1, 2, 4, 4, 3, 2],
  G7: [3, 2, 0, 0, 0, 1],
  Cmaj7: [-1, 3, 2, 0, 0, 0],
  Am7: [-1, 0, 2, 0, 1, 0],
};

const FREE_GUITAR_LESSON_URL = `${import.meta.env.BASE_URL}songs/wildestDreamsTaylorSwiftGuitar.json`;

const DEMO_LESSON = normalizeLesson({
  title: 'Four-chord acoustic lesson',
  artist: 'Practice session',
  bpm: 78,
  events: [
    { time: 0, chord: 'C', duration: 1.9, direction: 'down' },
    { time: 2, chord: 'G', duration: 1.9, direction: 'down' },
    { time: 4, chord: 'Am', duration: 1.9, direction: 'down' },
    { time: 6, chord: 'F', duration: 1.9, direction: 'down' },
    { time: 8, chord: 'C', duration: 1.9, direction: 'up' },
    { time: 10, chord: 'G', duration: 1.9, direction: 'up' },
    { time: 12, chord: 'Am', duration: 1.9, direction: 'down' },
    { time: 14, chord: 'F', duration: 2.2, direction: 'down' },
  ],
});

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeStringIndex(value) {
  if (Number.isInteger(value) && value >= 0 && value <= 5) return value;
  const number = Number(value);
  if (Number.isInteger(number) && number >= 1 && number <= 6) return number - 1;
  const text = String(value || '').trim().toLowerCase();
  const names = {
    'low e': 0,
    e2: 0,
    a: 1,
    a2: 1,
    d: 2,
    d3: 2,
    g: 3,
    g3: 3,
    b: 4,
    b3: 4,
    'high e': 5,
    e4: 5,
  };
  return names[text] ?? null;
}

function normalizeFrets(value) {
  if (!Array.isArray(value)) return null;
  return Array.from({ length: 6 }, (_, index) => {
    const fret = Number(value[index]);
    return Number.isFinite(fret) ? clamp(Math.round(fret), -1, 24) : -1;
  });
}

function normalizeLesson(raw, metadata = {}) {
  const sourceEvents = raw?.events || raw?.tabs || raw?.notes || [];
  const events = sourceEvents
    .map((event, index) => {
      const time = Number(event.time ?? event.at ?? event.start ?? 0);
      if (!Number.isFinite(time) || time < 0) return null;
      const chord = String(event.chord || '').trim();
      const frets = normalizeFrets(event.frets || event.shape) || CHORDS[chord] || null;
      const stringIndex = normalizeStringIndex(event.stringIndex ?? event.string);
      const fret = Number(event.fret);

      if (!frets && (stringIndex === null || !Number.isFinite(fret))) return null;

      return {
        ...event,
        id: event.id || `guitar-${index}-${time.toFixed(4)}`,
        time: Number(time.toFixed(4)),
        duration: clamp(Number(event.duration ?? event.length ?? 1.4) || 1.4, 0.08, 12),
        velocity: clamp(Number(event.velocity ?? 0.8) || 0.8, 0.04, 1),
        chord: chord || undefined,
        frets,
        stringIndex: frets ? undefined : stringIndex,
        fret: frets ? undefined : clamp(Math.round(fret), 0, 24),
        direction: event.direction === 'up' ? 'up' : 'down',
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.time - b.time);

  return {
    ...raw,
    ...metadata,
    title: raw?.title || metadata.sourceFileName || 'Uploaded guitar lesson',
    artist: raw?.artist || raw?.composer || '',
    bpm: Number(raw?.bpm || raw?.tempo || 80),
    events,
  };
}

function lessonLabel(lesson) {
  const artist = String(lesson?.artist || '').trim();
  return artist ? `${lesson.title} (${artist})` : lesson.title;
}

function lessonDuration(lesson) {
  return Math.max(0, ...(lesson.events || []).map((event) => event.time + event.duration)) + 0.35;
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

function fileExtension(filename) {
  return String(filename || '').split('.').pop()?.toLowerCase() || '';
}


function songToGuitarLesson(song, metadata = {}) {
  const notes = Array.isArray(song?.notes) ? song.notes : [];
  const groups = [];

  notes
    .filter((note) => Number.isFinite(Number(note.time)))
    .sort((a, b) => Number(a.time) - Number(b.time))
    .forEach((note) => {
      const time = Number(note.time);
      const latest = groups[groups.length - 1];
      if (latest && Math.abs(latest.time - time) <= 0.025) latest.notes.push(note);
      else groups.push({ time, notes: [note] });
    });

  const events = [];
  groups.forEach((group, groupIndex) => {
    const assignments = assignNotesToStrings(group.notes);
    if (!assignments.length) return;
    const duration = clamp(
      Math.max(...assignments.map(({ note }) => Number(note.audioDuration ?? note.duration ?? 0.45) || 0.45)),
      0.08,
      12,
    );
    const velocity = clamp(
      assignments.reduce((sum, { note }) => sum + (Number(note.velocity) || 0.72), 0) / assignments.length,
      0.04,
      1,
    );

    if (assignments.length === 1) {
      const assignment = assignments[0];
      events.push({
        id: `imported-guitar-${groupIndex}`,
        time: Number(group.time.toFixed(4)),
        duration,
        velocity,
        stringIndex: assignment.stringIndex,
        fret: assignment.fret,
        direction: 'down',
      });
      return;
    }

    const frets = Array(6).fill(-1);
    assignments.forEach(({ stringIndex, fret }) => { frets[stringIndex] = fret; });
    events.push({
      id: `imported-guitar-${groupIndex}`,
      time: Number(group.time.toFixed(4)),
      duration,
      velocity,
      frets,
      direction: 'down',
      strumSpeed: 0.012,
    });
  });

  return normalizeLesson({
    title: song?.title || metadata.sourceFileName || 'Imported guitar lesson',
    artist: song?.artist || song?.composer || '',
    bpm: Number(song?.bpm || song?.tempo || 80),
    youtubeSearchQuery: song?.youtubeSearchQuery || metadata.sourceFolderName || '',
    events,
  }, metadata);
}

export default function GuitarPage({ user, setUser, onNavigate }) {
  const [selectedChord, setSelectedChord] = useState('C');
  const [lesson, setLesson] = useState(DEMO_LESSON);
  const [freeLessons, setFreeLessons] = useState([DEMO_LESSON]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [leadTime, setLeadTime] = useState(3.8);
  const [toneMode, setToneMode] = useState('lounge');
  const [capoFret, setCapoFret] = useState(0);
  const [volume, setVolume] = useState(0.82);
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [teachingMode, setTeachingMode] = useState('regular');
  const [preferredSectionSeconds, setPreferredSectionSeconds] = useState(15);
  const [selectedSectionIndex, setSelectedSectionIndex] = useState(0);
  const [repeatSection, setRepeatSection] = useState(true);
  const [practiceRange, setPracticeRange] = useState(null);
  const [manualStringVibration, setManualStringVibration] = useState(null);
  const [openMusicChooser, setOpenMusicChooser] = useState(null);

  const nextEventIndex = useRef(0);
  const startStamp = useRef(0);
  const pauseOffset = useRef(0);
  const scheduler = useRef(null);
  const animationFrame = useRef(null);
  const runId = useRef(0);
  const seekWasPlaying = useRef(false);
  const vibrationTimer = useRef(null);
  const guitarPlayerRef = useRef(null);

  const duration = useMemo(() => lessonDuration(lesson), [lesson]);
  const learningSections = useMemo(
    () => analyzeLearningSections(lesson.events, duration, preferredSectionSeconds),
    [lesson, duration, preferredSectionSeconds],
  );
  const activeEvent = useMemo(() => {
    let latest = null;
    for (const event of lesson.events) {
      if (event.time > currentTime + 0.03) break;
      if (currentTime <= event.time + event.duration + 0.08) latest = event;
    }
    return latest;
  }, [lesson, currentTime]);

  useEffect(() => {
    guitarAudio.setToneMode(toneMode);
  }, [toneMode]);

  useEffect(() => {
    guitarAudio.setCapoFret(capoFret);
    guitarAudio.stopAll(0.025);
  }, [capoFret]);

  useEffect(() => {
    guitarAudio.preloadSamples();
  }, []);

  useEffect(() => {
    guitarAudio.setMasterVolume(volume);
  }, [volume]);

  useEffect(() => () => {
    if (vibrationTimer.current) window.clearTimeout(vibrationTimer.current);
    runId.current += 1;
    stopClocks();
    guitarAudio.suspendAfter(90);
  }, []);

  useEffect(() => {
    setSelectedSectionIndex(0);
    setPracticeRange(null);
  }, [lesson.title]);

  useEffect(() => {
    let cancelled = false;

    fetch(FREE_GUITAR_LESSON_URL)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load the free guitar song.');
        return response.json();
      })
      .then((rawLesson) => {
        if (cancelled) return;
        const featuredLesson = normalizeLesson({
          ...rawLesson,
          libraryType: 'free',
          readyToPlay: true,
        });
        if (!featuredLesson.events.length) return;
        setFreeLessons((previous) => [
          ...previous.filter((candidate) => candidate.title !== featuredLesson.title),
          featuredLesson,
        ]);
      })
      .catch((error) => console.error(error));

    return () => {
      cancelled = true;
    };
  }, []);

  function clampTime(value) {
    return clamp(Number(value) || 0, 0, duration);
  }

  function stopClocks() {
    if (scheduler.current) window.clearInterval(scheduler.current);
    if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
    scheduler.current = null;
    animationFrame.current = null;
  }

  function silence(position = currentTime) {
    runId.current += 1;
    stopClocks();
    pauseOffset.current = clampTime(position);
    guitarAudio.stopAll(0.025);
    if (vibrationTimer.current) window.clearTimeout(vibrationTimer.current);
    vibrationTimer.current = null;
    setManualStringVibration(null);
  }

  function pulseStrings(primaryStrings, mutedStrings = [], milliseconds = 1250) {
    const primary = new Set(primaryStrings);
    const muted = new Set(mutedStrings);
    const sympathetic = STRINGS
      .map((_, index) => index)
      .filter((index) => !primary.has(index) && !muted.has(index));
    setManualStringVibration({ primary: [...primary], sympathetic });
    if (vibrationTimer.current) window.clearTimeout(vibrationTimer.current);
    vibrationTimer.current = window.setTimeout(() => {
      setManualStringVibration(null);
      vibrationTimer.current = null;
    }, milliseconds);
  }

  async function pressFret(stringIndex, fret) {
    if (!(await guitarAudio.prepareForPlayback())) return;
    guitarAudio.pluck(stringIndex, fret, 0.82, null, {
      duration: 3.2,
      releaseSeconds: 0.48,
    });
    pulseStrings([stringIndex]);
  }

  async function startAt(position, speedOverride = speed) {
    const target = clampTime(position);
    const safeSpeed = clamp(Number(speedOverride) || 1, 0.2, 1.75);
    runId.current += 1;
    if (!(await guitarAudio.prepareForPlayback())) return;
    nextEventIndex.current = findStartIndex(lesson.events, Math.max(0, target - 0.0005));
    pauseOffset.current = target;
    startStamp.current = performance.now() - (target * 1000) / safeSpeed;
    setCurrentTime(target);
    setIsPlaying(true);
    setPlaybackEpoch((value) => value + 1);
  }

  function stopLesson() {
    setIsPlaying(false);
    silence(0);
    pauseOffset.current = 0;
    nextEventIndex.current = 0;
    setCurrentTime(0);
  }

  async function togglePlayPause() {
    if (isPlaying) {
      silence(currentTime);
      setIsPlaying(false);
      return;
    }
    await startAt(pauseOffset.current);
  }

  function beginSeek() {
    seekWasPlaying.current = isPlaying;
    silence(currentTime);
    setIsPlaying(false);
  }

  function previewSeek(position) {
    const target = clampTime(position);
    pauseOffset.current = target;
    setCurrentTime(target);
  }

  async function commitSeek(position) {
    const target = clampTime(position);
    const resume = seekWasPlaying.current;
    seekWasPlaying.current = false;
    previewSeek(target);
    if (resume) await startAt(target);
  }

  async function jumpBy(seconds) {
    const resume = isPlaying;
    const target = clampTime(currentTime + seconds);
    silence(currentTime);
    setIsPlaying(false);
    previewSeek(target);
    if (resume) await startAt(target);
  }

  async function changeSpeed(value) {
    const safe = clamp(Number(value) || 1, 0.2, 1.75);
    const resume = isPlaying;
    const position = currentTime;
    if (resume) {
      silence(position);
      setIsPlaying(false);
    }
    setSpeed(safe);
    if (resume) await startAt(position, safe);
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
      const audioNow = guitarAudio.getCurrentTime();
      const lookAhead = nowSong + AUDIO_LOOKAHEAD_SECONDS * speed;

      while (nextEventIndex.current < lesson.events.length && lesson.events[nextEventIndex.current].time <= lookAhead && lesson.events[nextEventIndex.current].time < playbackEnd) {
        const event = lesson.events[nextEventIndex.current];
        nextEventIndex.current += 1;
        const delay = Math.max(0, (event.time - nowSong) / speed);
        guitarAudio.playEvent(event, audioNow + delay, speed);
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
  }, [isPlaying, lesson, speed, duration, playbackEpoch, teachingMode, practiceRange, repeatSection]);

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

  const readLessonFile = useCallback(async (file) => {
    const extension = fileExtension(file.name);
    const relativePath = String(file.webkitRelativePath || '');
    const folderName = relativePath.split('/').filter(Boolean)[0] || '';
    const metadata = {
      sourceFileName: file.name,
      sourceFolderName: folderName,
      youtubeSearchQuery: folderName,
    };

    if (extension === 'json') {
      const parsed = JSON.parse(await file.text());
      const looksLikeGuitar = Array.isArray(parsed?.events)
        || Array.isArray(parsed?.tabs)
        || (Array.isArray(parsed?.notes) && parsed.notes.some((note) => (
          note.chord || note.frets || note.shape || note.stringIndex !== undefined || note.string !== undefined
        )));
      if (looksLikeGuitar) {
        return normalizeLesson(parsed, {
          ...metadata,
          youtubeSearchQuery: parsed.youtubeSearchQuery || folderName,
        });
      }
    }

    if (['json', 'csv', 'mid', 'midi', 'musicxml', 'xml'].includes(extension)) {
      const importedSong = await parseUploadedSongFile(file);
      return songToGuitarLesson(importedSong, {
        ...metadata,
        youtubeSearchQuery: importedSong.youtubeSearchQuery || folderName,
      });
    }

    throw new Error('Use a ready-to-play guitar JSON, CSV, MIDI, or MusicXML file.');
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!user?.user_id) return undefined;
    apiRequest('/api/library')
      .then(async ({ purchasedSongs }) => {
        const guitarSongs = purchasedSongs.filter((item) => ['guitar', 'electric-guitar'].includes(item.instrument) && ['JSON', 'MIDI', 'MUSICXML'].includes(item.format));
        const loaded = await Promise.all(guitarSongs.map(async (item) => {
          const file = await fetchProtectedFile(`/api/listings/${item.id}/download`, item.filename || `${item.title}.${item.format.toLowerCase()}`);
          const parsed = await readLessonFile(file);
          return { ...parsed, title: item.title, artist: item.artist || parsed.artist };
        }));
        if (cancelled || !loaded.length) return;
        setFreeLessons((previous) => [...previous.filter((candidate) => !loaded.some((item) => item.title === candidate.title)), ...loaded]);
      })
      .catch((error) => console.error(`Your purchased guitar songs could not be loaded: ${error.message}`));
    return () => { cancelled = true; };
  }, [readLessonFile, user?.user_id]);

  async function loadReadyLesson(file, { commit = true, prepared = null } = {}) {
    const parsed = prepared || await readLessonFile(file);
    if (!parsed.events.length) throw new Error('No playable guitar events were found.');
    if (!commit) return parsed;
    stopLesson();
    setLesson(parsed);
    setOpenMusicChooser(null);
    return parsed;
  }

  function chooseFreeLesson(title) {
    const selected = freeLessons.find((candidate) => candidate.title === title);
    if (!selected) return;
    stopLesson();
    setLesson(selected);
  }

  function focusGuitarPlayer() {
    setOpenMusicChooser(null);
    window.setTimeout(() => {
      const player = guitarPlayerRef.current;
      if (!player) return;
      player.scrollIntoView({ behavior: 'smooth', block: 'start' });
      player.focus({ preventScroll: true });
    }, 80);
  }

  async function playChord(chord) {
    setSelectedChord(chord);
    guitarAudio.setToneMode(toneMode);
    if (!(await guitarAudio.prepareForPlayback())) return;
    guitarAudio.strum(CHORDS[chord], 0.82, 'down', null, {
      duration: 3.1,
      releaseSeconds: 0.5,
    });
    pulseStrings(
      CHORDS[chord].map((fret, index) => (fret >= 0 ? index : null)).filter((index) => index !== null),
      CHORDS[chord].map((fret, index) => (fret < 0 ? index : null)).filter((index) => index !== null),
    );
  }

  const displayedFrets = activeEvent?.frets || CHORDS[selectedChord];
  const scheduledStringVibration = useMemo(() => {
    if (!isPlaying || !activeEvent) return null;
    if (Array.isArray(activeEvent.frets)) {
      const primary = activeEvent.frets
        .map((fret, index) => (Number(fret) >= 0 ? index : null))
        .filter((index) => index !== null);
      return { primary, sympathetic: [] };
    }
    if (Number.isInteger(activeEvent.stringIndex)) {
      return {
        primary: [activeEvent.stringIndex],
        sympathetic: STRINGS.map((_, index) => index).filter((index) => index !== activeEvent.stringIndex),
      };
    }
    return null;
  }, [activeEvent, isPlaying]);
  const visibleStringVibration = manualStringVibration || scheduledStringVibration;

  return (
    <section className="page-shell guitar-page">
      <LearnModePanel
        mode={teachingMode}
        locked={!user?.admin && !user?.access?.learn}
        onUpgrade={() => onNavigate('payment', { productId: 'polymath-musician-monthly' })}
        onModeChange={(mode) => {
          stopLesson();
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

      <div className="guitar-studio-grid">
        <aside className="guitar-control-panel">
          <MusicChoiceDisclosure
            id="guitar-available-songs"
            title="Choose available songs"
            summary={lessonLabel(lesson)}
            expanded={openMusicChooser === 'available'}
            onToggle={() => setOpenMusicChooser((current) => current === 'available' ? null : 'available')}
          >
            <label className="field">Song
              <select value={lesson.title} onChange={(event) => chooseFreeLesson(event.target.value)}>
                {!freeLessons.some((candidate) => candidate.title === lesson.title) && (
                  <option value={lesson.title}>{lessonLabel(lesson)} — uploaded</option>
                )}
                {freeLessons.map((candidate) => (
                  <option key={candidate.title} value={candidate.title}>
                    {lessonLabel(candidate)}
                  </option>
                ))}
              </select>
            </label>

            <button className="primary song-play-now" type="button" onClick={focusGuitarPlayer}>
              Play now
            </button>
          </MusicChoiceDisclosure>
        </aside>

        <div ref={guitarPlayerRef} className="guitar-visual-stack" tabIndex="-1">
          <GuitarFallingNotes
            lesson={lesson}
            currentTime={currentTime}
            isPlaying={isPlaying}
            leadTime={leadTime}
            activeEventId={activeEvent?.id}
          />

          <GuitarTransport
            duration={duration}
            currentTime={currentTime}
            isPlaying={isPlaying}
            speed={speed}
            onSpeedChange={changeSpeed}
            onPlayPause={togglePlayPause}
            onStop={stopLesson}
            onSeekStart={beginSeek}
            onSeekPreview={previewSeek}
            onSeekCommit={commitSeek}
            onRewind={() => jumpBy(-10)}
            onForward={() => jumpBy(10)}
            minSpeed={0.2}
          />

          <details className="lesson-options player-settings">
            <summary>Settings</summary>
            <div className="lesson-options-content">
              <label className="field">Guitar sound
                <select value={toneMode} onChange={(event) => setToneMode(event.target.value)}>
                  {Object.entries(GUITAR_TONE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                </select>
              </label>
              <label className="field">Capo
                <select value={capoFret} onChange={(event) => setCapoFret(Number(event.target.value))}>
                  <option value="0">No capo</option>
                  {Array.from({ length: 7 }, (_, index) => index + 1).map((fret) => (
                    <option key={fret} value={fret}>Fret {fret}</option>
                  ))}
                </select>
              </label>
              <label className="field">Guitar volume: {Math.round(volume * 100)}%
                <input type="range" min="0.25" max="1.1" step="0.05" value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
              </label>
              <label className="field">Notes appear {leadTime.toFixed(1)}s early
                <input type="range" min="1.5" max="6" step="0.1" value={leadTime} onChange={(event) => setLeadTime(Number(event.target.value))} />
              </label>
            </div>
          </details>

          <details className="song-tools player-settings">
            <summary>Chord practice</summary>
            <div className="guitar-chord-grid">
              {Object.keys(CHORDS).map((chord) => (
                <button key={chord} type="button" className={selectedChord === chord ? 'active' : ''} onClick={() => playChord(chord)}>{chord}</button>
              ))}
            </div>
          </details>

          <section className="interactive-fretboard-card">
            <header>
              <div>
                <p className="eyebrow">Close-miked steel-string</p>
                <h2>{activeEvent?.chord || selectedChord}</h2>
                <small className="guitar-capo-status">{capoFret ? `Capo ${capoFret} · chord shapes stay relative to the capo` : 'Open tuning · no capo'}</small>
              </div>
              <button
                className="primary"
                type="button"
                onClick={async () => {
                  if (!(await guitarAudio.prepareForPlayback())) return;
                  guitarAudio.strum(displayedFrets, 0.84, 'down', null, {
                    duration: 3.1,
                    releaseSeconds: 0.5,
                  });
                  pulseStrings(
                    displayedFrets.map((fret, index) => (fret >= 0 ? index : null)).filter((index) => index !== null),
                    displayedFrets.map((fret, index) => (fret < 0 ? index : null)).filter((index) => index !== null),
                  );
                }}
              >
                Strum chord
              </button>
            </header>
            <div className="fretboard wide" role="grid" aria-label="Interactive guitar fretboard">
              {STRINGS.map((stringName, stringIndex) => (
                <div
                  className={`guitar-string ${visibleStringVibration?.primary.includes(stringIndex)
                    ? 'vibrating-primary'
                    : visibleStringVibration?.sympathetic.includes(stringIndex)
                      ? 'vibrating-sympathetic'
                      : ''}`}
                  key={stringName}
                >
                  <span className="string-name">{stringName}</span>
                  {Array.from({ length: 13 }, (_, fret) => {
                    const active = displayedFrets?.[stringIndex] === fret;
                    const muted = displayedFrets?.[stringIndex] < 0 && fret === 0;
                    return (
                      <button
                        key={fret}
                        type="button"
                        className={`${active ? 'active' : ''} ${muted ? 'muted-string' : ''}`}
                        onPointerDown={(event) => {
                          event.preventDefault();
                          event.currentTarget.setPointerCapture?.(event.pointerId);
                          pressFret(stringIndex, fret);
                        }}
                        onClick={(event) => {
                          if (event.detail === 0) {
                            pressFret(stringIndex, fret);
                          }
                        }}
                        aria-label={`${stringName} string fret ${fret}`}
                      >
                        <span>{muted ? '×' : active ? fret : ''}</span>
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </section>
        </div>

        <aside className="guitar-upload-card">
          <MusicChoiceDisclosure
            id="guitar-choose-music"
            title="Choose music"
            expanded={openMusicChooser === 'upload'}
            onToggle={() => setOpenMusicChooser((current) => current === 'upload' ? null : 'upload')}
          >
            <MusicUploadPanel
              compact
              user={user}
              setUser={setUser}
              onNavigate={onNavigate}
              instrument="guitar"
              onReadyFile={loadReadyLesson}
            />
          </MusicChoiceDisclosure>
        </aside>
      </div>
    </section>
  );
}
