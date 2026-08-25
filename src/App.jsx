import { useEffect, useMemo, useRef, useState } from 'react';
import AppNav from './components/AppNav.jsx';
import HeaderActions from './components/HeaderActions.jsx';
import ControlPanel from './components/ControlPanel.jsx';
import FallingNotes from './components/FallingNotes.jsx';
import PianoKeyboard, { keyboardMap } from './components/PianoKeyboard.jsx';
import SongUploader from './components/SongUploader.jsx';
import TransportDock from './components/TransportDock.jsx';
import LearnModePanel from './components/LearnModePanel.jsx';
import AccountPage from './pages/AccountPage.jsx';
import GuitarPage from './pages/GuitarPage.jsx';
import EnsemblePage from './pages/EnsemblePage.jsx';
import MarketplacePage from './pages/MarketplacePage.jsx';
import MessagesPage from './pages/MessagesPage.jsx';
import PaymentPage from './pages/PaymentPage.jsx';
import BandPage from './pages/BandPage.jsx';
import YourSongsPage from './pages/YourSongsPage.jsx';
import AdminDatabasePage from './pages/AdminDatabasePage.jsx';
import { loadFeaturedSongs, sampleSongs } from './data/sampleSongs.js';
import { pianoAudio, TONE_MODE_LABELS } from './engine/audioEngine.js';
import {
  calibrateDevice,
  getInitialPerformanceTier,
  lowerPerformanceTier,
  normalizePerformanceTier,
  raisePerformanceTier,
  refineTierFromLoading,
  savePerformanceProfile,
  visualFrameInterval,
} from './engine/devicePerformance.js';
import { buildAdaptivePianoLayout, buildLearningHandLayout } from './engine/grandPianoLayout.js';
import {
  findStartIndex,
  getPedalStateAt,
  getSongDuration,
  normalizeSong,
} from './engine/scheduler.js';
import { apiRequest, fetchProtectedFile, getAuthToken, setAuthToken } from './services/api.js';
import { parseUploadedSongFile } from './utils/songParser.js';
import { analyzeLearningSections } from './utils/learningSections.js';

const AUDIO_LOOKAHEAD_SECONDS = 0.18;
const AUDIO_SCHEDULER_INTERVAL_MS = 25;

function readRoute() {
  const redirectParams = new URLSearchParams(window.location.search);
  if (redirectParams.get('paymentStatus')) {
    const params = new URLSearchParams();
    params.set('status', redirectParams.get('paymentStatus'));
    if (redirectParams.get('productId')) params.set('productId', redirectParams.get('productId'));
    const paymentToken = redirectParams.get('subscription_id')
      || redirectParams.get('subscriptionId')
      || redirectParams.get('token')
      || redirectParams.get('ba_token');
    if (paymentToken) params.set('token', paymentToken);
    return { page: 'payment', params };
  }

  const raw = window.location.hash.replace(/^#/, '') || 'studio';
  const [page, query = ''] = raw.split('?');
  return { page: page || 'studio', params: new URLSearchParams(query) };
}

function isTypingTarget(target) {
  const tag = target?.tagName?.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable;
}

function pianoHandForEvent(event) {
  const explicit = String(event?.hand || '').toLowerCase();
  const role = String(event?.scoreRole || '').toLowerCase();
  if (explicit === 'left' || role.includes('left') || role.includes('bass')) return 'left';
  if (explicit === 'right' || role.includes('right') || role.includes('melody') || role.includes('top')) return 'right';
  const midi = Number(event?.midi);
  return Number.isFinite(midi) && midi < 60 ? 'left' : 'right';
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [user, setUser] = useState(null);
  const [songs, setSongs] = useState(() => sampleSongs.map(normalizeSong));
  const [songTitle, setSongTitle] = useState(sampleSongs[0].title);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeNotes, setActiveNotes] = useState(new Set());
  const [strikeVersions, setStrikeVersions] = useState(new Map());
  const [speed, setSpeed] = useState(1);
  const [leadTime, setLeadTime] = useState(3.4);
  const [toneMode, setToneMode] = useState('pianella');
  const [autoplayVolume, setAutoplayVolume] = useState(1);
  const [pedalDown, setPedalDown] = useState(false);
  const [showKeyNotes, setShowKeyNotes] = useState(true);
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [teachingMode, setTeachingMode] = useState('regular');
  const [preferredSectionSeconds, setPreferredSectionSeconds] = useState(15);
  const [selectedSectionIndex, setSelectedSectionIndex] = useState(0);
  const [repeatSection, setRepeatSection] = useState(true);
  const [practiceRange, setPracticeRange] = useState(null);
  const [pianoHandMode, setPianoHandMode] = useState('left');
  const [portraitDevice, setPortraitDevice] = useState(() => (
    window.innerWidth <= 1024 && window.innerHeight > window.innerWidth
  ));
  const [orientationPromptDismissed, setOrientationPromptDismissed] = useState(false);
  const [keyboardPreparationStatus, setKeyboardPreparationStatus] = useState('locked');
  const [keyboardPreparationProgress, setKeyboardPreparationProgress] = useState(0);
  const [keyboardPreparationStage, setKeyboardPreparationStage] = useState('Tap once to prepare');
  const [performanceTier, setPerformanceTier] = useState(getInitialPerformanceTier);

  const nextEventIndex = useRef(0);
  const nextPedalIndex = useRef(0);
  const activeNoteCounts = useRef(new Map());
  const manualVoices = useRef(new Map());
  const startStamp = useRef(0);
  const pauseOffset = useRef(0);
  const animationFrame = useRef(null);
  const lastVisualFrame = useRef(0);
  const audioScheduler = useRef(null);
  const visualTimers = useRef([]);
  const pedalTimers = useRef([]);
  const playbackRunId = useRef(0);
  const seekWasPlaying = useRef(false);
  const studioPlayerRef = useRef(null);
  const keyboardReadyRef = useRef(false);
  const keyboardPreparationPromise = useRef(null);
  const performanceTierRef = useRef(performanceTier);
  const calibrationRef = useRef(null);
  const runtimePerformanceRef = useRef({
    lastFrame: 0,
    windowStarted: 0,
    frameDeltas: [],
    audioEvents: 0,
    lateAudioEvents: 0,
    stableWindows: 0,
    lastChangeAt: 0,
  });

  const song = useMemo(
    () => songs.find((candidate) => candidate.title === songTitle) || songs[0],
    [songs, songTitle],
  );
  const pianoLayout = useMemo(
    () => teachingMode === 'learn'
      ? buildLearningHandLayout(song, pianoHandMode)
      : buildAdaptivePianoLayout(song),
    [song, teachingMode, pianoHandMode],
  );
  const playbackNotes = useMemo(() => (
    teachingMode === 'learn' && pianoHandMode !== 'both'
      ? song.notes.filter((event) => pianoHandForEvent(event) === pianoHandMode)
      : song.notes
  ), [song, teachingMode, pianoHandMode]);
  const teachingSong = useMemo(() => ({ ...song, notes: playbackNotes }), [song, playbackNotes]);
  const learningSections = useMemo(
    () => analyzeLearningSections(song.notes, getSongDuration(song), preferredSectionSeconds),
    [song, preferredSectionSeconds],
  );

  useEffect(() => {
    if (!window.location.hash) window.history.replaceState(null, '', '#studio');
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    // Dismissal lasts only for this app session. Remove the legacy permanent
    // preference so the recommendation can return the next time the app opens.
    window.localStorage.removeItem('polymath-orientation-prompt-dismissed');
  }, []);

  useEffect(() => {
    function checkOrientation() {
      setPortraitDevice(window.innerWidth <= 1024 && window.innerHeight > window.innerWidth);
    }
    window.addEventListener('resize', checkOrientation);
    window.addEventListener('orientationchange', checkOrientation);
    return () => {
      window.removeEventListener('resize', checkOrientation);
      window.removeEventListener('orientationchange', checkOrientation);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadFeaturedSongs()
      .then((featured) => {
        if (cancelled || !featured.length) return;
        const normalized = featured.map(normalizeSong);
        setSongs((previous) => [...normalized, ...previous.filter((item) => !normalized.some((featuredSong) => featuredSong.title === item.title))]);
        setSongTitle(normalized[0].title);
      })
      .catch((error) => console.error(error));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!getAuthToken()) return;
    apiRequest('/api/auth/me')
      .then((data) => setUser(data.user))
      .catch(() => setAuthToken(''));
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!user?.user_id) return undefined;
    apiRequest('/api/library')
      .then(async ({ purchasedSongs }) => {
        const pianoSongs = purchasedSongs.filter((item) => item.instrument === 'piano' && ['JSON', 'MIDI', 'MUSICXML'].includes(item.format));
        const loaded = await Promise.all(pianoSongs.map(async (item) => {
          const file = await fetchProtectedFile(`/api/listings/${item.id}/download`, item.filename || `${item.title}.${item.format.toLowerCase()}`);
          const parsed = await parseUploadedSongFile(file);
          return normalizeSong({ ...parsed, title: item.title, composer: item.artist || parsed.composer });
        }));
        if (cancelled || !loaded.length) return;
        setSongs((previous) => [...previous.filter((songItem) => !loaded.some((librarySong) => librarySong.title === songItem.title)), ...loaded]);
      })
      .catch((error) => console.error('Purchased piano songs could not be loaded:', error));
    return () => { cancelled = true; };
  }, [user?.user_id]);

  useEffect(() => {
    pianoAudio.setToneMode(toneMode);
  }, [toneMode]);

  function navigate(page, params = {}) {
    if (window.location.search) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.hash || ''}`);
    }
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') query.set(key, String(value));
    });
    window.location.hash = `${page}${query.size ? `?${query.toString()}` : ''}`;
  }

  function refreshActiveNotes() {
    setActiveNotes(new Set(activeNoteCounts.current.keys()));
  }

  function bumpStrike(note) {
    setStrikeVersions((previous) => {
      const next = new Map(previous);
      next.set(note, (next.get(note) || 0) + 1);
      return next;
    });
  }

  function addActiveNote(note, shouldStrike = true) {
    activeNoteCounts.current.set(note, (activeNoteCounts.current.get(note) || 0) + 1);
    if (shouldStrike) bumpStrike(note);
    refreshActiveNotes();
  }

  function removeActiveNote(note) {
    const count = activeNoteCounts.current.get(note) || 0;
    if (count <= 1) activeNoteCounts.current.delete(note);
    else activeNoteCounts.current.set(note, count - 1);
    refreshActiveNotes();
  }

  function clearActiveNotes() {
    activeNoteCounts.current.clear();
    setActiveNotes(new Set());
  }

  function clearTimers(timerRef) {
    timerRef.current.forEach((timerId) => window.clearTimeout(timerId));
    timerRef.current = [];
  }

  function clampSongTime(value) {
    return Math.max(
      0,
      Math.min(
        getSongDuration(song),
        Number(value) || 0,
      ),
    );
  }

  function focusMobilePlayer(force = false) {
    if (!force && !window.matchMedia('(max-width: 1100px)').matches) return;
    window.setTimeout(() => {
      const player = studioPlayerRef.current;
      if (!player) return;
      player.scrollIntoView({ behavior: 'smooth', block: 'start' });
      player.focus({ preventScroll: true });
    }, 120);
  }

  function setSustainPedal(nextState) {
    const down = Boolean(nextState);
    pianoAudio.setSustainPedal(down);
    setPedalDown(down);
  }

  function applyPerformanceTier(nextTier, measurements = {}, preserveSampleSet = keyboardReadyRef.current) {
    const normalized = normalizePerformanceTier(nextTier, performanceTierRef.current);
    performanceTierRef.current = normalized;
    setPerformanceTier(normalized);
    pianoAudio.setPerformanceTier(normalized, { preserveSampleSet });
    savePerformanceProfile(normalized, measurements);
    window.__POLYMATH_PERFORMANCE__ = {
      tier: normalized,
      ...measurements,
      audio: pianoAudio.getDiagnostics(),
    };
    return normalized;
  }

  async function prepareKeyboard() {
    if (keyboardReadyRef.current) return true;
    if (keyboardPreparationPromise.current) return keyboardPreparationPromise.current;

    // Create/resume Web Audio directly inside the tap event so iOS grants audio access.
    pianoAudio.ensure();
    setKeyboardPreparationStatus('calibrating');
    setKeyboardPreparationStage('Checking this device');
    setKeyboardPreparationProgress(3);

    const task = calibrateDevice()
      .then(async (calibration) => {
        calibrationRef.current = calibration;
        const calibratedTier = applyPerformanceTier(calibration.tier, { calibration }, false);
        setKeyboardPreparationStatus('loading');
        setKeyboardPreparationStage('Loading ' + calibratedTier + ' piano');
        setKeyboardPreparationProgress(8);
        const loading = await pianoAudio.prepareKeyboard(({ percent }) => {
          const mappedProgress = 8 + (Math.max(0, Math.min(100, Number(percent) || 0)) * 0.92);
          setKeyboardPreparationProgress(Math.round(mappedProgress));
        });
        const refinedTier = refineTierFromLoading(calibratedTier, loading);
        applyPerformanceTier(refinedTier, { calibration, loading }, true);
        return { loading, refinedTier };
      })
      .then(() => {
        keyboardReadyRef.current = true;
        setKeyboardPreparationProgress(100);
        setKeyboardPreparationStatus('ready');
        setKeyboardPreparationStage(performanceTierRef.current + ' mode ready');
        return true;
      })
      .catch((error) => {
        console.error('Piano preparation failed:', error);
        setKeyboardPreparationStatus('error');
        return false;
      })
      .finally(() => {
        keyboardPreparationPromise.current = null;
      });

    keyboardPreparationPromise.current = task;
    return task;
  }

  function pressNote(note, velocity = 0.85, duration = null, source = 'manual', playbackOptions = {}) {
    if (!keyboardReadyRef.current) return null;
    const voice = pianoAudio.play(note, velocity, duration, {
      source,
      retriggerSameNote: source === 'manual',
      retriggerReleaseSeconds: playbackOptions.retriggerReleaseSeconds,
      releaseSeconds: playbackOptions.releaseSeconds,
    });
    if (!voice) return null;
    addActiveNote(note, true);
    if (source === 'manual' && duration === null) manualVoices.current.set(note, voice);
    return voice;
  }

  function releaseNote(note) {
    const manualVoice = manualVoices.current.get(note);
    if (manualVoice && typeof pianoAudio.releaseVoice === 'function') {
      pianoAudio.releaseVoice(manualVoice);
      manualVoices.current.delete(note);
    } else {
      pianoAudio.release(note);
    }
    removeActiveNote(note);
  }

  function stopPlaybackClocks() {
    if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
    if (audioScheduler.current) window.clearInterval(audioScheduler.current);
    animationFrame.current = null;
    audioScheduler.current = null;
    clearTimers(visualTimers);
    clearTimers(pedalTimers);
  }

  function silencePlayback(position = currentTime) {
    playbackRunId.current += 1;
    stopPlaybackClocks();
    pauseOffset.current = clampSongTime(position);
    manualVoices.current.clear();
    pianoAudio.stopAll({ releaseSeconds: 0.028 });
    setPedalDown(false);
    clearActiveNotes();
  }

  function resetPlaybackState() {
    silencePlayback(0);
    nextEventIndex.current = 0;
    nextPedalIndex.current = 0;
    pauseOffset.current = 0;
  }

  function stopPlayback() {
    setIsPlaying(false);
    setCurrentTime(0);
    resetPlaybackState();
  }

  async function startPlaybackAt(position, speedOverride = speed) {
    if (!keyboardReadyRef.current && !(await prepareKeyboard())) return;
    const target = clampSongTime(position);
    const playbackSpeed = Math.max(0.2, Math.min(2, Number(speedOverride) || 1));

    const requestedRunId = playbackRunId.current + 1;
    playbackRunId.current = requestedRunId;
    pianoAudio.setToneMode(toneMode);
    await pianoAudio.preloadSongNotes(song);
    if (playbackRunId.current !== requestedRunId) return;

    pauseOffset.current = target;
    nextEventIndex.current = findStartIndex(playbackNotes, Math.max(0, target - 0.0005));
    nextPedalIndex.current = findStartIndex(song.pedals || [], target + 0.0001);
    setSustainPedal(getPedalStateAt(song.pedals, target));
    startStamp.current = performance.now() - (target * 1000) / playbackSpeed;
    setCurrentTime(target);
    setIsPlaying(true);
    setPlaybackEpoch((value) => value + 1);
  }

  async function togglePlayPause() {
    if (isPlaying) {
      silencePlayback(currentTime);
      setIsPlaying(false);
      return;
    }

    await startPlaybackAt(pauseOffset.current);
  }

  function beginSeek() {
    seekWasPlaying.current = isPlaying;
    silencePlayback(currentTime);
    setIsPlaying(false);
  }

  function previewSeek(position) {
    const target = clampSongTime(position);
    pauseOffset.current = target;
    setCurrentTime(target);
  }

  async function commitSeek(position) {
    const target = clampSongTime(position);
    const shouldResume = seekWasPlaying.current;
    seekWasPlaying.current = false;
    previewSeek(target);

    if (shouldResume) {
      await startPlaybackAt(target);
    }
  }

  async function jumpBy(seconds) {
    const shouldResume = isPlaying;
    const target = clampSongTime(currentTime + seconds);
    silencePlayback(currentTime);
    setIsPlaying(false);
    previewSeek(target);

    if (shouldResume) {
      await startPlaybackAt(target);
    }
  }

  async function handleSpeedChange(nextSpeed) {
    const rawValue = typeof nextSpeed === 'function' ? nextSpeed(speed) : nextSpeed;
    const safeSpeed = Math.max(0.2, Math.min(2, Number(rawValue) || 1));
    const shouldResume = isPlaying;
    const position = currentTime;

    if (shouldResume) {
      silencePlayback(position);
      setIsPlaying(false);
    }

    setSpeed(safeSpeed);

    if (shouldResume) {
      await startPlaybackAt(position, safeSpeed);
    }
  }

  function handleSongChange(title) {
    stopPlayback();
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setSongTitle(title);
  }

  function handleUpload(uploadedSong) {
    const normalized = normalizeSong(uploadedSong);
    stopPlayback();
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setSongs((previous) => [normalized, ...previous.filter((candidate) => candidate.title !== normalized.title)]);
    setSongTitle(normalized.title);
    focusMobilePlayer();
  }

  function getSongTimeFromPerformanceClock(now = performance.now()) {
    return ((now - startStamp.current) / 1000) * speed;
  }

  function scheduleVisualStrike(event, delaySeconds, visualDuration, runId) {
    const startTimer = window.setTimeout(() => {
      if (playbackRunId.current !== runId) return;
      addActiveNote(event.note, true);
      const stopTimer = window.setTimeout(() => {
        if (playbackRunId.current === runId) removeActiveNote(event.note);
      }, visualDuration * 1000 + 70);
      visualTimers.current.push(stopTimer);
    }, delaySeconds * 1000);
    visualTimers.current.push(startTimer);
  }

  useEffect(() => {
    function down(event) {
      if (event.repeat || isTypingTarget(event.target)) return;
      if (event.code === 'Space') {
        event.preventDefault();
        setSustainPedal(true);
        return;
      }
      const note = keyboardMap[event.key.toLowerCase()];
      if (!note) return;
      event.preventDefault();
      pressNote(note, 0.85, null, 'manual');
    }

    function up(event) {
      if (isTypingTarget(event.target)) return;
      if (event.code === 'Space') {
        event.preventDefault();
        setSustainPedal(false);
        return;
      }
      const note = keyboardMap[event.key.toLowerCase()];
      if (!note) return;
      event.preventDefault();
      releaseNote(note);
    }

    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [toneMode]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const duration = teachingMode === 'learn' && practiceRange
      ? Math.min(getSongDuration(song), practiceRange.end)
      : getSongDuration(song);
    const runId = playbackRunId.current;
    runtimePerformanceRef.current = {
      lastFrame: 0,
      windowStarted: performance.now(),
      frameDeltas: [],
      audioEvents: 0,
      lateAudioEvents: 0,
      stableWindows: 0,
      lastChangeAt: 0,
    };

    function evaluateRuntimePerformance(now) {
      const monitor = runtimePerformanceRef.current;
      if (now - monitor.windowStarted < 5000 || document.visibilityState === 'hidden') return;

      const sortedFrames = [...monitor.frameDeltas].sort((a, b) => a - b);
      const p95FrameMs = sortedFrames.length
        ? sortedFrames[Math.min(sortedFrames.length - 1, Math.floor(sortedFrames.length * 0.95))]
        : 100;
      const delayedFrameRatio = sortedFrames.length
        ? sortedFrames.filter((value) => value > 34).length / sortedFrames.length
        : 1;
      const lateAudioRatio = monitor.audioEvents
        ? monitor.lateAudioEvents / monitor.audioEvents
        : 0;
      const struggling = p95FrameMs > 42
        || delayedFrameRatio > 0.18
        || lateAudioRatio > 0.12;
      const strong = p95FrameMs <= 25
        && delayedFrameRatio < 0.05
        && lateAudioRatio < 0.03;
      const runtime = {
        p95FrameMs: Math.round(p95FrameMs * 10) / 10,
        delayedFrameRatio: Math.round(delayedFrameRatio * 1000) / 1000,
        lateAudioRatio: Math.round(lateAudioRatio * 1000) / 1000,
        observedFrames: sortedFrames.length,
        observedAudioEvents: monitor.audioEvents,
      };

      if (struggling && performanceTierRef.current !== 'lite' && now - monitor.lastChangeAt > 8000) {
        const nextTier = lowerPerformanceTier(performanceTierRef.current);
        applyPerformanceTier(nextTier, {
          calibration: calibrationRef.current,
          runtime,
          reason: 'runtime-slowdown',
        }, true);
        setKeyboardPreparationStage(nextTier + ' mode ready');
        monitor.stableWindows = 0;
        monitor.lastChangeAt = now;
      } else if (strong) {
        monitor.stableWindows += 1;
        if (
          monitor.stableWindows >= 3
          && performanceTierRef.current !== 'full'
          && now - monitor.lastChangeAt > 12000
        ) {
          const nextTier = raisePerformanceTier(performanceTierRef.current);
          applyPerformanceTier(nextTier, {
            calibration: calibrationRef.current,
            runtime,
            reason: 'runtime-stable',
          }, true);
          setKeyboardPreparationStage(nextTier + ' mode ready');
          monitor.stableWindows = 0;
          monitor.lastChangeAt = now;
        }
      } else {
        monitor.stableWindows = 0;
      }

      monitor.windowStarted = now;
      monitor.frameDeltas = [];
      monitor.audioEvents = 0;
      monitor.lateAudioEvents = 0;
    }

    function scheduleDueAudioEvents() {
      if (playbackRunId.current !== runId) return;
      const songNow = getSongTimeFromPerformanceClock();
      const audioNow = pianoAudio.getCurrentTime();
      const lookAheadSongTime = songNow + AUDIO_LOOKAHEAD_SECONDS * speed;

      while (nextPedalIndex.current < (song.pedals?.length || 0) && song.pedals[nextPedalIndex.current].time <= lookAheadSongTime && song.pedals[nextPedalIndex.current].time < duration) {
        const event = song.pedals[nextPedalIndex.current];
        nextPedalIndex.current += 1;
        const delaySeconds = Math.max(0, (event.time - songNow) / speed);
        const timer = window.setTimeout(() => {
          if (playbackRunId.current === runId) setSustainPedal(event.down);
        }, delaySeconds * 1000);
        pedalTimers.current.push(timer);
      }

      while (nextEventIndex.current < playbackNotes.length && playbackNotes[nextEventIndex.current].time <= lookAheadSongTime && playbackNotes[nextEventIndex.current].time < duration) {
        const event = playbackNotes[nextEventIndex.current];
        nextEventIndex.current += 1;
        const schedulingLatenessMs = Math.max(0, ((songNow - event.time) / speed) * 1000);
        runtimePerformanceRef.current.audioEvents += 1;
        if (schedulingLatenessMs > 45) {
          runtimePerformanceRef.current.lateAudioEvents += 1;
        }
        const delaySeconds = Math.max(0, (event.time - songNow) / speed);
        const audioStartAt = audioNow + delaySeconds;
        const noteDuration = Math.max(0.035, (event.audioDuration ?? event.duration) / speed);
        const visualDuration = Math.max(0.035, (event.visualDuration ?? event.duration) / speed);
        const eventVelocity = Math.max(0.02, Math.min(1.15, Number(event.velocity ?? 0.7) * autoplayVolume));

        pianoAudio.playAt(event.note, eventVelocity, noteDuration, audioStartAt, {
          source: 'autoplay',
          retriggerSameNote: false,
          releaseSeconds: event.releaseSeconds,
        });
        scheduleVisualStrike(event, delaySeconds, visualDuration, runId);
      }

      if (songNow >= duration) {
        if (teachingMode === 'learn' && practiceRange && repeatSection) {
          const restartAt = practiceRange.start;
          silencePlayback(restartAt);
          setIsPlaying(false);
          setCurrentTime(restartAt);
          window.setTimeout(() => startPlaybackAt(restartAt), 35);
        } else {
          silencePlayback(practiceRange?.start || 0);
          setIsPlaying(false);
          setCurrentTime(duration);
        }
      }
    }

    function tick(now) {
      if (playbackRunId.current !== runId) return;
      const monitor = runtimePerformanceRef.current;
      if (monitor.lastFrame) {
        const frameDelta = now - monitor.lastFrame;
        if (frameDelta > 0 && frameDelta < 250) monitor.frameDeltas.push(frameDelta);
      }
      monitor.lastFrame = now;
      evaluateRuntimePerformance(now);
      const nextTime = getSongTimeFromPerformanceClock(now);
      if (
        now - lastVisualFrame.current >= visualFrameInterval(performanceTierRef.current)
        || nextTime >= duration
      ) {
        lastVisualFrame.current = now;
        setCurrentTime(nextTime);
      }
      if (nextTime >= duration) {
        if (!(teachingMode === 'learn' && practiceRange)) stopPlayback();
        return;
      }
      animationFrame.current = requestAnimationFrame(tick);
    }

    scheduleDueAudioEvents();
    audioScheduler.current = window.setInterval(scheduleDueAudioEvents, AUDIO_SCHEDULER_INTERVAL_MS);
    animationFrame.current = requestAnimationFrame(tick);

    return () => {
      if (audioScheduler.current) window.clearInterval(audioScheduler.current);
      if (animationFrame.current) cancelAnimationFrame(animationFrame.current);
      audioScheduler.current = null;
      animationFrame.current = null;
    };
  }, [isPlaying, song, playbackNotes, speed, autoplayVolume, toneMode, playbackEpoch, teachingMode, practiceRange, repeatSection]);

  function selectLearningSection(index) {
    const safeIndex = Math.max(0, Math.min(learningSections.length - 1, index));
    const section = learningSections[safeIndex];
    if (!section) return;
    silencePlayback(section.start);
    setIsPlaying(false);
    setSelectedSectionIndex(safeIndex);
    setPracticeRange(section);
    setCurrentTime(section.start);
  }

  async function practiseLearningSection(section) {
    const index = learningSections.findIndex((candidate) => candidate.id === section.id);
    selectLearningSection(index < 0 ? 0 : index);
    setPracticeRange(section);
    await startPlaybackAt(section.start);
  }

  const paymentProductId = route.params.get('productId') || 'polymath-chill-monthly';
  const messageUserId = route.params.get('userId');
  const messageName = route.params.get('name') || 'Composer';
  const content = (() => {
    if (user?.mustChangePassword && route.page !== 'account') {
      return <AccountPage user={user} setUser={setUser} onNavigate={navigate} />;
    }
    if (route.page === 'guitar') return <GuitarPage user={user} setUser={setUser} onNavigate={navigate} />;
    if (route.page === 'ensemble') return <EnsemblePage user={user} setUser={setUser} onNavigate={navigate} />;
    if (route.page === 'band') {
      if (!user?.admin && !user?.access?.band) {
        return (
          <PaymentPage
            user={user}
            setUser={setUser}
            productId="polymath-musician-monthly"
            onNavigate={navigate}
          />
        );
      }
      return <BandPage user={user} setUser={setUser} onNavigate={navigate} />;
    }
    if (route.page === 'published-songs') return <MarketplacePage user={user} setUser={setUser} onNavigate={navigate} />;
    if (route.page === 'your-songs') return <YourSongsPage user={user} onNavigate={navigate} />;
    if (route.page === 'admin-database') return <AdminDatabasePage user={user} onNavigate={navigate} />;
    if (route.page === 'messages') return <MessagesPage user={user} initialUser={messageUserId ? { user_id: messageUserId, name: messageName } : null} onNavigate={navigate} />;
    if (route.page === 'account') return (
      <AccountPage
        user={user}
        setUser={setUser}
        onNavigate={navigate}
        returnPage={route.params.get('next')}
        returnProductId={route.params.get('productId')}
      />
    );
    if (route.page === 'payment') {
      return (
        <PaymentPage
          user={user}
          setUser={setUser}
          productId={paymentProductId}
          paymentStatus={route.params.get('status')}
          paymentToken={route.params.get('token')}
          onNavigate={navigate}
        />
      );
    }

    return (
      <section className="studio-page">
        <LearnModePanel
          mode={teachingMode}
          locked={!user?.admin && !user?.access?.learn}
          onUpgrade={() => navigate('payment', { productId: 'polymath-musician-monthly' })}
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
          onPreferredSecondsChange={(value) => setPreferredSectionSeconds(Math.max(5, Math.min(60, value || 15)))}
        />

        {teachingMode === 'learn' && (
          <section className="piano-hand-selector" aria-label="Choose piano hands to practise">
            <div>
              <p className="eyebrow">Hand progression</p>
              <h3>Build each hand, then combine them.</h3>
            </div>
            <div className="piano-hand-options" role="group" aria-label="Piano hand selection">
              {[
                ['left', '1. Left hand', 'Bass and accompaniment'],
                ['right', '2. Right hand', 'Melody and upper voice'],
                ['both', '3. Both hands', 'Standard double layer'],
              ].map(([value, label, description]) => (
                <button
                  type="button"
                  key={value}
                  className={pianoHandMode === value ? 'active' : ''}
                  onClick={() => {
                    silencePlayback(currentTime);
                    setIsPlaying(false);
                    setPianoHandMode(value);
                  }}
                >
                  <strong>{label}</strong>
                  <small>{description}</small>
                </button>
              ))}
            </div>
          </section>
        )}

        <section className="studio-grid">
          <div className='mobile-flow-guide mobile-source-guide'>
            <span>1</span>
            <div>
              <strong>Choose your music</strong>
              <small>Pick a song or upload a ready sheet, PDF, MP3, or video.</small>
            </div>
          </div>

          <ControlPanel
            song={song}
            songs={songs}
            onSongChange={handleSongChange}
            onPlayNow={() => focusMobilePlayer(true)}
          />

          <div className='mobile-flow-guide mobile-player-guide'>
            <span>2</span>
            <div>
              <strong>Play and learn</strong>
              <small>Follow the falling notes, then use the keyboard and controls below.</small>
            </div>
          </div>
          <div ref={studioPlayerRef} className="visual-stack" tabIndex="-1">
            <FallingNotes song={teachingSong} layout={pianoLayout} currentTime={currentTime} isPlaying={isPlaying} leadTime={leadTime} activeNotes={activeNotes} performanceTier={performanceTier} />
            <div className="piano-scroll-wrap">
              <PianoKeyboard
                layout={pianoLayout}
                activeNotes={activeNotes}
                strikeVersions={strikeVersions}
                showKeyNotes={showKeyNotes}
                onPress={(note) => pressNote(note, 0.85, null, 'manual')}
                onRelease={releaseNote}
                preparationStatus={keyboardPreparationStatus}
                preparationProgress={keyboardPreparationProgress}
                preparationStage={keyboardPreparationStage}
                performanceTier={performanceTier}
                onPrepare={prepareKeyboard}
              />
            </div>
            <TransportDock
              song={song}
              isPlaying={isPlaying}
              onPlayPause={togglePlayPause}
              onStop={stopPlayback}
              currentTime={currentTime}
              speed={speed}
              setSpeed={handleSpeedChange}
              pedalDown={pedalDown}
              onPedalChange={setSustainPedal}
              onSeekStart={beginSeek}
              onSeekPreview={previewSeek}
              onSeekCommit={commitSeek}
              onRewind={() => jumpBy(-10)}
              onForward={() => jumpBy(10)}
              showKeyNotes={showKeyNotes}
              onShowKeyNotesChange={setShowKeyNotes}
              minSpeed={0.2}
            />
            <details className="lesson-options player-settings">
              <summary>Settings</summary>
              <div className="lesson-options-content">
                <label className="field">
                  Piano sound
                  <select value={toneMode} onChange={(event) => setToneMode(event.target.value)}>
                    <option value="pianella">{TONE_MODE_LABELS.pianella}</option>
                    <option value="grand">{TONE_MODE_LABELS.grand}</option>
                  </select>
                </label>
                <label className="field">
                  Autoplay volume: {Math.round((autoplayVolume ?? 1) * 100)}%
                  <input type="range" min="0.25" max="1.25" step="0.05" value={autoplayVolume ?? 1} onChange={(event) => setAutoplayVolume(Number(event.target.value))} />
                </label>
                <label className="field">
                  Notes appear {leadTime.toFixed(1)}s early
                  <input type="range" min="1.4" max="5.5" step="0.1" value={leadTime} onChange={(event) => setLeadTime(Number(event.target.value))} />
                </label>
              </div>
            </details>
          </div>

          <SongUploader onUpload={handleUpload} user={user} setUser={setUser} onNavigate={navigate} />
        </section>
      </section>
    );
  })();

  return (
    <div className="app-root" data-performance-tier={performanceTier}>
      {portraitDevice && !orientationPromptDismissed && (
        <aside className="orientation-recommendation" role="dialog" aria-label="Landscape orientation recommendation">
          <div className="orientation-phone-icon" aria-hidden="true"><span /></div>
          <div>
            <strong>Sideways is recommended for learning</strong>
            <p>Turn your phone or tablet sideways so the piano, falling notes, guitar strings, and timing lanes have more width than height.</p>
          </div>
          <button
            type="button"
            className="ghost"
            onClick={() => setOrientationPromptDismissed(true)}
          >
            Ignore
          </button>
        </aside>
      )}
      <div className="top-shell">
        <AppNav route={route.page} onNavigate={navigate} user={user} />
        <HeaderActions user={user} onNavigate={navigate} route={route.page} />
      </div>
      <main className="app-shell">{content}</main>
    </div>
  );
}
