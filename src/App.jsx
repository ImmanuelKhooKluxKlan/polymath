import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import AppNav from './components/AppNav.jsx';
import HeaderActions from './components/HeaderActions.jsx';
import ControlPanel from './components/ControlPanel.jsx';
import FallingNotes from './components/FallingNotes.jsx';
import PianoKeyboard, { keyboardMap } from './components/PianoKeyboard.jsx';
import SongUploader from './components/SongUploader.jsx';
import TransportDock from './components/TransportDock.jsx';
import PianoLearnJourney from './components/PianoLearnJourney.jsx';
import PianoTeacherStudio from './components/PianoTeacherStudio.jsx';
import SupportAssistant from './components/SupportAssistant.jsx';
import { loadFeaturedSongs, sampleSongs } from './data/sampleSongs.js';
import { pianoAudio, TONE_MODE_LABELS } from './engine/audioEngine.js';
import {
  capTierForDevice,
  calibrateDevice,
  detectDeviceClass,
  getInitialPerformanceTier,
  lowerPerformanceTier,
  normalizePerformanceTier,
  raisePerformanceTier,
  readSavedPerformanceProfile,
  refineTierFromLoading,
  savePerformanceProfile,
  visualFrameInterval,
} from './engine/devicePerformance.js';
import { buildAdaptivePianoLayout, buildLearningHandLayout } from './engine/grandPianoLayout.js';
import {
  analyzePracticeAttempt,
  buildLearningArrangement,
  learningLevelById,
  learningSessionById,
  readLearningProgress,
  recordLearningAttempt,
  writeLearningProgress,
} from './engine/learningCoach.js';
import { midiToNote } from './engine/noteMath.js';
import {
  buildTeacherHandTargets,
  prepareTeacherHandTimeline,
  TEACHER_PROFILES,
} from './engine/teacherHands.js';
import {
  findStartIndex,
  getPedalStateAt,
  getSongDuration,
  normalizeSong,
} from './engine/scheduler.js';
import { apiAssetUrl, apiRequest, fetchProtectedFile, getAuthToken, setAuthToken } from './services/api.js';
import { parseUploadedSongFile } from './utils/songParser.js';
import { analyzeLearningSections } from './utils/learningSections.js';

const AUDIO_LOOKAHEAD_SECONDS = 0.18;
const AUDIO_SCHEDULER_INTERVAL_MS = 25;

const AccountPage = lazy(() => import('./pages/AccountPage.jsx'));
const GuitarPage = lazy(() => import('./pages/GuitarPage.jsx'));
const EnsemblePage = lazy(() => import('./pages/EnsemblePage.jsx'));
const MarketplacePage = lazy(() => import('./pages/MarketplacePage.jsx'));
const TeacherMarketplacePage = lazy(() => import('./pages/TeacherMarketplacePage.jsx'));
const MessagesPage = lazy(() => import('./pages/MessagesPage.jsx'));
const CommunityPage = lazy(() => import('./pages/CommunityPage.jsx'));
const PaymentPage = lazy(() => import('./pages/PaymentPage.jsx'));
const BandPage = lazy(() => import('./pages/BandPage.jsx'));
const YourSongsPage = lazy(() => import('./pages/YourSongsPage.jsx'));
const AdminDatabasePage = lazy(() => import('./pages/AdminDatabasePage.jsx'));
const ModelLabPage = lazy(() => import('./pages/ModelLabPage.jsx'));

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

function manualVoiceKey(note, interaction = {}) {
  return interaction.pointerId === undefined
    ? `key:${note}`
    : `pointer:${interaction.pointerId}`;
}

function songLibraryId(song) {
  return song?.libraryId
    || (song?.personalSongId ? `personal:${song.personalSongId}` : '')
    || `${song?.libraryType || 'song'}:${song?.title || 'Untitled Song'}:${song?.composer || song?.artist || 'Unknown'}`;
}

function materializeTeacherProfile(profile, base = {}) {
  const armTone = profile.armTone || base.armTone || 'light';
  const image = profile.imagePath ? apiAssetUrl(profile.imagePath) : base.image;
  return {
    ...base,
    ...profile,
    image,
    stageImage: image || base.stageImage,
    portraitPosition: base.portraitPosition || '50% 18%',
    armImage: base.armImage || (armTone === 'dark'
      ? '/teachers/arm-dark-full-v1.webp'
      : '/teachers/arm-light-full-v1.webp'),
    handCameraImage: base.handCameraImage || (armTone === 'dark'
      ? '/teachers/pianist-hands-overhead-dark-v1.webp'
      : '/teachers/pianist-hands-overhead-v1.webp'),
    pressedHandCameraImage: base.pressedHandCameraImage || (armTone === 'dark'
      ? '/teachers/pianist-hands-pressed-dark-v2.webp'
      : '/teachers/pianist-hands-pressed-v2.webp'),
    modelUrl: profile.modelPath ? apiAssetUrl(profile.modelPath) : base.modelUrl || '',
    minimumAge: Math.max(0, Number(profile.minimumAge || 0)),
    requiresAdultConfirmation: Number(profile.minimumAge || 0) >= 18
      || Boolean(profile.requiresAdultConfirmation),
    adultCompanionEnabled: Boolean(
      profile.adultCompanionEnabled
        ?? base.adultCompanionEnabled
        ?? profile.requiresAdultConfirmation,
    ),
    pricePer30MinutesMcoins: profile.pricePer30MinutesMcoins ?? null,
    effectivePricePer30MinutesMcoins: profile.effectivePricePer30MinutesMcoins ?? null,
    look: base.look || 'custom',
  };
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [user, setUser] = useState(null);
  const [songs, setSongs] = useState(() => sampleSongs.map(normalizeSong));
  const [songSelectionId, setSongSelectionId] = useState(() => songLibraryId(sampleSongs[0]));
  const [personalSongs, setPersonalSongs] = useState([]);
  const [loadingPersonalSongId, setLoadingPersonalSongId] = useState('');
  const [personalSongStatus, setPersonalSongStatus] = useState('');
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
  const [openMusicChooser, setOpenMusicChooser] = useState(null);
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [teachingMode, setTeachingMode] = useState('regular');
  const [learningLevel, setLearningLevel] = useState(() => learningLevelById(window.localStorage.getItem('polymath-learning-level')).id);
  const [learningSession, setLearningSession] = useState(() => learningSessionById(window.localStorage.getItem('polymath-learning-session')).id);
  const [learningAttemptStatus, setLearningAttemptStatus] = useState('idle');
  const [learningReport, setLearningReport] = useState(null);
  const [learningProgress, setLearningProgress] = useState(() => readLearningProgress(window.localStorage, 'guest'));
  const [midiInput, setMidiInput] = useState(() => ({
    supported: typeof navigator !== 'undefined' && typeof navigator.requestMIDIAccess === 'function',
    status: 'idle',
    name: '',
    error: '',
  }));
  const [preferredSectionSeconds, setPreferredSectionSeconds] = useState(() => learningSessionById(window.localStorage.getItem('polymath-learning-session')).partSeconds);
  const [selectedSectionIndex, setSelectedSectionIndex] = useState(0);
  const [repeatSection, setRepeatSection] = useState(true);
  const [practiceRange, setPracticeRange] = useState(null);
  const [pianoHandMode, setPianoHandMode] = useState('both');
  const [pianoTeacherId, setPianoTeacherId] = useState(() => {
    const savedTeacher = window.localStorage.getItem('polymath-piano-teacher-v2');
    const adultConfirmed = window.localStorage.getItem('polymath-teacher-adult-confirmed') === 'true';
    if (savedTeacher === 'padme') return adultConfirmed ? 'nova' : 'aria';
    if (!savedTeacher || (savedTeacher === 'nova' && !adultConfirmed)) return 'aria';
    return savedTeacher;
  });
  const [remoteTeacherDirectory, setRemoteTeacherDirectory] = useState(null);
  const [teacherHandsEnabled, setTeacherHandsEnabled] = useState(() => window.localStorage.getItem('polymath-teacher-hands-v2') !== 'false');
  const [teacherDemonstration, setTeacherDemonstration] = useState(null);
  const [portraitDevice, setPortraitDevice] = useState(() => (
    window.innerWidth <= 1024 && window.innerHeight > window.innerWidth
  ));
  const [orientationPromptDismissed, setOrientationPromptDismissed] = useState(false);
  const [keyboardPreparationStatus, setKeyboardPreparationStatus] = useState('locked');
  const [keyboardPreparationProgress, setKeyboardPreparationProgress] = useState(0);
  const [keyboardPreparationStage, setKeyboardPreparationStage] = useState('Tap once to prepare');
  const [performanceTier, setPerformanceTier] = useState(getInitialPerformanceTier);
  const [deviceClass, setDeviceClass] = useState(detectDeviceClass);

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
  const playbackSpeedRef = useRef(speed);
  const songDurationRef = useRef(0);
  const practicePlaybackModeRef = useRef('listen');
  const learningCaptureRef = useRef({ status: 'idle', notes: [], activeNotes: new Map(), pedals: [] });
  const midiAccessRef = useRef(null);
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
    () => songs.find((candidate) => songLibraryId(candidate) === songSelectionId) || songs[0],
    [songs, songSelectionId],
  );
  playbackSpeedRef.current = speed;
  songDurationRef.current = getSongDuration(song);
  const learningArrangement = useMemo(
    () => teachingMode === 'learn' ? buildLearningArrangement(song.notes, learningLevel) : song.notes,
    [song, teachingMode, learningLevel],
  );
  const pianoLayout = useMemo(
    () => teachingMode === 'learn'
      ? buildLearningHandLayout({ ...song, notes: learningArrangement }, pianoHandMode)
      : buildAdaptivePianoLayout(song),
    [song, learningArrangement, teachingMode, pianoHandMode],
  );
  const playbackNotes = useMemo(() => (
    teachingMode === 'learn' && pianoHandMode !== 'both'
      ? learningArrangement.filter((event) => pianoHandForEvent(event) === pianoHandMode)
      : learningArrangement
  ), [learningArrangement, teachingMode, pianoHandMode]);
  const teachingSong = useMemo(() => ({ ...song, notes: playbackNotes }), [song, playbackNotes]);
  const teacherProfiles = useMemo(() => {
    const baseById = new Map(TEACHER_PROFILES.map((profile) => [profile.id, profile]));
    if (!remoteTeacherDirectory) return TEACHER_PROFILES;
    if (remoteTeacherDirectory.authoritative) {
      return remoteTeacherDirectory.profiles.map((profile) => (
        materializeTeacherProfile(profile, baseById.get(profile.id) || {})
      ));
    }
    return [
      ...TEACHER_PROFILES,
      ...remoteTeacherDirectory.profiles.map((profile) => materializeTeacherProfile(profile)),
    ];
  }, [remoteTeacherDirectory]);
  const pianoTeacher = useMemo(
    () => teacherProfiles.find((profile) => profile.id === pianoTeacherId)
      || teacherProfiles.find((profile) => profile.id === 'aria')
      || teacherProfiles[0],
    [pianoTeacherId, teacherProfiles],
  );
  const teacherHandTimeline = useMemo(
    () => prepareTeacherHandTimeline(teachingSong.notes),
    [teachingSong],
  );
  const teacherHandTargets = useMemo(
    () => buildTeacherHandTargets(teacherHandTimeline, currentTime, { handMode: pianoHandMode }),
    [teacherHandTimeline, currentTime, pianoHandMode],
  );
  const learningSections = useMemo(
    () => analyzeLearningSections(learningArrangement, getSongDuration(song), preferredSectionSeconds),
    [song, learningArrangement, preferredSectionSeconds],
  );
  const activeLearningRange = useMemo(() => {
    if (learningSession === 'full') {
      const duration = getSongDuration(song);
      return { id: 'full-song', name: 'Full song', start: 0, end: duration, duration };
    }
    return learningSections[selectedSectionIndex] || learningSections[0] || null;
  }, [learningSession, learningSections, selectedSectionIndex, song]);

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
    let cancelled = false;
    const loadVirtualTeachers = () => {
      apiRequest('/api/virtual-teachers')
        .then((data) => {
          if (cancelled) return;
          const authoritative = Number(data.catalogVersion || 0) >= 2 && Array.isArray(data.catalog);
          setRemoteTeacherDirectory({
            authoritative,
            profiles: authoritative ? data.catalog : (Array.isArray(data.characters) ? data.characters : []),
          });
        })
        .catch((error) => console.error('Custom virtual teachers could not be loaded:', error));
    };
    loadVirtualTeachers();
    window.addEventListener('polymath:virtual-teachers-changed', loadVirtualTeachers);
    return () => {
      cancelled = true;
      window.removeEventListener('polymath:virtual-teachers-changed', loadVirtualTeachers);
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem('polymath-piano-teacher-v2', pianoTeacher.id);
  }, [pianoTeacher.id]);

  useEffect(() => {
    window.localStorage.setItem('polymath-teacher-hands-v2', String(teacherHandsEnabled));
  }, [teacherHandsEnabled]);

  useEffect(() => {
    const learnerId = user?.user_id || 'guest';
    setLearningProgress(readLearningProgress(window.localStorage, learnerId));
  }, [user?.user_id]);

  useEffect(() => {
    if (selectedSectionIndex < learningSections.length) return;
    setSelectedSectionIndex(Math.max(0, learningSections.length - 1));
  }, [learningSections.length, selectedSectionIndex]);

  useEffect(() => () => {
    const access = midiAccessRef.current;
    if (!access) return;
    access.onstatechange = null;
    access.inputs.forEach((input) => { input.onmidimessage = null; });
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
        setSongSelectionId(songLibraryId(normalized[0]));
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
    if (!user?.user_id) {
      setPersonalSongs([]);
      return undefined;
    }
    apiRequest('/api/library')
      .then(async ({ personalSongs: cloudSongs = [], purchasedSongs }) => {
        if (!cancelled) setPersonalSongs(cloudSongs);
        const pianoSongs = purchasedSongs.filter((item) => item.instrument === 'piano' && ['JSON', 'MIDI', 'MUSICXML'].includes(item.format));
        const loaded = await Promise.all(pianoSongs.map(async (item) => {
          const file = await fetchProtectedFile(`/api/listings/${item.id}/download`, item.filename || `${item.title}.${item.format.toLowerCase()}`);
          const parsed = await parseUploadedSongFile(file);
          return normalizeSong({
            ...parsed,
            title: item.title,
            composer: item.artist || parsed.composer,
            libraryId: `purchase:${item.id}`,
            libraryType: 'purchased',
          });
        }));
        if (cancelled || !loaded.length) return;
        setSongs((previous) => [
          ...previous.filter((songItem) => !loaded.some((librarySong) => songLibraryId(librarySong) === songLibraryId(songItem))),
          ...loaded,
        ]);
      })
      .catch((error) => console.error('Purchased piano songs could not be loaded:', error));
    return () => { cancelled = true; };
  }, [user?.user_id]);

  useEffect(() => {
    pianoAudio.setToneMode(toneMode);
  }, [toneMode]);

  useEffect(() => {
    if (route.page === 'studio') return;

    stopPlayback();
    manualVoices.current.clear();
    clearActiveNotes();
    pianoAudio.suspendAfter(90);
  }, [route.page]);

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
        songDurationRef.current,
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

  function learningCaptureTime(now = performance.now()) {
    const capture = learningCaptureRef.current;
    if (capture.status !== 'running') return clampSongTime(currentTime);
    return clampSongTime(getSongTimeFromPerformanceClock(now));
  }

  function finishCapturedNote(key, endTime = learningCaptureTime()) {
    const capture = learningCaptureRef.current;
    const started = capture.activeNotes.get(key);
    if (!started) return;
    capture.activeNotes.delete(key);
    capture.notes.push({
      ...started,
      end: endTime,
      duration: Math.max(0.035, endTime - started.start),
    });
  }

  function finishAllCapturedNotes(endTime = learningCaptureTime()) {
    [...learningCaptureRef.current.activeNotes.keys()].forEach((key) => finishCapturedNote(key, endTime));
  }

  function captureNoteStart(note, velocity, playbackOptions = {}) {
    const capture = learningCaptureRef.current;
    if (capture.status !== 'running') return;
    const key = manualVoiceKey(note, playbackOptions);
    const start = learningCaptureTime();
    finishCapturedNote(key, start);
    capture.activeNotes.set(key, {
      note,
      start,
      velocity: Math.max(0.02, Math.min(1, Number(velocity) || 0.85)),
      inputType: playbackOptions.inputType || playbackOptions.pointerType || 'screen',
      dynamicCapable: playbackOptions.inputType === 'midi'
        || (playbackOptions.pointerType === 'pen' && Number(playbackOptions.pressure) > 0),
    });
  }

  function setSustainPedal(nextState, source = 'manual') {
    const down = Boolean(nextState);
    pianoAudio.setSustainPedal(down);
    setPedalDown(down);
    const capture = learningCaptureRef.current;
    if (source === 'manual' && capture.status === 'running') {
      const previous = capture.pedals[capture.pedals.length - 1];
      if (!previous || previous.down !== down) capture.pedals.push({ time: learningCaptureTime(), down });
    }
  }

  function applyPerformanceTier(nextTier, measurements = {}, preserveSampleSet = keyboardReadyRef.current) {
    const normalized = capTierForDevice(
      normalizePerformanceTier(nextTier, performanceTierRef.current),
      deviceClass,
    );
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
    const savedProfile = readSavedPerformanceProfile();
    setKeyboardPreparationStage(savedProfile
      ? `Using saved ${savedProfile.deviceClass || deviceClass} settings`
      : `Checking this ${deviceClass} for 3 seconds`);
    setKeyboardPreparationProgress(0);

    const calibrationTask = savedProfile
      ? Promise.resolve({
        ...(savedProfile.measurements?.calibration || {}),
        tier: savedProfile.tier,
        deviceClass: savedProfile.deviceClass || deviceClass,
        skipped: true,
        reason: 'saved-profile',
      })
      : calibrateDevice({
        durationMs: 3000,
        onProgress: (progress) => {
          setKeyboardPreparationProgress(Math.round(Math.max(0, Math.min(1, progress)) * 20));
        },
      });

    const task = calibrationTask
      .then(async (calibration) => {
        calibrationRef.current = calibration;
        setDeviceClass(calibration.deviceClass || deviceClass);
        const calibratedTier = applyPerformanceTier(calibration.tier, { calibration }, false);
        setKeyboardPreparationStatus('loading');
        setKeyboardPreparationStage('Loading ' + calibratedTier + ' piano');
        setKeyboardPreparationProgress(20);
        const loading = await pianoAudio.prepareKeyboard(({ percent }) => {
          const mappedProgress = 20 + (Math.max(0, Math.min(100, Number(percent) || 0)) * 0.8);
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
    if (source === 'manual' && duration === null) {
      captureNoteStart(note, velocity, playbackOptions);
      manualVoices.current.set(
        manualVoiceKey(note, playbackOptions),
        voice
      );
    }
    return voice;
  }

  function releaseNote(note, interaction = {}) {
    const key = manualVoiceKey(note, interaction);
    finishCapturedNote(key);
    const manualVoice = manualVoices.current.get(key);
    if (manualVoice && typeof pianoAudio.releaseVoice === 'function') {
      pianoAudio.releaseVoice(manualVoice);
      manualVoices.current.delete(key);
      removeActiveNote(note);
    } else if (interaction.pointerId === undefined) {
      pianoAudio.release(note);
      removeActiveNote(note);
    }
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
    if (learningCaptureRef.current.status === 'running' || learningCaptureRef.current.status === 'paused') {
      resetLearningCapture('idle');
      practicePlaybackModeRef.current = 'listen';
    }
    setIsPlaying(false);
    setCurrentTime(0);
    resetPlaybackState();
  }

  async function startPlaybackAt(position, speedOverride = speed) {
    if (!keyboardReadyRef.current && !(await prepareKeyboard())) return false;
    const target = clampSongTime(position);
    const playbackSpeed = Math.max(0.2, Math.min(2, Number(speedOverride) || 1));

    const requestedRunId = playbackRunId.current + 1;
    playbackRunId.current = requestedRunId;
    pianoAudio.setToneMode(toneMode);
    await pianoAudio.preloadSongNotes(song, { startTime: target });
    if (playbackRunId.current !== requestedRunId) return false;

    pauseOffset.current = target;
    nextEventIndex.current = findStartIndex(playbackNotes, Math.max(0, target - 0.0005));
    nextPedalIndex.current = findStartIndex(song.pedals || [], target + 0.0001);
    // An attempt starts with the learner's pedal released. Guide playback may
    // restore the score's pedal state, but that state must never be recorded as
    // if the learner pressed the pedal themselves.
    const initialPedalState = practicePlaybackModeRef.current === 'attempt'
      ? false
      : getPedalStateAt(song.pedals, target);
    setSustainPedal(initialPedalState, 'guide');
    playbackSpeedRef.current = playbackSpeed;
    startStamp.current = performance.now() - (target * 1000) / playbackSpeed;
    setCurrentTime(target);
    setIsPlaying(true);
    setPlaybackEpoch((value) => value + 1);
    return true;
  }

  function pauseLearningCapture(position = currentTime) {
    if (learningCaptureRef.current.status !== 'running') return false;
    finishAllCapturedNotes(clampSongTime(position));
    learningCaptureRef.current.status = 'paused';
    setLearningAttemptStatus('paused');
    return true;
  }

  async function resumePlaybackAt(position, speedOverride = speed) {
    const resumesAttempt = learningCaptureRef.current.status === 'paused';
    if (resumesAttempt) {
      learningCaptureRef.current.status = 'preparing';
      setLearningAttemptStatus('preparing');
      practicePlaybackModeRef.current = 'attempt';
    }
    let started = false;
    try {
      started = await startPlaybackAt(position, speedOverride);
    } catch (error) {
      console.error('Playback could not resume:', error);
    } finally {
      if (resumesAttempt) {
        learningCaptureRef.current.status = started ? 'running' : 'paused';
        setLearningAttemptStatus(started ? 'running' : 'paused');
      }
    }
    return started;
  }

  async function togglePlayPause() {
    if (isPlaying) {
      pauseLearningCapture(currentTime);
      silencePlayback(currentTime);
      setIsPlaying(false);
      return;
    }

    await resumePlaybackAt(pauseOffset.current);
  }

  function beginSeek() {
    seekWasPlaying.current = isPlaying;
    pauseLearningCapture(currentTime);
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
      await resumePlaybackAt(target);
    }
  }

  async function jumpBy(seconds) {
    const shouldResume = isPlaying;
    const target = clampSongTime(currentTime + seconds);
    pauseLearningCapture(currentTime);
    silencePlayback(currentTime);
    setIsPlaying(false);
    previewSeek(target);

    if (shouldResume) {
      await resumePlaybackAt(target);
    }
  }

  async function handleSpeedChange(nextSpeed) {
    const rawValue = typeof nextSpeed === 'function' ? nextSpeed(speed) : nextSpeed;
    const safeSpeed = Math.max(0.2, Math.min(2, Number(rawValue) || 1));
    const shouldResume = isPlaying;
    const position = currentTime;

    if (shouldResume) {
      pauseLearningCapture(position);
      silencePlayback(position);
      setIsPlaying(false);
    }

    setSpeed(safeSpeed);

    if (shouldResume) {
      await resumePlaybackAt(position, safeSpeed);
    }
  }

  function handleSongChange(libraryId) {
    stopPlayback();
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setLearningReport(null);
    setSongSelectionId(libraryId);
  }

  function handleUpload(uploadedSong) {
    const normalized = normalizeSong(uploadedSong);
    const libraryId = songLibraryId(normalized);
    stopPlayback();
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setLearningReport(null);
    setSongs((previous) => [normalized, ...previous.filter((candidate) => songLibraryId(candidate) !== libraryId)]);
    setSongSelectionId(libraryId);
    setOpenMusicChooser(null);
    focusMobilePlayer();
  }

  function rememberPersonalSong(personalSong) {
    if (!personalSong?.id) return;
    setPersonalSongs((previous) => [
      personalSong,
      ...previous.filter((candidate) => candidate.id !== personalSong.id),
    ]);
  }

  async function loadPersonalPianoSong(personalSong) {
    if (!personalSong?.id || loadingPersonalSongId) return;
    setLoadingPersonalSongId(personalSong.id);
    setPersonalSongStatus('Loading your cloud song…');
    try {
      const file = await fetchProtectedFile(
        `/api/personal-songs/${personalSong.id}/download`,
        personalSong.filename || `${personalSong.title}.json`,
      );
      const parsed = await parseUploadedSongFile(file);
      if (!parsed.notes?.length) throw new Error('This song does not contain notes that can be played on piano.');
      handleUpload({
        ...parsed,
        title: personalSong.title || parsed.title,
        composer: personalSong.artist || parsed.composer,
        personalSongId: personalSong.id,
        libraryId: `personal:${personalSong.id}`,
        libraryType: 'personal',
      });
      setPersonalSongStatus('Cloud song ready.');
    } catch (error) {
      setPersonalSongStatus(error.message || 'The cloud song could not be loaded.');
    } finally {
      setLoadingPersonalSongId('');
    }
  }

  function connectMidiMessages(access) {
    const available = [...access.inputs.values()];
    available.forEach((input) => {
      input.onmidimessage = (event) => {
        const [status = 0, data1 = 0, data2 = 0] = event.data || [];
        const command = status & 0xf0;
        const channel = status & 0x0f;
        if (command === 0x90 && data2 > 0) {
          const note = midiToNote(data1);
          pressNote(note, data2 / 127, null, 'manual', {
            pointerId: `midi:${input.id}:${channel}:${data1}`,
            inputType: 'midi',
          });
        } else if (command === 0x80 || (command === 0x90 && data2 === 0)) {
          releaseNote(midiToNote(data1), { pointerId: `midi:${input.id}:${channel}:${data1}` });
        } else if (command === 0xb0 && data1 === 64) {
          setSustainPedal(data2 >= 64, 'manual');
        }
      };
    });
    const connected = available.filter((input) => input.state === 'connected');
    setMidiInput((current) => ({
      ...current,
      status: connected.length ? 'connected' : 'waiting',
      name: connected.map((input) => input.name || input.manufacturer || 'MIDI piano').join(', '),
      error: connected.length ? '' : 'No MIDI keyboard was found. Connect one, then tap Connect MIDI again.',
    }));
  }

  async function connectMidiInput() {
    if (typeof navigator.requestMIDIAccess !== 'function') {
      setMidiInput({ supported: false, status: 'unsupported', name: '', error: 'This browser does not provide Web MIDI. Screen and computer keys still work.' });
      return;
    }
    setMidiInput((current) => ({ ...current, status: 'connecting', error: '' }));
    try {
      const access = await navigator.requestMIDIAccess({ sysex: false });
      midiAccessRef.current = access;
      access.onstatechange = () => connectMidiMessages(access);
      connectMidiMessages(access);
    } catch (error) {
      setMidiInput((current) => ({
        ...current,
        status: 'error',
        error: error?.name === 'SecurityError'
          ? 'MIDI permission was blocked by this browser.'
          : 'The MIDI keyboard could not be connected.',
      }));
    }
  }

  function resetLearningCapture(status = 'idle') {
    learningCaptureRef.current = { status, notes: [], activeNotes: new Map(), pedals: [] };
    setLearningAttemptStatus(status);
  }

  function prepareLearningRange(range) {
    if (!range) return false;
    silencePlayback(range.start);
    setIsPlaying(false);
    const index = learningSections.findIndex((candidate) => candidate.id === range.id);
    if (index >= 0) setSelectedSectionIndex(index);
    setPracticeRange(range);
    setCurrentTime(range.start);
    return true;
  }

  async function listenToLearningRange(range) {
    if (!prepareLearningRange(range)) return;
    practicePlaybackModeRef.current = 'listen';
    resetLearningCapture('idle');
    await startPlaybackAt(range.start);
    focusMobilePlayer(true);
  }

  async function startLearningAttempt(range) {
    if (!range) return;
    if (!keyboardReadyRef.current && !(await prepareKeyboard())) return;
    prepareLearningRange(range);
    practicePlaybackModeRef.current = 'attempt';
    setRepeatSection(false);
    resetLearningCapture('preparing');
    let started = false;
    try {
      started = await startPlaybackAt(range.start);
    } catch (error) {
      console.error('The learning attempt could not start:', error);
    }
    if (!started) {
      resetLearningCapture('idle');
      practicePlaybackModeRef.current = 'listen';
      return;
    }
    learningCaptureRef.current = {
      status: 'running',
      notes: [],
      activeNotes: new Map(),
      pedals: [],
      range,
      songId: songLibraryId(song),
      levelId: learningLevel,
    };
    setLearningAttemptStatus('running');
    focusMobilePlayer(true);
  }

  function completeLearningAttempt(endTime) {
    const capture = learningCaptureRef.current;
    if (capture.status !== 'running') return;
    finishAllCapturedNotes(endTime);
    capture.status = 'complete';
    const report = analyzePracticeAttempt({
      expectedNotes: playbackNotes,
      playedNotes: capture.notes,
      expectedPedals: song.pedals || [],
      playedPedals: capture.pedals,
      range: capture.range,
      levelId: capture.levelId,
    });
    setLearningAttemptStatus('complete');
    setLearningReport(report);
    const learnerId = user?.user_id || 'guest';
    setLearningProgress((current) => {
      const next = recordLearningAttempt(current, capture.songId, report);
      writeLearningProgress(window.localStorage, learnerId, next);
      return next;
    });
  }

  function getSongTimeFromPerformanceClock(now = performance.now()) {
    return ((now - startStamp.current) / 1000) * playbackSpeedRef.current;
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
      pressNote(note, 0.85, null, 'manual', { inputType: 'computer' });
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
  }, [toneMode, speed, song]);

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
        if (practicePlaybackModeRef.current !== 'attempt') {
          const delaySeconds = Math.max(0, (event.time - songNow) / speed);
          const timer = window.setTimeout(() => {
            if (playbackRunId.current === runId) setSustainPedal(event.down, 'guide');
          }, delaySeconds * 1000);
          pedalTimers.current.push(timer);
        }
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

        if (practicePlaybackModeRef.current !== 'attempt') {
          pianoAudio.playAt(event.note, eventVelocity, noteDuration, audioStartAt, {
            source: 'autoplay',
            retriggerSameNote: true,
            releaseSeconds: event.releaseSeconds,
          });
        }
        scheduleVisualStrike(event, delaySeconds, visualDuration, runId);
      }

      if (songNow >= duration) {
        if (practicePlaybackModeRef.current === 'attempt' && learningCaptureRef.current.status === 'running') {
          completeLearningAttempt(duration);
          practicePlaybackModeRef.current = 'listen';
          silencePlayback(practiceRange?.start || 0);
          setIsPlaying(false);
          setCurrentTime(duration);
        } else if (teachingMode === 'learn' && practiceRange && repeatSection) {
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

  function changeLearningLevel(levelId) {
    const level = learningLevelById(levelId);
    stopPlayback();
    setLearningLevel(level.id);
    setPianoHandMode(level.handMode);
    setSpeed(level.speed);
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setLearningReport(null);
    window.localStorage.setItem('polymath-learning-level', level.id);
  }

  function changeLearningSession(sessionId) {
    const session = learningSessionById(sessionId);
    stopPlayback();
    setLearningSession(session.id);
    setPreferredSectionSeconds(session.partSeconds);
    setPracticeRange(null);
    setSelectedSectionIndex(0);
    setLearningReport(null);
    window.localStorage.setItem('polymath-learning-session', session.id);
  }

  function changeLearningHand(hand) {
    stopPlayback();
    setPianoHandMode(hand);
    setPracticeRange(null);
    setLearningReport(null);
  }

  function openLearningMusicChoice(choice) {
    setOpenMusicChooser(choice);
    window.setTimeout(() => {
      const selector = choice === 'upload' ? '.uploader-card' : '.control-panel';
      document.querySelector(selector)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 80);
  }

  function openVirtualTeacher() {
    setTeacherHandsEnabled(true);
    window.setTimeout(() => document.querySelector('.piano-teacher-studio')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
  }

  function requestVirtualTeacherDemonstration(action) {
    if (action?.type !== 'demonstrate_range') return;
    stopPlayback();
    setTeacherHandsEnabled(true);
    setRepeatSection(false);
    if (['left', 'right', 'both'].includes(action.hand)) setPianoHandMode(action.hand);
    if (action.speed !== null && action.speed !== undefined && Number.isFinite(Number(action.speed))) {
      setSpeed(Math.max(0.2, Math.min(2, Number(action.speed))));
    }
    setTeacherDemonstration({ ...action, requestId: Date.now() });
  }

  useEffect(() => {
    if (!teacherDemonstration) return;
    const songDuration = getSongDuration(song);
    if (songDuration <= 0) {
      setTeacherDemonstration(null);
      return;
    }
    const start = Math.max(0, Math.min(Math.max(0, songDuration - 0.5), Number(teacherDemonstration.startSeconds) || 0));
    const requestedEnd = Number(teacherDemonstration.endSeconds) || start + 5;
    const end = Math.min(songDuration, Math.max(start + 0.5, requestedEnd));
    const range = {
      id: `teacher-demo-${teacherDemonstration.requestId}`,
      name: 'Teacher demonstration',
      start,
      end,
      duration: end - start,
    };
    setTeacherDemonstration(null);
    void listenToLearningRange(range);
  }, [teacherDemonstration]);

  const paymentProductId = route.params.get('productId') || 'polymath-chill-monthly';
  const messageUserId = route.params.get('userId');
  const messageName = route.params.get('name') || 'Composer';
  const content = (() => {
    if (route.page === 'model-lab' && import.meta.env.DEV) {
      if (!user?.admin) return <AdminDatabasePage user={user} onNavigate={navigate} />;
      return <ModelLabPage onNavigate={navigate} />;
    }
    if (user?.mustChangePassword && route.page !== 'account') {
      return <AccountPage user={user} setUser={setUser} onNavigate={navigate} />;
    }
    if (route.page === 'guitar') return (
      <GuitarPage
        user={user}
        setUser={setUser}
        onNavigate={navigate}
        personalSongs={personalSongs}
        onPersonalSongSaved={rememberPersonalSong}
      />
    );
    if (route.page === 'ensemble') return (
      <EnsemblePage
        user={user}
        setUser={setUser}
        onNavigate={navigate}
        personalSongs={personalSongs}
        onPersonalSongSaved={rememberPersonalSong}
      />
    );
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
    if (route.page === 'community') return <CommunityPage user={user} onNavigate={navigate} />;
    if (route.page === 'find-teacher') return <TeacherMarketplacePage user={user} onNavigate={navigate} />;
    if (route.page === 'your-songs') return <YourSongsPage user={user} onNavigate={navigate} />;
    if (route.page === 'admin-database') return <AdminDatabasePage user={user} onNavigate={navigate} />;
    if (route.page === 'messages') return <MessagesPage user={user} initialUser={messageUserId ? { user_id: messageUserId, name: messageName } : null} context={route.params.get('context')} onNavigate={navigate} />;
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
      <section className={`studio-page ${teachingMode === 'learn' ? 'is-learning-journey' : ''}`}>
        <PianoLearnJourney
          mode={teachingMode}
          locked={!user?.admin && !user?.access?.learn}
          onUpgrade={() => navigate('payment', { productId: 'polymath-musician-monthly' })}
          onModeChange={(mode) => {
            stopPlayback();
            setTeachingMode(mode);
            if (mode === 'regular') {
              setPracticeRange(null);
              resetLearningCapture('idle');
              practicePlaybackModeRef.current = 'listen';
            }
          }}
          song={song}
          songKey={songLibraryId(song)}
          levelId={learningLevel}
          onLevelChange={changeLearningLevel}
          sessionId={learningSession}
          onSessionChange={changeLearningSession}
          sections={learningSections}
          selectedIndex={selectedSectionIndex}
          onSelectSection={selectLearningSection}
          activeRange={activeLearningRange}
          repeatSection={repeatSection}
          onRepeatChange={setRepeatSection}
          handMode={pianoHandMode}
          onHandModeChange={changeLearningHand}
          onChooseMusic={openLearningMusicChoice}
          onPrepare={prepareKeyboard}
          preparationStatus={keyboardPreparationStatus}
          preparationProgress={keyboardPreparationProgress}
          preparationStage={keyboardPreparationStage}
          midi={midiInput}
          onConnectMidi={connectMidiInput}
          onListen={listenToLearningRange}
          onStartAttempt={startLearningAttempt}
          attemptStatus={learningAttemptStatus}
          report={learningReport}
          progress={learningProgress}
          onOpenTeacher={openVirtualTeacher}
          onFindTeacher={() => navigate('find-teacher')}
          onOpenBand={() => navigate('band')}
          onFocusPlayer={() => focusMobilePlayer(true)}
        />

        <section className={`studio-grid ${openMusicChooser === 'upload' ? 'upload-open' : ''}`}>
          <ControlPanel
            song={song}
            songs={songs}
            onSongChange={handleSongChange}
            onPlayNow={() => {
              setOpenMusicChooser(null);
              focusMobilePlayer(true);
            }}
            expanded={openMusicChooser === 'available'}
            onToggle={() => setOpenMusicChooser((current) => current === 'available' ? null : 'available')}
            personalSongs={personalSongs}
            onPersonalSongChange={loadPersonalPianoSong}
            loadingPersonalSongId={loadingPersonalSongId}
            personalSongStatus={personalSongStatus}
          />
          <div ref={studioPlayerRef} className="visual-stack" tabIndex="-1">
            <FallingNotes song={teachingSong} layout={pianoLayout} currentTime={currentTime} isPlaying={isPlaying} leadTime={leadTime} activeNotes={activeNotes} performanceTier={performanceTier} />
            <div className="piano-scroll-wrap">
              <PianoKeyboard
                layout={pianoLayout}
                activeNotes={activeNotes}
                strikeVersions={strikeVersions}
                showKeyNotes={showKeyNotes}
                onPress={(note, interaction = {}) => pressNote(
                  note,
                  interaction.pointerType === 'pen' && Number(interaction.pressure) > 0
                    ? Math.max(0.12, Math.min(1, Number(interaction.pressure)))
                    : 0.85,
                  null,
                  'manual',
                  { ...interaction, inputType: interaction.pointerType || 'screen' }
                )}
                onRelease={releaseNote}
                preparationStatus={keyboardPreparationStatus}
                preparationProgress={keyboardPreparationProgress}
                preparationStage={keyboardPreparationStage}
                performanceTier={performanceTier}
                deviceClass={deviceClass}
                onPrepare={prepareKeyboard}
                teacher={pianoTeacher}
                teacherTargets={teacherHandTargets}
                showTeacherHands={teachingMode === 'learn' && teacherHandsEnabled}
                teacherHandMode={pianoHandMode}
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
            {teachingMode === 'learn' && (
              <PianoTeacherStudio
                profiles={teacherProfiles}
                teacherId={pianoTeacher.id}
                onTeacherChange={setPianoTeacherId}
                showHands={teacherHandsEnabled}
                onShowHandsChange={(enabled) => {
                  setTeacherHandsEnabled(enabled);
                  if (enabled) {
                    window.setTimeout(() => studioPlayerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
                  }
                }}
                targets={teacherHandTargets}
                isPlaying={isPlaying}
                practiceReport={learningReport}
                lessonContext={{
                  title: song?.title || 'Selected song',
                  composer: song?.composer || song?.artist || '',
                  bpm: song?.bpm || null,
                  level: learningLevel,
                  session: learningSession,
                  hand: pianoHandMode,
                  currentTime,
                  duration: getSongDuration(song),
                  activeRange: activeLearningRange,
                }}
                user={user}
                setUser={setUser}
                onNavigate={navigate}
                onDemonstrate={requestVirtualTeacherDemonstration}
              />
            )}
          </div>

          <SongUploader
            onUpload={handleUpload}
            user={user}
            setUser={setUser}
            onNavigate={navigate}
            expanded={openMusicChooser === 'upload'}
            onToggle={() => setOpenMusicChooser((current) => current === 'upload' ? null : 'upload')}
            onPersonalSongSaved={rememberPersonalSong}
          />
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
      <main className="app-shell">
        <Suspense fallback={<div className="route-loading" role="status">Opening this section…</div>}>
          {content}
        </Suspense>
      </main>
      {user && <SupportAssistant user={user} />}
    </div>
  );
}
