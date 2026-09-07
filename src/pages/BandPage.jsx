import { useEffect, useMemo, useRef, useState } from 'react';
import InstrumentIcon from '../components/InstrumentIcon.jsx';
import InstrumentTeacherSurface from '../components/InstrumentTeacherSurface.jsx';
import MediaTranscriptionPanel from '../components/MediaTranscriptionPanel.jsx';
import { INSTRUMENTS } from '../data/instruments.js';
import { pianoAudio } from '../engine/audioEngine.js';
import { ensembleAudio } from '../engine/ensembleEngine.js';
import { guitarAudio } from '../engine/guitarEngine.js';
import { parseNote, midiToNote } from '../engine/noteMath.js';
import { getSongDuration, normalizeSong } from '../engine/scheduler.js';
import { apiRequest } from '../services/api.js';
import { parseUploadedSongFile } from '../utils/songParser.js';

const GUITAR_TUNING = [40, 45, 50, 55, 59, 64];
const ACCESS_LABELS = {
  private: 'Private rehearsal',
  open: 'Open to everyone',
  password: 'Band password',
  invite: 'Friends by invite code',
  paid: 'Paid entry',
};
const LOCAL_BAND_KEY = 'polymath_local_band_draft';
const PETERSENS_BLUEGRASS_INSTRUMENTS = [
  { instrument: 'guitar', name: 'Acoustic Guitar' },
  { instrument: 'fiddle', name: 'Five-string Fiddle' },
  { instrument: 'banjo', name: 'Five-string Banjo' },
  { instrument: 'mandolin', name: 'Mandolin' },
  { instrument: 'dobro', name: 'Dobro / Resonator Guitar' },
  { instrument: 'upright-bass', name: 'Upright Double Bass' },
];

function readLocalBand() {
  try {
    return JSON.parse(window.localStorage.getItem(LOCAL_BAND_KEY) || 'null');
  } catch {
    return null;
  }
}

function saveLocalBand(band) {
  if (band) window.localStorage.setItem(LOCAL_BAND_KEY, JSON.stringify(band));
  else window.localStorage.removeItem(LOCAL_BAND_KEY);
}

function memberInitials(name) {
  return String(name || '?').split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
}

function resizeAvatar(file) {
  return new Promise((resolve, reject) => {
    if (!file?.type.startsWith('image/')) {
      reject(new Error('Choose an image for your profile picture.'));
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      reject(new Error('Choose an image smaller than 12 MB.'));
      return;
    }
    const image = new window.Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const context = canvas.getContext('2d');
      const sourceSize = Math.min(image.naturalWidth, image.naturalHeight);
      const sourceX = (image.naturalWidth - sourceSize) / 2;
      const sourceY = (image.naturalHeight - sourceSize) / 2;
      context.drawImage(image, sourceX, sourceY, sourceSize, sourceSize, 0, 0, size, size);
      URL.revokeObjectURL(objectUrl);
      resolve(canvas.toDataURL('image/jpeg', 0.82));
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error('That profile picture could not be read.'));
    };
    image.src = objectUrl;
  });
}

function MemberAvatar({ member, className = '' }) {
  return member?.avatarUrl
    ? <img className={`band-avatar ${className}`} src={member.avatarUrl} alt={`${member.name} profile`} />
    : <span className={`band-avatar band-avatar-fallback ${className}`} aria-hidden="true">{memberInitials(member?.name)}</span>;
}

function guitarPosition(note) {
  try {
    const midi = parseNote(note).midi;
    const candidates = GUITAR_TUNING
      .map((openMidi, stringIndex) => ({ stringIndex, fret: midi - openMidi }))
      .filter((item) => item.fret >= 0 && item.fret <= 20)
      .sort((a, b) => a.fret - b.fret || b.stringIndex - a.stringIndex);
    return candidates[0] || null;
  } catch {
    return null;
  }
}

function guitarEventsToSong(data, filename) {
  const notes = [];
  (data.events || data.tabs || []).forEach((event, eventIndex) => {
    const frets = Array.isArray(event.frets)
      ? event.frets
      : Number.isInteger(event.stringIndex) ? Array.from({ length: 6 }, (_, index) => (
        index === event.stringIndex ? event.fret : -1
      )) : [];
    frets.forEach((fret, stringIndex) => {
      if (!Number.isFinite(Number(fret)) || Number(fret) < 0) return;
      notes.push({
        id: `guitar-${eventIndex}-${stringIndex}`,
        note: midiToNote(GUITAR_TUNING[stringIndex] + Number(fret)),
        time: Number(event.time || 0),
        duration: Number(event.duration || 0.5),
        velocity: Number(event.velocity || 0.82),
        stringIndex,
        fret: Number(fret),
      });
    });
  });
  return normalizeSong({
    title: data.title || filename.replace(/\.[^.]+$/, ''),
    composer: data.artist || data.composer || 'Guitar import',
    bpm: data.bpm || 120,
    notes,
  });
}

async function readBandScore(file) {
  if (/\.json$/i.test(file.name)) {
    const data = JSON.parse(await file.text());
    if (Array.isArray(data?.events) || Array.isArray(data?.tabs)) {
      return guitarEventsToSong(data, file.name);
    }
  }
  return parseUploadedSongFile(file);
}

function bandDuration(band) {
  return Math.max(0, ...(band?.instruments || []).map((part) => (
    part.score ? getSongDuration(part.score) : band.generalScore ? getSongDuration(band.generalScore) : 0
  )), band?.generalScore ? getSongDuration(band.generalScore) : 0);
}

export default function BandPage({ user, setUser, onNavigate }) {
  const [bands, setBands] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [publishSourceId, setPublishSourceId] = useState('');
  const [joinSecret, setJoinSecret] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [newInstrument, setNewInstrument] = useState('piano');
  const [create, setCreate] = useState({
    name: '',
    description: '',
    accessMode: 'private',
    password: '',
    entryFeeMcoins: 20,
  });
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activePartNotes, setActivePartNotes] = useState(new Map());
  const [chatMessages, setChatMessages] = useState([]);
  const [chatText, setChatText] = useState('');
  const [chatStatus, setChatStatus] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [avatarBusy, setAvatarBusy] = useState(false);
  const playbackStart = useRef(0);
  const nextIndexes = useRef(new Map());
  const clock = useRef(null);
  const visualTimers = useRef([]);
  const chatBottom = useRef(null);

  const selectedBand = bands.find((band) => band.id === selectedId) || bands[0] || null;
  const duration = useMemo(() => bandDuration(selectedBand), [selectedBand]);

  async function loadBands(preferredId = '') {
    setLoading(true);
    try {
      const requests = [apiRequest('/api/bands')];
      if (user) requests.push(apiRequest('/api/bands/me'));
      const results = await Promise.all(requests);
      const localBand = readLocalBand();
      const combined = [localBand, ...(results[1]?.bands || []), ...(results[0]?.bands || [])].filter(Boolean);
      const unique = [...new Map(combined.map((band) => [band.id, band])).values()];
      setBands(unique);
      setSelectedId((current) => preferredId || current || unique[0]?.id || '');
      setStatus('');
    } catch (error) {
      const localBand = readLocalBand();
      if (localBand) {
        setBands([localBand]);
        setSelectedId(localBand.id);
      }
      setStatus(error.message || 'Bands could not be loaded.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadBands();
  }, [user?.user_id]);

  useEffect(() => {
    setChatMessages([]);
    setChatStatus('');
    if (!selectedBand?.joined || selectedBand.localOnly) return undefined;
    let cancelled = false;
    async function refreshChat() {
      try {
        const data = await apiRequest(`/api/bands/${selectedBand.id}/chat`);
        if (cancelled) return;
        setChatMessages((previous) => {
          const previousLast = previous[previous.length - 1]?.id;
          const nextLast = data.messages[data.messages.length - 1]?.id;
          return previous.length === data.messages.length && previousLast === nextLast ? previous : data.messages;
        });
        setChatStatus('');
      } catch (error) {
        if (!cancelled) setChatStatus(error.message);
      }
    }
    refreshChat();
    const interval = window.setInterval(refreshChat, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [selectedBand?.id, selectedBand?.joined, selectedBand?.localOnly]);

  useEffect(() => {
    chatBottom.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [chatMessages.length]);

  useEffect(() => () => {
    if (clock.current) window.clearInterval(clock.current);
    pianoAudio.stopAll({ releaseSeconds: 0.03 });
    guitarAudio.stopAll();
    ensembleAudio.stopAll();
  }, []);

  useEffect(() => {
    selectedBand?.instruments.forEach((part) => {
      if (!['piano', 'guitar'].includes(part.instrument)) {
        ensembleAudio.preloadInstrument(part.instrument);
      }
    });
  }, [selectedBand]);

  function replaceBand(nextBand) {
    if (nextBand.localOnly) saveLocalBand(nextBand);
    setBands((previous) => [
      nextBand,
      ...previous.filter((band) => band.id !== nextBand.id),
    ]);
    setSelectedId(nextBand.id);
  }

  async function sendChatMessage(event) {
    event.preventDefault();
    const text = chatText.trim();
    if (!text || !selectedBand || chatBusy) return;
    setChatBusy(true);
    try {
      const data = await apiRequest(`/api/bands/${selectedBand.id}/chat`, {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      setChatMessages((previous) => previous.some((message) => message.id === data.message.id)
        ? previous
        : [...previous, data.message]);
      setChatText('');
      setChatStatus('');
    } catch (error) {
      setChatStatus(error.message);
    } finally {
      setChatBusy(false);
    }
  }

  async function updateProfilePicture(file) {
    if (!file || avatarBusy) return;
    setAvatarBusy(true);
    try {
      const avatarDataUrl = await resizeAvatar(file);
      const data = await apiRequest('/api/profile/avatar', {
        method: 'PUT',
        body: JSON.stringify({ avatarDataUrl }),
      });
      setUser(data.user);
      await loadBands(selectedBand?.id);
      setStatus('Profile picture updated for your bandmates.');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setAvatarBusy(false);
    }
  }

  async function moderateMember(member, action) {
    if (!selectedBand?.isHost || member.role === 'host') return;
    const verb = action === 'ban' ? 'ban' : 'remove';
    if (!window.confirm(`${verb === 'ban' ? 'Ban' : 'Remove'} ${member.name} from ${selectedBand.name}?`)) return;
    try {
      const data = await apiRequest(action === 'ban'
        ? `/api/bands/${selectedBand.id}/bans`
        : `/api/bands/${selectedBand.id}/members/${member.userId}`, {
        method: action === 'ban' ? 'POST' : 'DELETE',
        ...(action === 'ban' ? { body: JSON.stringify({ userId: member.userId }) } : {}),
      });
      replaceBand(data.band);
      setStatus(`${member.name} was ${action === 'ban' ? 'banned' : 'removed'} from the band.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function unbanMember(member) {
    try {
      const data = await apiRequest(`/api/bands/${selectedBand.id}/bans/${member.userId}`, { method: 'DELETE' });
      replaceBand(data.band);
      setStatus(`${member.name} can join the band again.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function createBand(event) {
    event.preventDefault();
    if (create.accessMode === 'private') {
      const now = new Date().toISOString();
      const localBand = {
        id: publishSourceId || `local-band-${Date.now()}`,
        name: create.name.trim(),
        description: create.description.trim(),
        host: { userId: user?.user_id || 'local-user', name: user?.name || 'You' },
        accessMode: 'private',
        entryFeeMcoins: 0,
        memberCount: 1,
        members: [{ userId: user?.user_id || 'local-user', name: user?.name || 'You', role: 'host', joinedAt: now }],
        instruments: publishSourceId === selectedBand?.id ? selectedBand.instruments : [],
        generalScore: publishSourceId === selectedBand?.id ? selectedBand.generalScore || null : null,
        isHost: true,
        joined: true,
        role: 'host',
        localOnly: true,
        createdAt: now,
      };
      replaceBand(localBand);
      setShowCreate(false);
      setPublishSourceId('');
      setCreate({ name: '', description: '', accessMode: 'private', password: '', entryFeeMcoins: 20 });
      setStatus('Private rehearsal created. No account is required.');
      return;
    }
    if (!user) {
      const draft = {
        ...(publishSourceId === selectedBand?.id ? selectedBand : {}),
        id: publishSourceId || `local-band-${Date.now()}`,
        name: create.name.trim(),
        description: create.description.trim(),
        accessMode: 'private',
        localOnly: true,
        joined: true,
        isHost: true,
        instruments: publishSourceId === selectedBand?.id ? selectedBand.instruments : [],
        generalScore: publishSourceId === selectedBand?.id ? selectedBand.generalScore || null : null,
      };
      saveLocalBand(draft);
      onNavigate('account');
      return;
    }
    setStatus('Creating band…');
    try {
      const data = await apiRequest('/api/bands', {
        method: 'POST',
        body: JSON.stringify(create),
      });
      let publishedBand = data.band;
      const localSource = publishSourceId === selectedBand?.id ? selectedBand : null;
      if (localSource?.generalScore) {
        const general = await apiRequest(`/api/bands/${publishedBand.id}/general-score`, {
          method: 'PUT',
          body: JSON.stringify({ score: localSource.generalScore }),
        });
        publishedBand = general.band;
      }
      for (const localPart of localSource?.instruments || []) {
        const added = await apiRequest(`/api/bands/${publishedBand.id}/instruments`, {
          method: 'POST',
          body: JSON.stringify({ instrument: localPart.instrument, name: localPart.name }),
        });
        publishedBand = added.band;
        if (localPart.score) {
          const remotePart = publishedBand.instruments[publishedBand.instruments.length - 1];
          const uploaded = await apiRequest(`/api/bands/${publishedBand.id}/instruments/${remotePart.id}`, {
            method: 'PUT',
            body: JSON.stringify({ score: localPart.score, muted: localPart.muted, volume: localPart.volume }),
          });
          publishedBand = uploaded.band;
        }
      }
      saveLocalBand(null);
      replaceBand(publishedBand);
      if (data.user) setUser(data.user);
      setShowCreate(false);
      setPublishSourceId('');
      setCreate({ name: '', description: '', accessMode: 'private', password: '', entryFeeMcoins: 20 });
      setStatus(localSource ? 'Private rehearsal published for your team.' : 'Band created. Start by adding an instrument.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function joinBand() {
    if (!user) {
      onNavigate('account');
      return;
    }
    setStatus('Joining band…');
    try {
      const data = await apiRequest(`/api/bands/${selectedBand.id}/join`, {
        method: 'POST',
        body: JSON.stringify({ password: joinSecret }),
      });
      replaceBand(data.band);
      if (data.user) setUser(data.user);
      setJoinSecret('');
      setStatus(`You joined ${data.band.name}.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function joinByCode(event) {
    event.preventDefault();
    if (!user) {
      onNavigate('account');
      return;
    }
    try {
      const data = await apiRequest('/api/bands/join-by-code', {
        method: 'POST',
        body: JSON.stringify({ code: inviteCode }),
      });
      replaceBand(data.band);
      setInviteCode('');
      setStatus(`You joined ${data.band.name} with a friend invite.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function addInstrument() {
    if (!selectedBand?.joined) return;
    try {
      const selected = INSTRUMENTS.find((instrument) => instrument.id === newInstrument);
      if (selectedBand.localOnly) {
        replaceBand({
          ...selectedBand,
          instruments: [...selectedBand.instruments, {
            id: `local-part-${Date.now()}`,
            instrument: newInstrument,
            name: selected?.label || 'Instrument',
            addedBy: user?.user_id || 'local-user',
            score: null,
            muted: false,
            volume: 0.82,
            visualEnabled: false,
            createdAt: new Date().toISOString(),
          }],
        });
        setStatus(`${selected?.label || 'Instrument'} added to the private rehearsal.`);
        return;
      }
      const data = await apiRequest(`/api/bands/${selectedBand.id}/instruments`, {
        method: 'POST',
        body: JSON.stringify({ instrument: newInstrument, name: selected?.label }),
      });
      replaceBand(data.band);
      setStatus(`${selected?.label || 'Instrument'} added to the band.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function buildPresetParts(existing = []) {
    const existingIds = new Set(existing.map((part) => part.instrument));
    const now = Date.now();
    return PETERSENS_BLUEGRASS_INSTRUMENTS
      .filter((preset) => !existingIds.has(preset.instrument))
      .map((preset, index) => ({
        id: `local-petersens-part-${now}-${index}`,
        ...preset,
        addedBy: user?.user_id || 'local-user',
        score: null,
        muted: false,
        volume: 0.82,
        visualEnabled: false,
        createdAt: new Date(now + index).toISOString(),
      }));
  }

  async function setupPetersensBluegrass() {
    const now = new Date().toISOString();
    if (!selectedBand) {
      const localBand = {
        id: `local-band-${Date.now()}`,
        name: 'Bluegrass — The Petersens',
        description: 'Six-part bluegrass arrangement: acoustic guitar, fiddle, banjo, mandolin, dobro, and upright bass.',
        host: { userId: user?.user_id || 'local-user', name: user?.name || 'You' },
        accessMode: 'private',
        entryFeeMcoins: 0,
        memberCount: 1,
        members: [{ userId: user?.user_id || 'local-user', name: user?.name || 'You', role: 'host', joinedAt: now }],
        instruments: buildPresetParts(),
        generalScore: null,
        isHost: true,
        joined: true,
        role: 'host',
        localOnly: true,
        createdAt: now,
      };
      replaceBand(localBand);
      setStatus('Bluegrass — The Petersens setup is ready with all six instruments.');
      return;
    }
    if (!selectedBand.joined) {
      setStatus('Join this band before changing its instrument setup.');
      return;
    }
    const missing = PETERSENS_BLUEGRASS_INSTRUMENTS.filter((preset) => (
      !selectedBand.instruments.some((part) => part.instrument === preset.instrument)
    ));
    if (!missing.length) {
      setStatus('All six Bluegrass — The Petersens instruments are already on this stage.');
      return;
    }
    try {
      if (selectedBand.localOnly) {
        replaceBand({
          ...selectedBand,
          instruments: [...selectedBand.instruments, ...buildPresetParts(selectedBand.instruments)],
        });
      } else {
        let updatedBand = selectedBand;
        for (const preset of missing) {
          const data = await apiRequest(`/api/bands/${updatedBand.id}/instruments`, {
            method: 'POST',
            body: JSON.stringify(preset),
          });
          updatedBand = data.band;
        }
        replaceBand(updatedBand);
      }
      setStatus(`Added ${missing.length} missing bluegrass instrument${missing.length === 1 ? '' : 's'}.`);
    } catch (error) {
      setStatus(error.message || 'The bluegrass setup could not be completed.');
    }
  }

  async function uploadPart(part, file) {
    if (!file) return;
    setStatus(`Reading ${file.name} for ${part.name}…`);
    try {
      const parsed = await readBandScore(file);
      if (!parsed.notes.length) throw new Error('No playable notes were found in that file.');
      if (selectedBand.localOnly) {
        replaceBand({
          ...selectedBand,
          instruments: selectedBand.instruments.map((candidate) => (
            candidate.id === part.id ? { ...candidate, score: parsed } : candidate
          )),
        });
        setStatus(`${parsed.title} loaded into ${part.name}.`);
        return;
      }
      const data = await apiRequest(`/api/bands/${selectedBand.id}/instruments/${part.id}`, {
        method: 'PUT',
        body: JSON.stringify({ score: parsed }),
      });
      replaceBand(data.band);
      setStatus(`${parsed.title} loaded into ${part.name}.`);
    } catch (error) {
      setStatus(error.message || 'That part could not be loaded.');
    }
  }

  async function uploadGeneralScore(file, rethrow = false) {
    if (!file) return;
    setStatus(`Reading general music sheet: ${file.name}…`);
    try {
      const parsed = await readBandScore(file);
      if (!parsed.notes.length) throw new Error('No playable notes were found in that sheet.');
      if (selectedBand.localOnly) {
        replaceBand({ ...selectedBand, generalScore: parsed });
      } else {
        const data = await apiRequest(`/api/bands/${selectedBand.id}/general-score`, {
          method: 'PUT',
          body: JSON.stringify({ score: parsed }),
        });
        replaceBand(data.band);
      }
      setStatus(`${parsed.title} is now the general sheet. Every instrument without its own sheet will play it.`);
      return parsed;
    } catch (error) {
      setStatus(error.message || 'The general music sheet could not be loaded.');
      if (rethrow) throw error;
      return null;
    }
  }

  async function updatePart(part, changes) {
    try {
      if (selectedBand.localOnly) {
        replaceBand({
          ...selectedBand,
          instruments: selectedBand.instruments.map((candidate) => (
            candidate.id === part.id ? { ...candidate, ...changes } : candidate
          )),
        });
        return;
      }
      const data = await apiRequest(`/api/bands/${selectedBand.id}/instruments/${part.id}`, {
        method: 'PUT',
        body: JSON.stringify(changes),
      });
      replaceBand(data.band);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function removePart(part) {
    try {
      if (selectedBand.localOnly) {
        replaceBand({
          ...selectedBand,
          instruments: selectedBand.instruments.filter((candidate) => candidate.id !== part.id),
        });
        setStatus(`${part.name} removed.`);
        return;
      }
      const data = await apiRequest(`/api/bands/${selectedBand.id}/instruments/${part.id}`, {
        method: 'DELETE',
      });
      replaceBand(data.band);
      setStatus(`${part.name} removed.`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function playPartNote(part, note) {
    const velocity = (note.velocity || 0.8) * (part.volume ?? 0.82);
    const noteDuration = Math.max(0.04, note.duration || 0.4);
    if (part.instrument === 'piano') {
      pianoAudio.play(note.note, velocity, noteDuration, { source: 'autoplay' });
      return;
    }
    if (part.instrument === 'guitar') {
      const position = Number.isInteger(note.stringIndex) && Number.isFinite(note.fret)
        ? { stringIndex: note.stringIndex, fret: note.fret }
        : guitarPosition(note.note);
      if (position) {
        guitarAudio.playEvent({ ...position, velocity, duration: noteDuration }, guitarAudio.getCurrentTime());
        return;
      }
    }
    ensembleAudio.play(note.note, part.instrument, velocity, noteDuration);
  }

  function highlightPartNote(part, note) {
    if (!part.visualEnabled) return;
    setActivePartNotes((previous) => {
      const next = new Map(previous);
      const notes = new Set(next.get(part.id) || []);
      notes.add(note.note);
      next.set(part.id, notes);
      return next;
    });
    const timer = window.setTimeout(() => {
      setActivePartNotes((previous) => {
        const next = new Map(previous);
        const notes = new Set(next.get(part.id) || []);
        notes.delete(note.note);
        if (notes.size) next.set(part.id, notes);
        else next.delete(part.id);
        return next;
      });
    }, Math.max(100, Number(note.duration || 0.4) * 1000));
    visualTimers.current.push(timer);
  }

  function playVisualTarget(part, note) {
    playPartNote(part, { note, velocity: 0.86, duration: 0.7 });
    highlightPartNote(part, { note, duration: 0.7 });
  }

  function stopBand(reset = true) {
    if (clock.current) window.clearInterval(clock.current);
    clock.current = null;
    setIsPlaying(false);
    pianoAudio.stopAll({ releaseSeconds: 0.03 });
    guitarAudio.stopAll();
    ensembleAudio.stopAll();
    visualTimers.current.forEach((timer) => window.clearTimeout(timer));
    visualTimers.current = [];
    setActivePartNotes(new Map());
    if (reset) setCurrentTime(0);
  }

  function playBand() {
    if (!selectedBand || !duration) return;
    if (isPlaying) {
      stopBand(false);
      return;
    }
    nextIndexes.current = new Map(
      selectedBand.instruments.map((part) => {
        const score = part.score || selectedBand.generalScore;
        return [part.id, score?.notes.findIndex((note) => note.time >= currentTime) ?? 0];
      }),
    );
    playbackStart.current = performance.now() - currentTime * 1000;
    setIsPlaying(true);
    clock.current = window.setInterval(() => {
      const now = (performance.now() - playbackStart.current) / 1000;
      setCurrentTime(now);
      selectedBand.instruments.forEach((part) => {
        const score = part.score || selectedBand.generalScore;
        if (part.muted || !score) return;
        let index = Math.max(0, nextIndexes.current.get(part.id) || 0);
        while (index < score.notes.length && score.notes[index].time <= now + 0.035) {
          playPartNote(part, score.notes[index]);
          highlightPartNote(part, score.notes[index]);
          index += 1;
        }
        nextIndexes.current.set(part.id, index);
      });
      if (now >= duration) stopBand(true);
    }, 20);
  }

  function preparePublish() {
    setPublishSourceId(selectedBand.id);
    setCreate({
      name: selectedBand.name,
      description: selectedBand.description || '',
      accessMode: 'open',
      password: '',
      entryFeeMcoins: 20,
    });
    setShowCreate(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  return (
    <section className="page-shell band-page">
      <header className="band-heading">
        <div>
          <p className="eyebrow">Polymath Musician Band</p>
          <h1>Build a band, one instrument at a time.</h1>
          <p>Start with an empty stage, add the parts your team needs, and load compatible JSON, MIDI, MusicXML, or CSV into each instrument.</p>
        </div>
        <div className="band-heading-actions">
          <button className="ghost" type="button" onClick={setupPetersensBluegrass}>
            Bluegrass — The Petersens
          </button>
          <button className="primary" type="button" onClick={() => setShowCreate((value) => !value)}>
            Create a band
          </button>
        </div>
      </header>

      {showCreate && (
        <form className="band-create-card" onSubmit={createBand}>
          <div><p className="eyebrow">{publishSourceId ? 'Publish rehearsal' : 'New band'}</p><h2>{publishSourceId ? 'Choose how your team enters' : 'Start privately or invite a team'}</h2></div>
          <label className="field">Band name<input value={create.name} onChange={(event) => setCreate({ ...create, name: event.target.value })} required /></label>
          <label className="field">Description<input value={create.description} onChange={(event) => setCreate({ ...create, description: event.target.value })} placeholder="What are you rehearsing?" /></label>
          <label className="field">Entry control<select value={create.accessMode} onChange={(event) => setCreate({ ...create, accessMode: event.target.value })}>
            {Object.entries(ACCESS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          {create.accessMode === 'password' && <label className="field">Band password<input type="password" minLength="4" value={create.password} onChange={(event) => setCreate({ ...create, password: event.target.value })} required /></label>}
          {create.accessMode === 'paid' && <label className="field">Entry fee in Mcoins<input type="number" min="1" max="100000" value={create.entryFeeMcoins} onChange={(event) => setCreate({ ...create, entryFeeMcoins: Number(event.target.value) })} required /></label>}
          <button className="primary" type="submit">{create.accessMode === 'private' ? 'Start private rehearsal' : user ? 'Publish band' : 'Sign in to publish'}</button>
        </form>
      )}

      <div className="band-browser">
        <aside className="band-list-card">
          <div className="band-list-heading"><div><p className="eyebrow">Bands</p><h2>Your stage or a new team</h2></div><button className="text-button" type="button" onClick={() => loadBands()}>Refresh</button></div>
          <form className="band-code-form" onSubmit={joinByCode}>
            <input value={inviteCode} onChange={(event) => setInviteCode(event.target.value)} placeholder="Friend invite code" aria-label="Friend invite code" />
            <button className="ghost" type="submit">Join</button>
          </form>
          <div className="band-list">
            {loading && <p className="muted">Loading bands…</p>}
            {!loading && !bands.length && <p className="muted">No public bands yet. Create the first one.</p>}
            {bands.map((band) => (
              <button key={band.id} type="button" className={selectedBand?.id === band.id ? 'active' : ''} onClick={() => setSelectedId(band.id)}>
                <strong>{band.name}</strong>
                <span>{band.host.name} · {band.memberCount} member{band.memberCount === 1 ? '' : 's'}</span>
                <small>{ACCESS_LABELS[band.accessMode]}{band.accessMode === 'paid' ? ` · ${band.entryFeeMcoins} Mcoins` : ''}</small>
              </button>
            ))}
          </div>
        </aside>

        <main className="band-workspace">
          {!selectedBand ? (
            <div className="band-empty-state"><span>+</span><h2>No band selected</h2><p>Create a band or join one to begin adding instruments.</p></div>
          ) : (
            <>
              <section className="band-overview">
                <div>
                  <p className="eyebrow">{selectedBand.localOnly ? 'Private rehearsal' : selectedBand.isHost ? 'Your band' : selectedBand.joined ? 'Joined band' : 'Band preview'}</p>
                  <h2>{selectedBand.name}</h2>
                  <p className="muted">{selectedBand.description || `Hosted by ${selectedBand.host.name}`}</p>
                </div>
                <div className="band-access-badge">
                  <strong>{ACCESS_LABELS[selectedBand.accessMode]}</strong>
                  <span>{selectedBand.memberCount} member{selectedBand.memberCount === 1 ? '' : 's'}</span>
                </div>
                {selectedBand.isHost && !selectedBand.localOnly && <div className="band-invite-code"><span>Friend invite code</span><strong>{selectedBand.inviteCode}</strong></div>}
                {selectedBand.localOnly && <button className="primary band-publish-button" type="button" onClick={preparePublish}>Publish for a team</button>}
                {!selectedBand.joined && (
                  <div className="band-join-box">
                    {selectedBand.accessMode === 'password' && <input type="password" value={joinSecret} onChange={(event) => setJoinSecret(event.target.value)} placeholder="Band password" />}
                    <button className="primary" type="button" onClick={joinBand}>
                      {selectedBand.accessMode === 'paid' ? `Join for ${selectedBand.entryFeeMcoins} Mcoins` : 'Join band'}
                    </button>
                  </div>
                )}
              </section>

              {selectedBand.joined && (
                <>
                  {!selectedBand.localOnly && (
                    <section className="band-room">
                      <header className="band-room-heading">
                        <div>
                          <p className="eyebrow">Band room</p>
                          <h2>Meet, chat, rehearse.</h2>
                        </div>
                        <label className={`ghost band-avatar-upload ${avatarBusy ? 'disabled' : ''}`}>
                          <input type="file" accept="image/jpeg,image/png,image/webp,image/gif" disabled={avatarBusy} onChange={(event) => updateProfilePicture(event.target.files?.[0])} />
                          {avatarBusy ? 'Updating…' : user?.avatarUrl ? 'Change my photo' : 'Add my photo'}
                        </label>
                      </header>

                      <div className="band-room-grid">
                        <aside className="band-member-panel">
                          <div className="band-section-label"><strong>Members</strong><span>{selectedBand.members?.length || 0}</span></div>
                          <div className="band-member-list">
                            {(selectedBand.members || []).map((member) => (
                              <article className="band-member-card" key={member.userId}>
                                <MemberAvatar member={member} />
                                <div>
                                  <strong>{member.name}</strong>
                                  <small>{member.role === 'host' ? 'Creator' : 'Member'}</small>
                                </div>
                                {selectedBand.isHost && member.role !== 'host' && (
                                  <div className="band-member-actions">
                                    <button className="text-button" type="button" onClick={() => moderateMember(member, 'kick')}>Kick</button>
                                    <button className="text-button danger-text" type="button" onClick={() => moderateMember(member, 'ban')}>Ban</button>
                                  </div>
                                )}
                              </article>
                            ))}
                          </div>
                          {selectedBand.isHost && selectedBand.bannedMembers?.length > 0 && (
                            <details className="band-ban-list">
                              <summary>Banned accounts ({selectedBand.bannedMembers.length})</summary>
                              {selectedBand.bannedMembers.map((member) => (
                                <div key={member.userId}>
                                  <MemberAvatar member={member} className="small" />
                                  <span>{member.name}</span>
                                  <button className="text-button" type="button" onClick={() => unbanMember(member)}>Unban</button>
                                </div>
                              ))}
                            </details>
                          )}
                        </aside>

                        <section className="band-chat" aria-label={`${selectedBand.name} chat`}>
                          <div className="band-section-label"><strong>Chat</strong><span>Members only</span></div>
                          <div className="band-chat-messages" aria-live="polite">
                            {!chatMessages.length && <p className="band-chat-empty">Say hello to your band.</p>}
                            {chatMessages.map((message) => {
                              const own = message.author.userId === user?.user_id;
                              return (
                                <article className={`band-chat-message ${own ? 'own' : ''}`} key={message.id}>
                                  {!own && <MemberAvatar member={message.author} className="small" />}
                                  <div>
                                    <span><strong>{own ? 'You' : message.author.name}</strong><time>{new Date(message.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></span>
                                    <p>{message.text}</p>
                                  </div>
                                </article>
                              );
                            })}
                            <div ref={chatBottom} />
                          </div>
                          <form className="band-chat-form" onSubmit={sendChatMessage}>
                            <textarea
                              value={chatText}
                              onChange={(event) => setChatText(event.target.value)}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter' && !event.shiftKey) {
                                  event.preventDefault();
                                  event.currentTarget.form?.requestSubmit();
                                }
                              }}
                              maxLength="1000"
                              rows="2"
                              placeholder="Message the band…"
                              aria-label="Message the band"
                            />
                            <button className="primary" type="submit" disabled={chatBusy || !chatText.trim()}>{chatBusy ? 'Sending…' : 'Send'}</button>
                          </form>
                          {chatStatus && <p className="form-status band-chat-status">{chatStatus}</p>}
                        </section>
                      </div>
                    </section>
                  )}

                  <section className="band-transport">
                    <button className="primary" type="button" disabled={!duration} onClick={playBand}>{isPlaying ? 'Pause band' : currentTime > 0 ? 'Resume band' : 'Play band'}</button>
                    <button className="ghost" type="button" onClick={() => stopBand(true)}>Restart</button>
                    <label><span>{currentTime.toFixed(1)}s / {duration.toFixed(1)}s</span><input type="range" min="0" max={Math.max(duration, 0.1)} step="0.05" value={Math.min(currentTime, duration)} onChange={(event) => { stopBand(false); setCurrentTime(Number(event.target.value)); }} /></label>
                  </section>

                  <section className="band-stage">
                    <div className="band-bluegrass-preset">
                      <div>
                        <p className="eyebrow">Bluegrass — The Petersens</p>
                        <h2>Load the complete six-instrument stage</h2>
                        <p>Acoustic guitar, five-string fiddle, banjo, mandolin, dobro, and upright bass—ready for individual parts or one shared MIDI/JSON arrangement.</p>
                      </div>
                      <button className="primary" type="button" onClick={setupPetersensBluegrass}>
                        Auto-set up instruments
                      </button>
                    </div>
                    <div className={`band-general-sheet ${selectedBand.generalScore ? 'loaded' : ''}`}>
                      <div>
                        <p className="eyebrow">General music sheet</p>
                        <h2>{selectedBand.generalScore ? selectedBand.generalScore.title : 'One sheet for the whole band'}</h2>
                        <p>{selectedBand.generalScore
                          ? `${selectedBand.generalScore.notes.length} notes · Every instrument without an individual sheet will play this arrangement.`
                          : 'Upload a piano or other JSON/MIDI sheet here and every added instrument will play the same notes through its own sound.'}</p>
                      </div>
                      <label className="band-general-upload">
                        <input type="file" accept=".json,.mid,.midi" onChange={(event) => uploadGeneralScore(event.target.files?.[0])} />
                        {selectedBand.generalScore ? 'Replace general sheet' : 'Upload JSON / MIDI'}
                      </label>
                      <details className="band-media-transcription">
                        <summary>Transcribe MP3 / Music Video with Polymath</summary>
                        <MediaTranscriptionPanel
                          user={user}
                          setUser={setUser}
                          onNavigate={onNavigate}
                          instrument="band"
                          onReadyFile={(file) => uploadGeneralScore(file, true)}
                        />
                      </details>
                    </div>
                    <div className="band-stage-heading">
                      <div><p className="eyebrow">Band arrangement</p><h2>{selectedBand.instruments.length ? `${selectedBand.instruments.length} instrument parts` : 'The stage is empty'}</h2></div>
                      <div className="band-add-instrument">
                        <select value={newInstrument} onChange={(event) => setNewInstrument(event.target.value)}>
                          {INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}
                        </select>
                        <button className="primary" type="button" onClick={addInstrument}>+ Add instrument</button>
                      </div>
                    </div>

                    {!selectedBand.instruments.length && (
                      <div className="band-empty-state">
                        <span>+</span>
                        <h3>Add the first instrument</h3>
                        <p>Piano, guitar, bass, drums, fiddle, banjo, mandolin, dobro, ukulele, electric guitar, and synth can share this stage.</p>
                      </div>
                    )}

                    <div className="band-part-grid">
                      {selectedBand.instruments.map((part) => (
                        <article className={`band-part-card ${part.muted ? 'muted-part' : ''} ${part.visualEnabled ? 'visual-part' : ''}`} key={part.id}>
                          <header>
                            <InstrumentIcon instrument={part.instrument} size="lg" />
                            <div><strong>{part.name}</strong><span>{part.score ? part.score.title : selectedBand.generalScore ? `General: ${selectedBand.generalScore.title}` : 'Waiting for a part'}</span></div>
                            <button className="band-remove-part" type="button" onClick={() => removePart(part)} aria-label={`Remove ${part.name}`}>×</button>
                          </header>
                          {(part.score || selectedBand.generalScore) ? (
                            <div className="band-part-summary">
                              <strong>{(part.score || selectedBand.generalScore).notes.length}</strong>
                              <span>notes · {getSongDuration(part.score || selectedBand.generalScore).toFixed(1)} seconds{!part.score ? ' · general sheet' : ''}</span>
                            </div>
                          ) : (
                            <p className="muted">Upload a ready-to-play part. Piano MIDI/JSON can be routed to any pitched instrument.</p>
                          )}
                          <label className="band-part-upload">
                            <input type="file" accept=".json,.csv,.mid,.midi,.musicxml,.xml" onChange={(event) => uploadPart(part, event.target.files?.[0])} />
                            {part.score ? 'Replace sheet' : 'Upload MIDI / JSON'}
                          </label>
                          <div className="band-part-controls">
                            <button className="ghost" type="button" onClick={() => updatePart(part, { muted: !part.muted })}>{part.muted ? 'Unmute' : 'Mute'}</button>
                            <button className="ghost" type="button" onClick={() => updatePart(part, { visualEnabled: !part.visualEnabled })}>{part.visualEnabled ? 'Hide instrument' : 'Show instrument'}</button>
                            <label><span>Volume {Math.round((part.volume ?? 0.82) * 100)}%</span><input type="range" min="0" max="1.2" step="0.05" value={part.volume ?? 0.82} onChange={(event) => updatePart(part, { volume: Number(event.target.value) })} /></label>
                          </div>
                          {part.visualEnabled && (
                            <div className="band-instrument-visual">
                              <InstrumentTeacherSurface
                                instrument={part.instrument}
                                activeNotes={activePartNotes.get(part.id) || new Set()}
                                onPlay={(note) => playVisualTarget(part, note)}
                              />
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  </section>
                </>
              )}
            </>
          )}
        </main>
      </div>
      {status && <div className="floating-status">{status}</div>}
    </section>
  );
}
