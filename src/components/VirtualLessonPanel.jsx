import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';
import {
  selectTeacherVoice,
  speechSegments,
  teacherVoiceProfile,
} from '../engine/teacherVoice.js';

const DEFAULT_CATALOG = {
  rateMcoinsPerHour: 10,
  pricePer30MinutesMcoins: 5,
  durationStepMinutes: 30,
  minimumDurationMinutes: 30,
  maximumDurationMinutes: 720,
  defaultDurationMinutes: 60,
  mcoinsPerUsd: 1,
  memoryPolicy: 'session-only',
  conversationModes: [
    { id: 'music-coach', label: 'Music teacher' },
    { id: 'adult-companion', label: 'Flirty companion', requiresAdultConfirmation: true },
  ],
};

export function normalizeLessonMinutes(value, catalog = DEFAULT_CATALOG) {
  if (String(value ?? '').trim() === '') return null;
  const requested = Number(value);
  if (!Number.isFinite(requested)) return null;
  const step = Math.max(1, Number(catalog.durationStepMinutes) || 30);
  const minimum = Math.max(step, Number(catalog.minimumDurationMinutes) || step);
  const maximum = Math.max(minimum, Number(catalog.maximumDurationMinutes) || 720);
  return Math.min(maximum, Math.max(minimum, Math.round(requested / step) * step));
}

export function lessonQuoteForCatalog(value, catalog = DEFAULT_CATALOG) {
  const durationMinutes = normalizeLessonMinutes(value, catalog);
  if (!durationMinutes) return null;
  const requestedDurationMinutes = Number(value);
  const step = Math.max(1, Number(catalog.durationStepMinutes) || 30);
  const pricePerBlock = Math.max(0, Number(catalog.pricePer30MinutesMcoins) || 0);
  const priceMcoins = Number(((durationMinutes / step) * pricePerBlock).toFixed(2));
  return {
    requestedDurationMinutes,
    durationMinutes,
    rounded: requestedDurationMinutes !== durationMinutes,
    priceMcoins,
    priceUsd: priceMcoins,
  };
}

function requestId() {
  if (window.crypto?.randomUUID) return `lesson_${window.crypto.randomUUID()}`;
  return `lesson_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function remainingSeconds(session, now = Date.now()) {
  if (!session?.expiresAt || session.status !== 'active') return 0;
  return Math.max(0, Math.ceil((new Date(session.expiresAt).getTime() - now) / 1000));
}

export function formatLessonTime(seconds) {
  const safe = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${minutes}:${String(remainder).padStart(2, '0')}`;
}

function recognitionEngine() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function TeacherPortrait({ teacher }) {
  return (
    <span className="teacher-human-portrait" aria-hidden="true">
      <img
        src={teacher.image}
        alt=""
        loading="lazy"
        draggable="false"
        style={teacher.portraitPosition ? { objectPosition: teacher.portraitPosition } : undefined}
      />
    </span>
  );
}

export default function VirtualLessonPanel({
  user,
  setUser,
  teacher,
  onTeacherChange,
  lessonContext,
  observations,
  onDemonstrate,
  onNavigate,
}) {
  const [catalog, setCatalog] = useState(DEFAULT_CATALOG);
  const [assistantAvailable, setAssistantAvailable] = useState(true);
  const [session, setSession] = useState(null);
  const [durationMinutes, setDurationMinutes] = useState(60);
  const [draft, setDraft] = useState('');
  const [waiting, setWaiting] = useState(false);
  const [loading, setLoading] = useState(Boolean(user));
  const [status, setStatus] = useState('');
  const [now, setNow] = useState(Date.now());
  const [listening, setListening] = useState(false);
  const [voiceOutput, setVoiceOutput] = useState(true);
  const [conversationMode, setConversationMode] = useState('music-coach');
  const [companionStyle, setCompanionStyle] = useState('playful');
  const [adultConfirmed, setAdultConfirmed] = useState(() => Boolean(
    user?.adultCompanionConfirmed
    || window.localStorage.getItem('polymath-teacher-adult-confirmed') === 'true',
  ));
  const [companionConsent, setCompanionConsent] = useState(false);
  const [availableVoices, setAvailableVoices] = useState([]);
  const [selectedVoiceUri, setSelectedVoiceUri] = useState(() => (
    window.localStorage.getItem(`polymath-teacher-voice-${teacher.id}`) || ''
  ));
  const recognitionRef = useRef(null);
  const voiceFinalRef = useRef('');
  const checkoutRef = useRef('');
  const speechRunRef = useRef(0);
  const speechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;
  const recognitionSupported = typeof window !== 'undefined' && Boolean(recognitionEngine());
  const messages = session?.messages || [];
  const secondsLeft = remainingSeconds(session, now);
  const selectedQuote = useMemo(
    () => lessonQuoteForCatalog(durationMinutes, catalog),
    [catalog, durationMinutes],
  );
  const bestVoice = useMemo(() => selectTeacherVoice(
    availableVoices,
    teacher,
    navigator.language || 'en-US',
    selectedVoiceUri,
  ), [availableVoices, selectedVoiceUri, teacher]);
  const voiceChoices = useMemo(() => {
    const language = String(navigator.language || 'en-US').split('-')[0].toLowerCase();
    const matching = availableVoices.filter((voice) => String(voice.lang || '').toLowerCase().startsWith(language));
    return (matching.length ? matching : availableVoices)
      .slice()
      .sort((left, right) => String(left.name).localeCompare(String(right.name)));
  }, [availableVoices]);
  const quickPrompts = session?.conversationMode === 'adult-companion'
    ? ['Talk music with me', 'Ask me something fun', 'Help me practise']
    : ['Show the first 5 seconds', 'Help my timing', 'Ask something off-topic'];

  useEffect(() => {
    if (!user) {
      setSession(null);
      setLoading(false);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    apiRequest('/api/virtual-lessons')
      .then((data) => {
        if (cancelled) return;
        setCatalog(data.catalog || DEFAULT_CATALOG);
        setDurationMinutes((current) => (
          normalizeLessonMinutes(current, data.catalog || DEFAULT_CATALOG)
          || data.catalog?.defaultDurationMinutes
          || DEFAULT_CATALOG.defaultDurationMinutes
        ));
        setAssistantAvailable(Boolean(data.assistantAvailable));
        setSession(data.session || null);
        if (data.session?.conversationMode) setConversationMode(data.session.conversationMode);
        if (data.session?.conversationPreferences?.companionStyle) {
          setCompanionStyle(data.session.conversationPreferences.companionStyle);
        }
        if (data.session?.adultCompanionConfirmed) setAdultConfirmed(true);
        if (data.user) setUser?.(data.user);
        if (data.session?.teacher?.id && data.session.teacher.id !== teacher.id) {
          onTeacherChange?.(data.session.teacher.id);
        }
      })
      .catch((error) => {
        if (!cancelled) setStatus(error.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [user?.user_id]);

  useEffect(() => {
    if (!teacher.requiresAdultConfirmation && conversationMode === 'adult-companion') {
      setConversationMode('music-coach');
      setCompanionConsent(false);
    }
    setSelectedVoiceUri(window.localStorage.getItem(`polymath-teacher-voice-${teacher.id}`) || '');
  }, [teacher.id, teacher.requiresAdultConfirmation, conversationMode]);

  useEffect(() => {
    if (!speechSupported) return undefined;
    const synthesis = window.speechSynthesis;
    const refreshVoices = () => setAvailableVoices(synthesis.getVoices());
    refreshVoices();
    synthesis.addEventListener?.('voiceschanged', refreshVoices);
    return () => synthesis.removeEventListener?.('voiceschanged', refreshVoices);
  }, [speechSupported]);

  useEffect(() => {
    if (!session || session.status !== 'active') return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [session?.id, session?.status]);

  useEffect(() => {
    if (session?.status === 'active' && session.teacher?.id && session.teacher.id !== teacher.id) {
      onTeacherChange?.(session.teacher.id);
    }
  }, [session?.status, session?.teacher?.id, teacher.id, onTeacherChange]);

  useEffect(() => {
    if (!session || session.status !== 'active' || secondsLeft > 0) return;
    setSession((current) => current ? { ...current, status: 'expired', messages: [], memory: null } : null);
    setStatus('This private lesson has ended. Session memory was cleared.');
    window.speechSynthesis?.cancel();
    speechRunRef.current += 1;
    recognitionRef.current?.abort();
  }, [secondsLeft, session?.id, session?.status]);

  useEffect(() => () => {
    recognitionRef.current?.abort();
    speechRunRef.current += 1;
    window.speechSynthesis?.cancel();
  }, []);

  function speakTeacher(text) {
    if (!voiceOutput || !speechSupported || !text) return;
    window.speechSynthesis.cancel();
    const segments = speechSegments(text);
    if (!segments.length) return;
    const run = speechRunRef.current + 1;
    speechRunRef.current = run;
    const settings = teacherVoiceProfile(session?.teacher || teacher);
    const language = navigator.language || 'en-US';
    const voice = selectTeacherVoice(
      window.speechSynthesis.getVoices(),
      session?.teacher || teacher,
      language,
      selectedVoiceUri,
    );
    const speakSegment = (index) => {
      if (speechRunRef.current !== run || index >= segments.length) return;
      const utterance = new window.SpeechSynthesisUtterance(segments[index]);
      utterance.rate = settings.rate;
      utterance.pitch = settings.pitch;
      utterance.lang = voice?.lang || language;
      if (voice) utterance.voice = voice;
      utterance.onend = () => speakSegment(index + 1);
      window.speechSynthesis.speak(utterance);
    };
    speakSegment(0);
  }

  async function startLesson() {
    if (!user || waiting) return;
    if (!selectedQuote) {
      setStatus('Enter a session length in 30-minute blocks.');
      return;
    }
    if (conversationMode === 'adult-companion' && (!adultConfirmed || !companionConsent)) {
      setStatus('Confirm 18+ and opt in to the AI companion roleplay first.');
      return;
    }
    setWaiting(true);
    setDurationMinutes(selectedQuote.durationMinutes);
    setStatus('Securing your private lesson...');
    if (!checkoutRef.current) checkoutRef.current = requestId();
    try {
      const data = await apiRequest('/api/virtual-lessons', {
        method: 'POST',
        body: JSON.stringify({
          durationMinutes: selectedQuote.durationMinutes,
          clientRequestId: checkoutRef.current,
          teacher: {
            id: teacher.id,
            name: teacher.name,
            title: teacher.title,
            style: teacher.style,
            voice: teacher.voice,
            voiceType: teacher.voiceType,
            requiresAdultConfirmation: Boolean(teacher.requiresAdultConfirmation),
          },
          conversationMode,
          conversationPreferences: {
            companionStyle,
            preferredName: user.name?.split(' ')[0] || '',
          },
          adultConfirmed: conversationMode === 'adult-companion' && adultConfirmed,
          companionConsent: conversationMode === 'adult-companion' && companionConsent,
        }),
      });
      checkoutRef.current = '';
      setSession(data.session);
      setNow(Date.now());
      if (data.user) setUser?.(data.user);
      setStatus(`${teacher.name} is ready. Your session memory lasts only for this session.`);
      speakTeacher(conversationMode === 'adult-companion'
        ? `Hi ${user.name?.split(' ')[0] || 'there'}. I'm ${teacher.name}, your virtual companion for this session. What kind of mood are you in?`
        : `Hi ${user.name?.split(' ')[0] || 'there'}. Tell me what you want to improve or talk about today.`);
    } catch (error) {
      if (error.details?.session) setSession(error.details.session);
      setStatus(error.message);
    } finally {
      setWaiting(false);
    }
  }

  async function sendMessage(rawMessage) {
    const message = String(rawMessage || '').trim();
    if (!message || waiting || !session || secondsLeft <= 0) return;
    setDraft('');
    setWaiting(true);
    setStatus('');
    try {
      const data = await apiRequest(`/api/virtual-lessons/${session.id}/messages`, {
        method: 'POST',
        body: JSON.stringify({ message, lessonContext, observations }),
      });
      setSession(data.session);
      if (data.action) onDemonstrate?.(data.action);
      speakTeacher(data.reply);
    } catch (error) {
      if (error.details?.session) setSession(error.details.session);
      const restored = Number(error.details?.recoveredSeconds || 0);
      setStatus(restored > 0
        ? `${error.message} ${restored} seconds were restored to your lesson.`
        : error.message);
    } finally {
      setWaiting(false);
    }
  }

  function submitChat(event) {
    event.preventDefault();
    sendMessage(draft);
  }

  function toggleListening() {
    if (!recognitionSupported || !session || waiting) return;
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    window.speechSynthesis?.cancel();
    const Recognition = recognitionEngine();
    const recognition = new Recognition();
    recognitionRef.current = recognition;
    voiceFinalRef.current = '';
    recognition.lang = navigator.language || 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      setListening(true);
      setStatus('Listening... Speak naturally. Your transcript will send when you stop.');
    };
    recognition.onresult = (event) => {
      let interim = '';
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index][0]?.transcript || '';
        if (event.results[index].isFinal) voiceFinalRef.current += ` ${transcript}`;
        else interim += ` ${transcript}`;
      }
      setDraft(`${voiceFinalRef.current} ${interim}`.trim());
    };
    recognition.onerror = (event) => {
      const denied = ['not-allowed', 'service-not-allowed'].includes(event.error);
      setStatus(denied
        ? 'Microphone permission was not granted. You can still type to your teacher.'
        : 'I could not hear that clearly. Tap the microphone and try again.');
      if (denied) voiceFinalRef.current = '';
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      const finalText = voiceFinalRef.current.trim();
      voiceFinalRef.current = '';
      if (finalText) sendMessage(finalText);
    };
    try {
      recognition.start();
    } catch {
      setListening(false);
      setStatus('Voice input could not start. You can still type to your teacher.');
    }
  }

  async function endLesson() {
    if (!session || waiting) return;
    const confirmed = window.confirm('End this lesson now? Remaining time is not refunded and session memory will be erased.');
    if (!confirmed) return;
    setWaiting(true);
    try {
      const data = await apiRequest(`/api/virtual-lessons/${session.id}/end`, {
        method: 'POST',
        body: JSON.stringify({}),
      });
      setSession(data.session);
      setStatus('Lesson ended. The conversation and session memory were erased.');
      window.speechSynthesis?.cancel();
      recognitionRef.current?.abort();
    } catch (error) {
      setStatus(error.message);
    } finally {
      setWaiting(false);
    }
  }

  if (!user) {
    return (
      <div className="virtual-lesson-gate">
        <TeacherPortrait teacher={teacher} />
        <div><strong>Private lesson with {teacher.name}</strong><small>Sign in to choose a timed voice lesson.</small></div>
        <button type="button" className="primary" onClick={() => onNavigate?.('account')}>Sign in</button>
      </div>
    );
  }

  if (loading) return <div className="virtual-lesson-loading" role="status">Checking your private lesson...</div>;

  if (!session || session.status !== 'active') {
    return (
      <div className="virtual-lesson-checkout">
        <header>
          <TeacherPortrait teacher={teacher} />
          <div><span>Private virtual session</span><strong>{teacher.name}</strong><small>Music expertise, open conversation, voice, and piano demonstrations.</small></div>
        </header>
        <div className="virtual-lesson-mode" role="radiogroup" aria-label="Choose conversation mode">
          <button
            type="button"
            role="radio"
            aria-checked={conversationMode === 'music-coach'}
            className={conversationMode === 'music-coach' ? 'is-selected' : ''}
            onClick={() => {
              setConversationMode('music-coach');
              setCompanionConsent(false);
            }}
          >
            <strong>Music teacher</strong>
            <small>Music-first; other topics are welcome.</small>
          </button>
          {teacher.requiresAdultConfirmation && (
            <button
              type="button"
              role="radio"
              aria-checked={conversationMode === 'adult-companion'}
              className={conversationMode === 'adult-companion' ? 'is-selected' : ''}
              onClick={() => setConversationMode('adult-companion')}
            >
              <strong>Flirty companion</strong>
              <small>Optional 18+ AI roleplay.</small>
            </button>
          )}
        </div>
        {conversationMode === 'adult-companion' && (
          <div className="virtual-companion-consent">
            <label>
              <span>Vibe</span>
              <select value={companionStyle} onChange={(event) => setCompanionStyle(event.target.value)}>
                <option value="gentle">Gentle</option>
                <option value="playful">Playful</option>
                <option value="confident">Confident</option>
              </select>
            </label>
            <label className="rights-check">
              <input
                type="checkbox"
                checked={adultConfirmed}
                onChange={(event) => {
                  const confirmed = event.target.checked;
                  setAdultConfirmed(confirmed);
                  if (confirmed) window.localStorage.setItem('polymath-teacher-adult-confirmed', 'true');
                  else window.localStorage.removeItem('polymath-teacher-adult-confirmed');
                }}
              />
              <span>I confirm I am 18 or older.</span>
            </label>
            <label className="rights-check">
              <input type="checkbox" checked={companionConsent} onChange={(event) => setCompanionConsent(event.target.checked)} />
              <span>I opt in to flirty AI companion roleplay for this session.</span>
            </label>
            <small>{teacher.name} is an AI character, not a human partner. You can stop flirting or change topics at any time.</small>
          </div>
        )}
        <div className="virtual-lesson-duration-stepper">
          <div className="virtual-lesson-duration-copy">
            <strong>Session length</strong>
            <small>Private sessions must use 30-minute blocks. Other numbers round to the nearest block.</small>
          </div>
          <div className="virtual-lesson-duration-control">
            <button
              type="button"
              aria-label="Subtract 30 minutes"
              onClick={() => setDurationMinutes(Math.max(
                Number(catalog.minimumDurationMinutes) || 30,
                (selectedQuote?.durationMinutes || Number(catalog.defaultDurationMinutes) || 60)
                  - (Number(catalog.durationStepMinutes) || 30),
              ))}
            >-</button>
            <label>
              <span className="sr-only">Private session minutes</span>
              <input
                type="number"
                inputMode="numeric"
                min={catalog.minimumDurationMinutes || 30}
                max={catalog.maximumDurationMinutes || 720}
                step={catalog.durationStepMinutes || 30}
                value={durationMinutes}
                onChange={(event) => setDurationMinutes(event.target.value)}
                onBlur={() => setDurationMinutes(
                  selectedQuote?.durationMinutes
                  || catalog.defaultDurationMinutes
                  || DEFAULT_CATALOG.defaultDurationMinutes,
                )}
              />
              <span>min</span>
            </label>
            <button
              type="button"
              aria-label="Add 30 minutes"
              onClick={() => setDurationMinutes(Math.min(
                Number(catalog.maximumDurationMinutes) || 720,
                (selectedQuote?.durationMinutes || Number(catalog.defaultDurationMinutes) || 60)
                  + (Number(catalog.durationStepMinutes) || 30),
              ))}
            >+</button>
          </div>
          <small className={selectedQuote?.rounded ? 'is-rounded' : ''} aria-live="polite">
            {selectedQuote?.rounded
              ? `${durationMinutes} minutes rounds to ${selectedQuote.durationMinutes} minutes.`
              : `US$${Number(catalog.pricePer30MinutesMcoins || 0).toFixed(2)} per 30 minutes.`}
          </small>
        </div>
        <div className="virtual-lesson-purchase-row">
          <span>
            <strong>{user.unlimitedMcoins ? 'Included' : `US$${Number(selectedQuote?.priceUsd || 0).toFixed(2)}`}</strong>
            <small>{user.unlimitedMcoins ? 'No Mcoins charged.' : `${selectedQuote?.priceMcoins || 0} Mcoins from your wallet.`} Memory clears when time ends.</small>
          </span>
          <button type="button" className="primary" disabled={waiting || !assistantAvailable || !selectedQuote} onClick={startLesson}>
            {waiting ? 'Starting...' : 'Start private session'}
          </button>
        </div>
        {!assistantAvailable && <p className="teacher-chat-status">Virtual lessons are not configured on this server. Nothing can be charged.</p>}
        {!user.unlimitedMcoins && Number(user.mcoins || 0) < Number(selectedQuote?.priceMcoins || 0) && (
          <button type="button" className="virtual-lesson-wallet-link" onClick={() => onNavigate?.('payment', { productId: 'mcoins-50' })}>Add Mcoins to wallet</button>
        )}
        {status && <p className="teacher-chat-status" role="status">{status}</p>}
      </div>
    );
  }

  return (
    <div className="virtual-lesson-room">
      <header className="virtual-lesson-live-header">
        <TeacherPortrait teacher={teacher} />
        <div>
          <span>{session.conversationMode === 'adult-companion' ? '18+ AI companion' : 'Private music session'} · Session memory on</span>
          <strong>{teacher.name} with {user.name?.split(' ')[0] || 'you'}</strong>
        </div>
        <time dateTime={session.expiresAt}>{formatLessonTime(secondsLeft)}</time>
      </header>

      <div className="virtual-lesson-actions" role="group" aria-label="Lesson voice controls">
        <button
          type="button"
          className={listening ? 'is-live' : ''}
          aria-pressed={listening}
          disabled={!recognitionSupported || waiting}
          onClick={toggleListening}
        >
          <span aria-hidden="true">{listening ? '■' : '●'}</span>
          {listening ? 'Stop and send' : 'Talk to teacher'}
        </button>
        <button
          type="button"
          className={voiceOutput ? 'is-on' : ''}
          aria-pressed={voiceOutput}
          disabled={!speechSupported}
          onClick={() => {
            const enabled = !voiceOutput;
            setVoiceOutput(enabled);
            if (!enabled) {
              speechRunRef.current += 1;
              window.speechSynthesis?.cancel();
            }
          }}
        >
          {voiceOutput ? 'Teacher voice on' : 'Teacher voice off'}
        </button>
      </div>

      {speechSupported && (
        <details className="virtual-lesson-voice-picker">
          <summary>Voice: {bestVoice?.name || 'device default'}</summary>
          <label>
            <span>Choose the most natural voice installed on this device</span>
            <select
              value={selectedVoiceUri}
              onChange={(event) => {
                const value = event.target.value;
                setSelectedVoiceUri(value);
                if (value) window.localStorage.setItem(`polymath-teacher-voice-${teacher.id}`, value);
                else window.localStorage.removeItem(`polymath-teacher-voice-${teacher.id}`);
              }}
            >
              <option value="">Best available automatically</option>
              {voiceChoices.map((voice) => (
                <option key={`${voice.voiceURI}-${voice.lang}`} value={voice.voiceURI}>{voice.name} · {voice.lang}</option>
              ))}
            </select>
          </label>
        </details>
      )}

      {!recognitionSupported && <p className="virtual-lesson-browser-note">Voice recognition is unavailable in this browser. Text chat and spoken teacher replies still work.</p>}

      <div className="teacher-chat-messages virtual-lesson-messages" aria-live="polite">
        {!messages.length && (
          <div className="virtual-lesson-welcome">
            <strong>{session.conversationMode === 'adult-companion'
              ? 'What do you feel like talking about?'
              : 'What do you want to improve or discuss today?'}</strong>
            <small>{teacher.name} remembers your answers until this timer ends.</small>
          </div>
        )}
        {messages.map((message) => (
          <p key={message.id} className={`teacher-message teacher-message-${message.role === 'assistant' ? 'teacher' : 'student'}`}>{message.text}</p>
        ))}
        {waiting && <p className="teacher-message teacher-message-thinking" aria-label="Teacher is thinking"><i /><i /><i /></p>}
      </div>

      <div className="virtual-lesson-prompts" aria-label="Quick lesson requests">
        {quickPrompts.map((prompt) => (
          <button type="button" key={prompt} disabled={waiting} onClick={() => sendMessage(prompt)}>{prompt}</button>
        ))}
      </div>

      <form className="teacher-chat-form virtual-lesson-chat-form" onSubmit={submitChat}>
        <label className="sr-only" htmlFor="virtual-lesson-message">Message {teacher.name}</label>
        <input
          id="virtual-lesson-message"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          maxLength="1600"
          placeholder={`Ask ${teacher.name} or tap the microphone...`}
        />
        <button type="submit" className="primary" disabled={waiting || !draft.trim()}>Send</button>
      </form>
      {status && <p className="teacher-chat-status" role="status">{status}</p>}
      <footer className="virtual-lesson-footer">
        <small>Microphone starts only when tapped. Browser speech recognition may use your browser's speech service. Polymath does not save raw microphone audio.</small>
        <button type="button" onClick={endLesson} disabled={waiting}>End lesson</button>
      </footer>
    </div>
  );
}
