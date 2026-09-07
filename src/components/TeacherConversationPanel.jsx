import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

const INITIAL_MESSAGE = {
  role: 'assistant',
  content: 'I’m ready. Start my eyes and ears, choose a part of the song, or ask me a question.',
};

function chooseEnglishVoice() {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  const english = voices.filter((voice) => /^en(?:-|_)/i.test(voice.lang));
  const preferred = ['Samantha', 'Ava', 'Aria', 'Zira', 'Sonia', 'Jenny', 'Female'];
  return preferred.map((name) => english.find((voice) => voice.name.includes(name))).find(Boolean)
    || english.find((voice) => voice.localService)
    || english[0]
    || voices[0]
    || null;
}

export default function TeacherConversationPanel({
  song,
  currentTime = 0,
  observations = [],
  sensesRef,
  onSpeakingChange,
}) {
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [draft, setDraft] = useState('');
  const [capabilities, setCapabilities] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [voiceReplies, setVoiceReplies] = useState(true);
  const [listening, setListening] = useState(false);
  const [lastScene, setLastScene] = useState(null);
  const messageEndRef = useRef(null);
  const coachedSignalRef = useRef('');

  useEffect(() => {
    let cancelled = false;
    apiRequest('/api/teacher/capabilities')
      .then((result) => {
        if (!cancelled) setCapabilities(result);
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [messages, busy]);

  useEffect(() => () => {
    window.speechSynthesis?.cancel?.();
    onSpeakingChange?.(false);
  }, [onSpeakingChange]);

  function speak(text) {
    if (!voiceReplies || !window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new window.SpeechSynthesisUtterance(text.replace(/[*_#]/g, ''));
    utterance.voice = chooseEnglishVoice();
    utterance.rate = 0.98;
    utterance.pitch = 1.03;
    utterance.onstart = () => onSpeakingChange?.(true);
    utterance.onend = () => onSpeakingChange?.(false);
    utterance.onerror = () => onSpeakingChange?.(false);
    window.speechSynthesis.speak(utterance);
  }

  const latestCorrection = useMemo(
    () => [...observations].reverse().find((entry) => entry.type && entry.type !== 'correct'),
    [observations],
  );

  useEffect(() => {
    if (!latestCorrection) return;
    const identity = `${latestCorrection.type}:${latestCorrection.note}:${latestCorrection.startedSongTime}:${latestCorrection.observedDuration}`;
    if (coachedSignalRef.current === identity) return;
    coachedSignalRef.current = identity;
    setMessages((current) => [...current.slice(-28), {
      role: 'assistant',
      content: latestCorrection.message,
      measured: true,
    }]);
    speak(latestCorrection.message);
  }, [latestCorrection]);

  function lessonContext() {
    return {
      title: song?.title || 'Current lesson',
      composer: song?.composer || '',
      currentTime: Math.round(Number(currentTime || 0) * 100) / 100,
      duration: Math.round(Number(song?.duration || 0) * 100) / 100,
    };
  }

  async function requestReply(nextMessages, scene = lastScene) {
    const response = await apiRequest('/api/teacher/chat', {
      method: 'POST',
      body: JSON.stringify({
        messages: nextMessages.map(({ role, content }) => ({ role, content })),
        lessonContext: lessonContext(),
        observations: observations.slice(-10),
        scene,
      }),
    });
    const reply = { role: 'assistant', content: response.reply };
    setMessages((current) => [...current.slice(-28), reply]);
    speak(response.reply);
  }

  async function sendMessage(explicitText) {
    const text = String(explicitText ?? draft).trim();
    if (!text || busy) return;
    const userMessage = { role: 'user', content: text };
    const nextMessages = [...messages.slice(-14), userMessage];
    setMessages((current) => [...current.slice(-28), userMessage]);
    setDraft('');
    setError('');
    setBusy(true);
    try {
      await requestReply(nextMessages);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function shareScene() {
    if (busy) return;
    setError('');
    setBusy(true);
    const prompt = draft.trim() || 'What can you clearly see?';
    const userMessage = { role: 'user', content: prompt };
    setMessages((current) => [...current.slice(-28), userMessage]);
    setDraft('');
    try {
      const imageDataUrl = sensesRef.current?.captureSnapshot?.();
      if (!imageDataUrl) throw new Error('Start the teacher camera before asking her to look.');
      const scene = await apiRequest('/api/teacher/scene', {
        method: 'POST',
        body: JSON.stringify({ imageDataUrl, prompt }),
      });
      setLastScene(scene);
      await requestReply([...messages.slice(-14), userMessage], scene);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function startSpeechInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech input is not supported in this browser. You can still type to the teacher.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = navigator.language || 'en-US';
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onstart = () => {
      setError('');
      setListening(true);
    };
    recognition.onresult = (event) => setDraft(event.results?.[0]?.[0]?.transcript || '');
    recognition.onerror = (event) => setError(event.error === 'not-allowed'
      ? 'Microphone speech permission was blocked.'
      : 'I could not understand that. Please try again.');
    recognition.onend = () => setListening(false);
    recognition.start();
  }

  const conversationAvailable = capabilities?.conversation !== false;
  const sceneAvailable = Boolean(capabilities?.generalSceneVision);

  return (
    <section className="teacher-conversation" aria-label="Talk with virtual teacher">
      <div className="teacher-conversation-heading">
        <div>
          <p className="eyebrow">Session-only memory</p>
          <h3>Talk with your teacher</h3>
        </div>
        <label className="teacher-voice-toggle">
          <input
            type="checkbox"
            checked={voiceReplies}
            onChange={(event) => {
              setVoiceReplies(event.target.checked);
              if (!event.target.checked) {
                window.speechSynthesis?.cancel?.();
                onSpeakingChange?.(false);
              }
            }}
          />
          Voice
        </label>
      </div>

      <div className="teacher-chat-log" role="log" aria-live="polite">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`teacher-chat-message is-${message.role}`}>
            <small>{message.role === 'assistant' ? message.measured ? 'Teacher · measured' : 'Teacher' : 'You'}</small>
            <p>{message.content}</p>
          </div>
        ))}
        {busy && <div className="teacher-chat-thinking">Teacher is thinking…</div>}
        <span ref={messageEndRef} />
      </div>

      {error && <p className="teacher-chat-error" role="alert">{error}</p>}
      {!conversationAvailable && (
        <p className="teacher-chat-notice">Conversation needs the Chat Boss endpoint. Measured piano corrections still work locally.</p>
      )}

      <form
        className="teacher-chat-compose"
        onSubmit={(event) => {
          event.preventDefault();
          sendMessage();
        }}
      >
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask about the lesson or an item…"
          maxLength={2000}
          aria-label="Message to virtual teacher"
        />
        <button type="button" className={listening ? 'active' : 'ghost'} onClick={startSpeechInput} disabled={busy}>
          {listening ? 'Listening…' : 'Talk'}
        </button>
        <button
          type="button"
          className="ghost"
          onClick={shareScene}
          disabled={busy || !sceneAvailable}
          title={sceneAvailable ? 'Share one camera snapshot with the teacher' : 'Connect a vision-language model to enable general object recognition'}
        >
          Look
        </button>
        <button type="submit" className="primary" disabled={busy || !draft.trim() || !conversationAvailable}>Send</button>
      </form>

      <p className="teacher-chat-privacy">
        {sceneAvailable
          ? 'Look sends one snapshot only when pressed. Chat stays in this browser session. Browser speech input may use your browser provider.'
          : 'Keyboard vision is local. General object vision is awaiting a separate vision-model connection. Browser speech input may use your browser provider.'}
      </p>
    </section>
  );
}
