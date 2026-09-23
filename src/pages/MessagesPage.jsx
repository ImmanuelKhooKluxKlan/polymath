import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';

const SAVED_INVITE_KEY = 'polymath_human_lesson_invite';

function safeContact(contact) {
  return {
    user_id: String(contact?.user_id || ''),
    name: String(contact?.name || 'Polymath member').trim() || 'Polymath member',
  };
}

function newRequestId() {
  return window.crypto?.randomUUID?.() || `transfer_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export default function MessagesPage({ user, setUser, initialUser, context, onNavigate }) {
  const [threads, setThreads] = useState([]);
  const [activeUser, setActiveUser] = useState(initialUser ? safeContact(initialUser) : null);
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [status, setStatus] = useState('');
  const [showTransfer, setShowTransfer] = useState(false);
  const [transferAmount, setTransferAmount] = useState('30');
  const [minimumTransfer, setMinimumTransfer] = useState(30);
  const [sendingTransfer, setSendingTransfer] = useState(false);

  async function loadThreads() {
    if (!user) return;
    try {
      const data = await apiRequest('/api/messages/threads');
      const nextThreads = (data.threads || [])
        .filter((thread) => thread?.otherUser?.user_id)
        .map((thread) => ({
          ...thread,
          otherUser: safeContact(thread.otherUser),
          lastMessage: { text: 'Private message', createdAt: new Date(0).toISOString(), ...(thread.lastMessage || {}) },
        }));
      setThreads(nextThreads);
      if (!activeUser && nextThreads[0]) setActiveUser(nextThreads[0].otherUser);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function loadConversation(otherUser) {
    if (!user || !otherUser?.user_id) return;
    try {
      const data = await apiRequest(`/api/messages/${otherUser.user_id}`);
      setMessages(Array.isArray(data.messages) ? data.messages : []);
      if (data.otherUser) setActiveUser(safeContact(data.otherUser));
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadThreads(); }, [user?.user_id]);
  useEffect(() => { if (initialUser) setActiveUser(safeContact(initialUser)); }, [initialUser?.user_id]);
  useEffect(() => {
    setMessages([]);
    setShowTransfer(false);
    loadConversation(activeUser);
  }, [activeUser?.user_id, user?.user_id]);
  useEffect(() => {
    if (!user) return undefined;
    apiRequest('/api/human-lessons/config')
      .then((data) => {
        const minimum = Number(data.minimumTransferMcoins || 30);
        setMinimumTransfer(minimum);
        setTransferAmount(String(minimum));
      })
      .catch(() => {});
    const refresh = window.setInterval(() => {
      loadThreads();
      if (activeUser?.user_id) loadConversation(activeUser);
    }, 5000);
    return () => window.clearInterval(refresh);
  }, [user?.user_id, activeUser?.user_id]);

  const mergedThreads = useMemo(() => {
    if (!activeUser) return threads;
    if (threads.some((thread) => thread.otherUser.user_id === activeUser.user_id)) return threads;
    return [{ otherUser: activeUser, lastMessage: { text: 'Start a new conversation', createdAt: new Date(0).toISOString() } }, ...threads];
  }, [threads, activeUser]);

  async function send(event) {
    event.preventDefault();
    if (!text.trim() || !activeUser) return;
    try {
      await apiRequest('/api/messages', {
        method: 'POST',
        body: JSON.stringify({ toUserId: activeUser.user_id, text }),
      });
      setText('');
      await loadConversation(activeUser);
      await loadThreads();
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function transferMcoins(event) {
    event.preventDefault();
    if (!activeUser?.user_id || sendingTransfer) return;
    const amountMcoins = Number(transferAmount);
    if (!Number.isFinite(amountMcoins) || amountMcoins < minimumTransfer) {
      setStatus(`Minimum transfer is ${minimumTransfer} Mcoins.`);
      return;
    }
    if (!window.confirm(`Transfer ${amountMcoins.toFixed(2)} Mcoins to ${activeUser.name}?`)) return;
    setSendingTransfer(true);
    setStatus('Securing transfer…');
    try {
      const data = await apiRequest('/api/messages/transfers', {
        method: 'POST',
        body: JSON.stringify({
          toUserId: activeUser.user_id,
          amountMcoins,
          clientRequestId: newRequestId(),
        }),
      });
      setUser?.(data.user);
      setShowTransfer(false);
      setTransferAmount(String(minimumTransfer));
      setStatus(`${amountMcoins.toFixed(2)} Mcoins sent to ${activeUser.name}.`);
      await loadConversation(activeUser);
      await loadThreads();
    } catch (error) {
      setStatus(error.message);
    } finally {
      setSendingTransfer(false);
    }
  }

  function joinInvitedLesson(invite) {
    if (!invite?.meetingId || !invite?.accessCode) {
      setStatus('This invitation is missing its private credentials. Ask the teacher to resend it.');
      return;
    }
    window.sessionStorage.setItem(SAVED_INVITE_KEY, JSON.stringify({
      meetingId: invite.meetingId,
      accessCode: invite.accessCode,
      autoJoin: true,
      isHost: false,
    }));
    onNavigate('lesson-room', { meetingId: invite.meetingId });
  }

  if (!user) {
    return (
      <section className="page-shell narrow-page empty-state-page">
        <div className="empty-state">
          <h1>Sign in to view messages.</h1>
          <p>Your private conversations are stored under your account.</p>
          <button className="primary" type="button" onClick={() => onNavigate('account')}>Open account</button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-shell messages-page">
      <div className="page-heading">
        <p className="eyebrow">Messages</p>
        <h1>Private conversations.</h1>
      </div>
      <div className="chat-layout">
        <aside className="thread-list">
          <div className="thread-list-title"><strong>Chats</strong><small>{mergedThreads.length}</small></div>
          {mergedThreads.map((thread) => (
            <button
              key={thread.otherUser.user_id}
              className={activeUser?.user_id === thread.otherUser.user_id ? 'active' : ''}
              type="button"
              onClick={() => setActiveUser(thread.otherUser)}
            >
              <span className="avatar">{thread.otherUser.name.slice(0, 1).toUpperCase()}</span>
              <span><strong>{thread.otherUser.name}</strong><small>{String(thread.lastMessage?.text || 'Private message')}</small></span>
            </button>
          ))}
          {!mergedThreads.length && <p className="muted thread-empty">Start from a teacher or composer profile.</p>}
        </aside>
        <section className="conversation-panel">
          {activeUser ? (
            <>
              <header className="conversation-header">
                <span className="avatar">{activeUser.name.slice(0, 1).toUpperCase()}</span>
                <div><strong>{activeUser.name}</strong><small>{context === 'teacher' ? 'Teacher conversation' : 'Private conversation'}</small></div>
                <button className={showTransfer ? 'active' : ''} type="button" onClick={() => setShowTransfer((current) => !current)}>Send Mcoins</button>
              </header>
              <div className="message-stream">
                {messages.map((message) => (
                  <div key={message.id || `${message.createdAt}-${message.text}`} className={`message-bubble ${message.fromUserId === user.user_id ? 'mine' : ''} ${message.kind || ''}`}>
                    {message.kind === 'mcoin-transfer' && <span className="message-kind-label">Mcoin transfer</span>}
                    {message.kind === 'lesson-invite' && <span className="message-kind-label">Private lesson invite</span>}
                    <p>{String(message.text || 'Private message')}</p>
                    {message.lessonInvite && <button className="message-invite-join" type="button" onClick={() => joinInvitedLesson(message.lessonInvite)}>Join {message.lessonInvite.roomName || 'lesson'}</button>}
                    <small>{new Date(message.createdAt).toLocaleString()}</small>
                  </div>
                ))}
                {!messages.length && <div className="conversation-empty">Introduce yourself and ask your first question.</div>}
              </div>
              {showTransfer && (
                <form className="message-transfer-panel" onSubmit={transferMcoins}>
                  <label className="field">Mcoins to send<input type="number" min={minimumTransfer} max="1000000" step="0.01" value={transferAmount} onChange={(event) => setTransferAmount(event.target.value)} required /><small>Minimum {minimumTransfer} Mcoins · no added transfer fee</small></label>
                  <button className="primary" type="submit" disabled={sendingTransfer}>{sendingTransfer ? 'Sending…' : `Transfer to ${activeUser.name}`}</button>
                </form>
              )}
              <form className="message-composer" onSubmit={send}>
                <input value={text} onChange={(event) => setText(event.target.value)} placeholder={`Message ${activeUser.name}`} />
                <button className="primary" type="submit">Send</button>
              </form>
            </>
          ) : (
            <div className="conversation-empty large">Choose a conversation.</div>
          )}
        </section>
      </div>
      {status && <p className="form-status floating-status">{status}</p>}
    </section>
  );
}
