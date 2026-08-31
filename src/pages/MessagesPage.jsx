import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';

export default function MessagesPage({ user, initialUser, context, onNavigate }) {
  const [threads, setThreads] = useState([]);
  const [activeUser, setActiveUser] = useState(initialUser || null);
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [status, setStatus] = useState('');

  async function loadThreads() {
    if (!user) return;
    try {
      const data = await apiRequest('/api/messages/threads');
      setThreads(data.threads);
      if (!activeUser && data.threads[0]) setActiveUser(data.threads[0].otherUser);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function loadConversation(otherUser) {
    if (!user || !otherUser?.user_id) return;
    try {
      const data = await apiRequest(`/api/messages/${otherUser.user_id}`);
      setMessages(data.messages);
      if (data.otherUser) setActiveUser(data.otherUser);
    } catch (error) {
      setStatus(error.message);
    }
  }

  useEffect(() => { loadThreads(); }, [user?.user_id]);
  useEffect(() => { if (initialUser) setActiveUser(initialUser); }, [initialUser?.user_id]);
  useEffect(() => { loadConversation(activeUser); }, [activeUser?.user_id, user?.user_id]);

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
              <span><strong>{thread.otherUser.name}</strong><small>{thread.lastMessage.text}</small></span>
            </button>
          ))}
          {!mergedThreads.length && <p className="muted thread-empty">Start from a teacher or composer profile.</p>}
        </aside>
        <section className="conversation-panel">
          {activeUser ? (
            <>
              <header className="conversation-header"><span className="avatar">{activeUser.name.slice(0, 1).toUpperCase()}</span><div><strong>{activeUser.name}</strong><small>{context === 'teacher' ? 'Teacher conversation' : 'Private conversation'}</small></div></header>
              <div className="message-stream">
                {messages.map((message) => (
                  <div key={message.id} className={`message-bubble ${message.fromUserId === user.user_id ? 'mine' : ''}`}>
                    <p>{message.text}</p>
                    <small>{new Date(message.createdAt).toLocaleString()}</small>
                  </div>
                ))}
                {!messages.length && <div className="conversation-empty">Introduce yourself and ask your first question.</div>}
              </div>
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
