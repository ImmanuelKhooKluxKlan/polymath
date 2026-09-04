import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

function RoomMark({ room }) {
  return <span className={`community-room-mark room-${room.visibility}`} aria-hidden="true">{room.visibility === 'global' ? '∞' : room.name.slice(0, 1).toUpperCase()}</span>;
}

function timeLabel(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

export default function CommunityPage({ user, onNavigate }) {
  const [rooms, setRooms] = useState([]);
  const [activeRoomId, setActiveRoomId] = useState('');
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [groupForm, setGroupForm] = useState({ name: '', topic: '', visibility: 'public' });
  const [inviteCode, setInviteCode] = useState('');
  const streamRef = useRef(null);
  const activeRoomIdRef = useRef('');
  const latestMessageAtRef = useRef(new Map());
  const activeRoom = useMemo(
    () => rooms.find((room) => room.id === activeRoomId) || rooms[0] || null,
    [rooms, activeRoomId],
  );

  async function loadRooms({ quiet = false } = {}) {
    if (!user?.access?.community) return;
    if (!quiet) setLoading(true);
    try {
      const data = await apiRequest('/api/community/rooms');
      setRooms(data.rooms || []);
      setActiveRoomId((current) => (
        data.rooms?.some((room) => room.id === current) ? current : data.rooms?.[0]?.id || ''
      ));
      setStatus('');
    } catch (error) {
      setStatus(error.message);
    } finally {
      if (!quiet) setLoading(false);
    }
  }

  async function loadMessages(roomId, { quiet = false, incremental = false } = {}) {
    if (!roomId) return;
    try {
      const since = incremental ? latestMessageAtRef.current.get(roomId) : '';
      const query = since ? `?since=${encodeURIComponent(since)}` : '';
      const data = await apiRequest(`/api/community/rooms/${encodeURIComponent(roomId)}/messages${query}`);
      if (activeRoomIdRef.current !== roomId) return;
      const incoming = data.messages || [];
      setMessages((current) => {
        if (!incremental) return incoming;
        const byId = new Map(current.map((message) => [message.id, message]));
        incoming.forEach((message) => byId.set(message.id, message));
        return [...byId.values()]
          .sort((left, right) => String(left.createdAt).localeCompare(String(right.createdAt)))
          .slice(-250);
      });
      const latest = incoming[incoming.length - 1]?.createdAt;
      if (latest) latestMessageAtRef.current.set(roomId, latest);
      if (!quiet) setStatus('');
    } catch (error) {
      if (!quiet) setStatus(error.message);
    }
  }

  useEffect(() => { loadRooms(); }, [user?.user_id, user?.access?.community]);

  useEffect(() => {
    if (!activeRoom?.id || !user?.access?.community) return undefined;
    activeRoomIdRef.current = activeRoom.id;
    setMessages([]);
    loadMessages(activeRoom.id);
    let messagePolls = 0;
    const messageTimer = window.setInterval(() => {
      messagePolls += 1;
      // A full refresh every minute also removes messages deleted by moderators.
      loadMessages(activeRoom.id, { quiet: true, incremental: messagePolls % 24 !== 0 });
    }, 2500);
    const roomTimer = window.setInterval(() => loadRooms({ quiet: true }), 15000);
    return () => {
      window.clearInterval(messageTimer);
      window.clearInterval(roomTimer);
    };
  }, [activeRoom?.id, user?.user_id, user?.access?.community]);

  useEffect(() => {
    if (!streamRef.current) return;
    streamRef.current.scrollTop = streamRef.current.scrollHeight;
  }, [messages.length, activeRoom?.id]);

  async function sendMessage(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || !activeRoom || sending) return;
    setSending(true);
    try {
      const data = await apiRequest(`/api/community/rooms/${encodeURIComponent(activeRoom.id)}/messages`, {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      setMessages((current) => [...current, data.message]);
      setDraft('');
      setStatus('');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setSending(false);
    }
  }

  async function createGroup(event) {
    event.preventDefault();
    try {
      const data = await apiRequest('/api/community/rooms', {
        method: 'POST',
        body: JSON.stringify(groupForm),
      });
      setGroupForm({ name: '', topic: '', visibility: 'public' });
      await loadRooms({ quiet: true });
      setActiveRoomId(data.room.id);
      setStatus(data.room.inviteCode ? `Group ready. Invite code: ${data.room.inviteCode}` : 'Group ready.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function joinGroup(roomId, code = '') {
    try {
      const data = await apiRequest('/api/community/rooms/join', {
        method: 'POST',
        body: JSON.stringify(code ? { inviteCode: code } : { roomId }),
      });
      setInviteCode('');
      await loadRooms({ quiet: true });
      setActiveRoomId(data.room.id);
      setStatus('You joined the group.');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function deleteMessage(message) {
    try {
      await apiRequest(`/api/community/rooms/${encodeURIComponent(activeRoom.id)}/messages/${encodeURIComponent(message.id)}`, { method: 'DELETE' });
      setMessages((current) => current.filter((candidate) => candidate.id !== message.id));
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function reportMessage(message) {
    try {
      const data = await apiRequest(`/api/community/messages/${encodeURIComponent(message.id)}/report`, {
        method: 'POST',
        body: JSON.stringify({ reason: 'Please review this community message.' }),
      });
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    }
  }

  if (!user) {
    return (
      <section className="page-shell narrow-page community-gate">
        <span className="community-gate-mark" aria-hidden="true">∞</span>
        <p className="eyebrow">Community</p>
        <h1>Meet inside Polymath.</h1>
        <p>Sign in first. Community chat is included with Chill and Musician.</p>
        <button type="button" className="primary" onClick={() => onNavigate('account', { next: 'community' })}>Sign in</button>
      </section>
    );
  }

  if (!user.access?.community) {
    return (
      <section className="page-shell narrow-page community-gate">
        <span className="community-gate-mark" aria-hidden="true">∞</span>
        <p className="eyebrow">Community</p>
        <h1>Free Flow is for members.</h1>
        <p>Chill and Musician members can join the live room and create public or invite-only groups.</p>
        <button type="button" className="primary" onClick={() => onNavigate('payment', { productId: 'polymath-chill-monthly' })}>See membership</button>
      </section>
    );
  }

  return (
    <section className="page-shell community-page">
      <header className="community-heading">
        <div><p className="eyebrow">Community</p><h1>Free Flow</h1><p>Talk, meet, collaborate.</p></div>
        <span className="community-live"><i /> Live</span>
      </header>

      <div className="community-layout">
        <aside className="community-sidebar">
          <div className="community-sidebar-title"><strong>Rooms</strong><small>{rooms.length}</small></div>
          <div className="community-room-list">
            {rooms.map((room) => (
              <button type="button" key={room.id} className={activeRoom?.id === room.id ? 'is-active' : ''} onClick={() => setActiveRoomId(room.id)}>
                <RoomMark room={room} />
                <span><strong>{room.name}</strong><small>{room.visibility === 'private' ? 'Invite only' : room.memberCount === null ? 'Everyone together' : `${room.memberCount} member${room.memberCount === 1 ? '' : 's'}`}</small></span>
              </button>
            ))}
          </div>

          <details className="community-action-disclosure">
            <summary>New group</summary>
            <form onSubmit={createGroup}>
              <input required minLength="2" maxLength="60" value={groupForm.name} onChange={(event) => setGroupForm((current) => ({ ...current, name: event.target.value }))} placeholder="Group name" />
              <input maxLength="180" value={groupForm.topic} onChange={(event) => setGroupForm((current) => ({ ...current, topic: event.target.value }))} placeholder="What is it about?" />
              <select value={groupForm.visibility} onChange={(event) => setGroupForm((current) => ({ ...current, visibility: event.target.value }))}>
                <option value="public">Public group</option>
                <option value="private">Invite only</option>
              </select>
              <button type="submit" className="primary">Create</button>
            </form>
          </details>

          <details className="community-action-disclosure">
            <summary>Use invite code</summary>
            <form onSubmit={(event) => { event.preventDefault(); joinGroup('', inviteCode); }}>
              <input required maxLength="20" value={inviteCode} onChange={(event) => setInviteCode(event.target.value.toUpperCase())} placeholder="Invite code" />
              <button type="submit">Join</button>
            </form>
          </details>
        </aside>

        <section className="community-conversation">
          {activeRoom ? (
            <>
              <header className="community-room-header">
                <RoomMark room={activeRoom} />
                <div><h2>{activeRoom.name}</h2><p>{activeRoom.topic}</p></div>
                {activeRoom.inviteCode && <button type="button" className="community-code" onClick={() => navigator.clipboard?.writeText(activeRoom.inviteCode)} title="Copy invite code">{activeRoom.inviteCode}</button>}
              </header>
              <div ref={streamRef} className="community-message-stream" aria-live="polite">
                {messages.map((message) => (
                  <article key={message.id} className={`community-message ${message.mine ? 'is-mine' : ''}`}>
                    <span className="community-avatar" aria-hidden="true">
                      {message.author.avatarUrl
                        ? <img src={message.author.avatarUrl} alt="" loading="lazy" referrerPolicy="no-referrer" />
                        : message.author.name.slice(0, 1).toUpperCase()}
                    </span>
                    <div>
                      <header><strong>{message.mine ? 'You' : message.author.name}</strong><time>{timeLabel(message.createdAt)}</time></header>
                      <p>{message.text}</p>
                      <footer>
                        {!message.mine && <button type="button" onClick={() => reportMessage(message)}>Report</button>}
                        {message.canDelete && <button type="button" onClick={() => deleteMessage(message)}>Remove</button>}
                      </footer>
                    </div>
                  </article>
                ))}
                {!messages.length && !loading && <div className="community-empty"><span>♪</span><strong>Start the conversation.</strong><small>Be curious. Be kind. Keep private information private.</small></div>}
              </div>
              {!activeRoom.joined ? (
                <div className="community-join-bar"><span>Join to speak in this group.</span><button type="button" className="primary" onClick={() => joinGroup(activeRoom.id)}>Join group</button></div>
              ) : (
                <form className="community-composer" onSubmit={sendMessage}>
                  <label className="sr-only" htmlFor="community-message">Message {activeRoom.name}</label>
                  <textarea id="community-message" rows="1" maxLength="1200" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={`Message ${activeRoom.name}`} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} />
                  <button type="submit" className="primary" disabled={sending || !draft.trim()}>{sending ? '…' : 'Send'}</button>
                </form>
              )}
            </>
          ) : <div className="community-empty"><strong>{loading ? 'Opening community…' : 'No rooms available.'}</strong></div>}
        </section>
      </div>
      {status && <button type="button" className="community-status" onClick={() => setStatus('')}>{status}</button>}
    </section>
  );
}
