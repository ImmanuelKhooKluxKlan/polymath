import { useEffect, useState } from 'react';
import { apiRequest } from '../services/api.js';

const DEFAULT_SUPPORT = Object.freeze({
  unlimited: false,
  dailyLimit: 7,
  usedQuestions: 0,
  remainingQuestions: 7,
  resetsAt: '',
  contact: { email: '', phone: '' },
});

function resetTimeLabel(value) {
  if (!value) return 'the next daily reset';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'the next daily reset';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function phoneHref(value) {
  const display = String(value || '').trim();
  const prefix = display.startsWith('+') ? '+' : '';
  return `tel:${prefix}${display.replace(/\D/g, '')}`;
}

export default function SupportAssistant({ user }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [waiting, setWaiting] = useState(false);
  const [status, setStatus] = useState('');
  const [support, setSupport] = useState(DEFAULT_SUPPORT);

  useEffect(() => {
    if (!user) return undefined;
    let cancelled = false;
    apiRequest('/api/assistant/capabilities')
      .then((data) => {
        if (!cancelled && data.support) setSupport(data.support);
      })
      .catch((error) => {
        if (!cancelled) setStatus(error.message);
      });
    return () => { cancelled = true; };
  }, [user?.user_id, open]);

  useEffect(() => {
    if (!support.resetsAt || support.unlimited) return undefined;
    const resetAt = new Date(support.resetsAt).getTime();
    if (!Number.isFinite(resetAt)) return undefined;
    const delay = Math.max(1000, Math.min(2147483647, resetAt - Date.now() + 1000));
    const timer = window.setTimeout(() => {
      apiRequest('/api/assistant/capabilities')
        .then((data) => { if (data.support) setSupport(data.support); })
        .catch(() => {});
    }, delay);
    return () => window.clearTimeout(timer);
  }, [support.resetsAt, support.unlimited]);

  if (!user) return null;

  const exhausted = !support.unlimited && Number(support.remainingQuestions) <= 0;
  const remainingLabel = support.unlimited
    ? 'Unlimited Help'
    : `${Math.max(0, Number(support.remainingQuestions) || 0)}/${support.dailyLimit || 7} left today`;
  const contact = support.contact || DEFAULT_SUPPORT.contact;
  const hasContact = Boolean(contact.email || contact.phone);

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || waiting || exhausted) return;
    const next = [...messages, { role: 'user', content: text }].slice(-12);
    setMessages(next);
    setDraft('');
    setWaiting(true);
    setStatus('Polymath Support may need a moment to wake up.');
    try {
      const data = await apiRequest('/api/assistant/support', {
        method: 'POST',
        body: JSON.stringify({ messages: next }),
      });
      setMessages((current) => [...current, { role: 'assistant', content: data.reply }].slice(-12));
      if (data.support) setSupport(data.support);
      setStatus('');
    } catch (error) {
      if (error.details?.support) setSupport(error.details.support);
      setStatus(error.message);
    } finally {
      setWaiting(false);
    }
  }

  return (
    <aside className={`support-assistant ${open ? 'is-open' : ''}`} aria-label="Polymath customer support">
      {open && (
        <section className="support-assistant-panel">
          <header>
            <span className="support-orb" aria-hidden="true">P</span>
            <div><strong>Polymath Support</strong><small>{remainingLabel}</small></div>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close support">×</button>
          </header>
          <div className="support-message-stream" aria-live="polite">
            {!messages.length && <p className="support-welcome">Hi {user.name?.split(' ')[0] || 'there'}. What can I help you with?</p>}
            {messages.map((message, index) => <p key={`${message.role}-${index}`} className={`support-message is-${message.role}`}>{message.content}</p>)}
            {waiting && <p className="support-thinking"><i /><i /><i /></p>}
          </div>
          {exhausted ? (
            <div className="support-limit-card" role="status">
              <strong>Daily Help limit reached</strong>
              <p>Your 7 questions reset at {resetTimeLabel(support.resetsAt)}.</p>
              {hasContact ? (
                <div>
                  {contact.phone && <a href={phoneHref(contact.phone)}>Call {contact.phone}</a>}
                  {contact.email && <a href={`mailto:${contact.email}`}>Email {contact.email}</a>}
                </div>
              ) : <small>The administrator has not published helpline details yet.</small>}
            </div>
          ) : (
            <form onSubmit={send}>
              <input value={draft} onChange={(event) => setDraft(event.target.value)} maxLength="1600" placeholder="Ask for help..." aria-label="Help question" />
              <button type="submit" className="primary" disabled={waiting || !draft.trim()}>Send</button>
            </form>
          )}
          {status && <p className="support-status">{status}</p>}
          <small className="support-privacy">Never share passwords, OTPs, private keys, or card details.</small>
        </section>
      )}
      <button type="button" className="support-assistant-trigger" onClick={() => setOpen((current) => !current)} aria-expanded={open}>
        <span aria-hidden="true">?</span>{open ? 'Close' : 'Help'}
      </button>
    </aside>
  );
}
