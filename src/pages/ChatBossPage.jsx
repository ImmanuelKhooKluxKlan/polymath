import { useEffect, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';
import { userFacingError } from '../utils/userFacingError.js';
import TaskProgress from '../components/TaskProgress.jsx';

const HISTORY_KEY = 'polymath_chat_boss_history_v1';
const ACTIVE_JOB_KEY = 'polymath_chat_boss_active_job_v1';
const MAX_SAVED_MESSAGES = 60;

function readSavedHistory() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || '[]');
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((message) => ['user', 'assistant'].includes(message?.role) && typeof message?.content === 'string')
      .slice(-MAX_SAVED_MESSAGES);
  } catch {
    return [];
  }
}

function saveHistory(messages) {
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-MAX_SAVED_MESSAGES)));
}

function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      window.clearTimeout(timer);
      reject(new window.DOMException('Aborted', 'AbortError'));
    }, { once: true });
  });
}

export default function ChatBossPage({ user, onNavigate }) {
  const [messages, setMessages] = useState(readSavedHistory);
  const [draft, setDraft] = useState('');
  const [capabilities, setCapabilities] = useState(null);
  const [loadingCapabilities, setLoadingCapabilities] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState(() => window.localStorage.getItem(ACTIVE_JOB_KEY) || '');
  const [runStatus, setRunStatus] = useState(jobId ? 'IN_QUEUE' : '');
  const [error, setError] = useState('');
  const transcriptRef = useRef(null);
  const busy = submitting || Boolean(jobId);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, runStatus]);

  useEffect(() => {
    if (!user?.admin) return undefined;
    let cancelled = false;
    setLoadingCapabilities(true);
    apiRequest('/api/chat-boss/capabilities')
      .then((data) => {
        if (!cancelled) setCapabilities(data);
      })
      .catch((requestError) => {
        if (!cancelled) setError(userFacingError(requestError, 'This workspace is temporarily unavailable.'));
      })
      .finally(() => {
        if (!cancelled) setLoadingCapabilities(false);
      });
    return () => { cancelled = true; };
  }, [user?.admin]);

  useEffect(() => {
    if (!user?.admin || !jobId) return undefined;
    const controller = new window.AbortController();
    let consecutiveErrors = 0;

    async function poll() {
      while (!controller.signal.aborted) {
        try {
          const job = await apiRequest(`/api/chat-boss/jobs/${encodeURIComponent(jobId)}`, {
            signal: controller.signal,
          });
          consecutiveErrors = 0;
          setRunStatus(job.status);
          if (job.status === 'COMPLETED') {
            const reply = String(job.reply || '').trim() || 'That reply arrived empty. Please try again.';
            setMessages((current) => {
              const next = [...current, { role: 'assistant', content: reply, createdAt: new Date().toISOString() }];
              saveHistory(next);
              return next;
            });
            window.localStorage.removeItem(ACTIVE_JOB_KEY);
            setJobId('');
            setRunStatus('');
            return;
          }
          if (['FAILED', 'TIMED_OUT', 'CANCELLED'].includes(job.status)) {
            setError(userFacingError(job.error, 'That reply could not be completed. Please try again.'));
            window.localStorage.removeItem(ACTIVE_JOB_KEY);
            setJobId('');
            setRunStatus('');
            return;
          }
          await wait(2500, controller.signal);
        } catch (requestError) {
          if (requestError.name === 'AbortError') return;
          consecutiveErrors += 1;
          if (consecutiveErrors >= 5) {
            setError('The connection was interrupted. Your request is still safe; reload to check it again.');
            return;
          }
          await wait(3500, controller.signal);
        }
      }
    }

    poll();
    return () => controller.abort();
  }, [jobId, user?.admin]);

  async function sendMessage(event) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;
    const nextMessages = [...messages, { role: 'user', content, createdAt: new Date().toISOString() }];
    setMessages(nextMessages);
    saveHistory(nextMessages);
    setDraft('');
    setError('');
    setSubmitting(true);
    setRunStatus('IN_QUEUE');
    try {
      const job = await apiRequest('/api/chat-boss/jobs', {
        method: 'POST',
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content: messageContent }) => ({ role, content: messageContent })),
        }),
      });
      window.localStorage.setItem(ACTIVE_JOB_KEY, job.id);
      setJobId(job.id);
      setRunStatus(job.status || 'IN_QUEUE');
    } catch (requestError) {
      setError(userFacingError(requestError, 'That reply could not be completed. Please try again.'));
      setRunStatus('');
    } finally {
      setSubmitting(false);
    }
  }

  async function stopReply() {
    if (!jobId) return;
    setError('');
    try {
      await apiRequest(`/api/chat-boss/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
    } catch (requestError) {
      setError(userFacingError(requestError, 'The reply could not be stopped. Please try again.'));
      return;
    }
    window.localStorage.removeItem(ACTIVE_JOB_KEY);
    setJobId('');
    setRunStatus('');
  }

  function clearChat() {
    if (busy) return;
    setMessages([]);
    setError('');
    window.localStorage.removeItem(HISTORY_KEY);
  }

  function exportChat() {
    const text = messages
      .map((message) => `${message.role === 'user' ? 'You' : 'Chat Boss'}:\n${message.content}`)
      .join('\n\n');
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat-boss-${new Date().toISOString().slice(0, 10)}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  if (!user) {
    return (
      <section className="chat-boss-gate">
        <span className="chat-boss-kicker">Private AI</span>
        <h1>Chat Boss</h1>
        <p>Sign in with your Polymath administrator account to use the private AI workspace.</p>
        <button type="button" className="primary" onClick={() => onNavigate('account', { next: 'chat-boss' })}>
          Sign in
        </button>
      </section>
    );
  }

  if (!user.admin) {
    return (
      <section className="chat-boss-gate">
        <span className="chat-boss-kicker">Owner only</span>
        <h1>Chat Boss is private</h1>
        <p>This account cannot start administrator AI requests.</p>
        <button type="button" onClick={() => onNavigate('studio')}>Return to studio</button>
      </section>
    );
  }

  return (
    <section className="chat-boss-page">
      <header className="chat-boss-header">
        <div>
          <span className="chat-boss-kicker">Your private workspace</span>
          <h1>Chat Boss</h1>
          <p>Private planning, writing, and technical help.</p>
        </div>
        <div className="chat-boss-header-actions">
          <button type="button" onClick={exportChat} disabled={!messages.length}>Export</button>
          <button type="button" onClick={clearChat} disabled={!messages.length || busy}>Clear</button>
        </div>
      </header>

      {loadingCapabilities ? (
        <TaskProgress compact label="Opening your workspace…" ariaLabel="Workspace loading progress" />
      ) : (
        <div className="chat-boss-connection" role="status">
          <i className={capabilities?.configured ? 'is-ready' : ''} aria-hidden="true" />
          <span>{capabilities?.configured ? 'Ready' : 'Temporarily unavailable'}</span>
          <small>History is saved only in this browser.</small>
        </div>
      )}

      <div className="chat-boss-transcript" ref={transcriptRef} aria-live="polite">
        {!messages.length && (
          <div className="chat-boss-empty">
            <span>AI</span>
            <h2>Talk to Chat Boss</h2>
            <p>Use this private workspace for planning, writing, and technical questions.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <article className={`chat-boss-message is-${message.role}`} key={`${message.createdAt || 'message'}-${index}`}>
            <strong>{message.role === 'user' ? 'You' : 'Chat Boss'}</strong>
            <p>{message.content}</p>
          </article>
        ))}
        {busy && (
          <article className="chat-boss-message is-assistant is-pending">
            <strong>Chat Boss</strong>
            <TaskProgress compact label="Preparing your reply…" ariaLabel="Reply progress" />
          </article>
        )}
      </div>

      {error && <p className="chat-boss-error" role="alert">{error}</p>}

      <form className="chat-boss-composer" onSubmit={sendMessage}>
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          rows="3"
          maxLength="6000"
          placeholder="Message Chat Boss…"
          aria-label="Message Chat Boss"
          disabled={busy || capabilities?.configured === false}
        />
        <div className="chat-boss-composer-footer">
          <small>Enter to send · Shift + Enter for a new line</small>
          {jobId ? (
            <button type="button" className="chat-boss-stop" onClick={stopReply}>Stop</button>
          ) : (
            <button type="submit" className="primary" disabled={!draft.trim() || busy || capabilities?.configured === false}>
              {submitting ? 'Please wait…' : 'Send'}
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
