import { useEffect, useMemo, useRef, useState } from 'react';
import { apiRequest } from '../services/api.js';

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

function statusCopy(status) {
  if (status === 'IN_PROGRESS') return 'Qwen is writing your reply…';
  if (status === 'IN_QUEUE') return 'Waking the RunPod GPU… first reply can take a few minutes.';
  return 'Preparing Chat Boss…';
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
        if (!cancelled) setError(requestError.message);
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
            const reply = String(job.reply || '').trim() || 'Qwen completed the job but returned no text.';
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
            setError(job.error || `RunPod ended the job with status ${job.status}.`);
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
            setError(`${requestError.message} Your RunPod job is still saved; reload to try checking it again.`);
            return;
          }
          await wait(3500, controller.signal);
        }
      }
    }

    poll();
    return () => controller.abort();
  }, [jobId, user?.admin]);

  const modelLabel = useMemo(
    () => capabilities?.model || 'Qwen/Qwen3.5-35B-A3B',
    [capabilities?.model],
  );

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
      setError(requestError.message);
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
      setError(requestError.message);
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
        <p>Sign in with your Polymath administrator account to use your private RunPod model.</p>
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
        <p>This account cannot start paid GPU jobs.</p>
        <button type="button" onClick={() => onNavigate('studio')}>Return to studio</button>
      </section>
    );
  }

  return (
    <section className="chat-boss-page">
      <header className="chat-boss-header">
        <div>
          <span className="chat-boss-kicker">Your private RunPod chat</span>
          <h1>Chat Boss</h1>
          <p>{modelLabel} · original weights · no fine-tuned adapter</p>
        </div>
        <div className="chat-boss-header-actions">
          <button type="button" onClick={exportChat} disabled={!messages.length}>Export</button>
          <button type="button" onClick={clearChat} disabled={!messages.length || busy}>Clear</button>
        </div>
      </header>

      <div className="chat-boss-connection" role="status">
        <i className={capabilities?.configured ? 'is-ready' : ''} aria-hidden="true" />
        <span>
          {loadingCapabilities
            ? 'Checking RunPod…'
            : capabilities?.configured
              ? 'Connected · scales to zero when idle'
              : 'RunPod connection is not configured'}
        </span>
        <small>History is saved only in this browser.</small>
      </div>

      <div className="chat-boss-transcript" ref={transcriptRef} aria-live="polite">
        {!messages.length && (
          <div className="chat-boss-empty">
            <span>QB</span>
            <h2>Talk directly to Qwen</h2>
            <p>The first reply may be slow while a Serverless GPU wakes. Following replies should be faster.</p>
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
            <p><span className="chat-boss-thinking" aria-hidden="true"><i /><i /><i /></span>{statusCopy(runStatus)}</p>
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
              {submitting ? 'Starting…' : 'Send'}
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
