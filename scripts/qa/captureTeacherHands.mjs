#!/usr/bin/env node

import { spawn } from 'node:child_process';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

function argsFrom(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    values[argv[index].slice(2)] = argv[index + 1];
    index += 1;
  }
  return values;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForFile(filename, timeoutMs = 12000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      return await fsp.readFile(filename, 'utf8');
    } catch {
      await delay(100);
    }
  }
  throw new Error(`Timed out waiting for ${filename}`);
}

class CdpSession {
  constructor(url) {
    this.url = url;
    this.nextId = 1;
    this.pending = new Map();
    this.socket = null;
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (!message.id || !this.pending.has(message.id)) return;
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result || {});
    };
    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      this.socket.onerror = () => reject(new Error('Chrome DevTools connection failed.'));
    });
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.socket?.close();
  }
}

async function pageTarget(port) {
  const response = await fetch(`http://127.0.0.1:${port}/json/list`);
  const targets = await response.json();
  const page = targets.find((target) => target.type === 'page');
  if (!page?.webSocketDebuggerUrl) throw new Error('Chrome page target was not found.');
  return page.webSocketDebuggerUrl;
}

async function main() {
  const args = argsFrom(process.argv.slice(2));
  let token = String(process.env.QA_AUTH_TOKEN || '');
  const qaApi = String(args['qa-api'] || '').replace(/\/+$/, '');
  if (!token && qaApi) {
    const parsed = new URL(qaApi);
    if (!['127.0.0.1', 'localhost'].includes(parsed.hostname)) {
      throw new Error('--qa-api is restricted to a local isolated server.');
    }
    const email = String(args['qa-email'] || 'visual-qa@polymath.test');
    const password = 'VisualQAPassword123';
    const loginResponse = await fetch(`${qaApi}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ identifier: email, password }),
    });
    const login = await loginResponse.json();
    if (loginResponse.ok) {
      token = login.token;
      if (login.user?.mustChangePassword) {
        const passwordResponse = await fetch(`${qaApi}/api/auth/change-password`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ password }),
        });
        if (!passwordResponse.ok) throw new Error('QA administrator password setup failed.');
      }
    }
    else {
      const challengeResponse = await fetch(`${qaApi}/api/auth/register/otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: 'email', email }),
      });
      const challenge = await challengeResponse.json();
      if (!challengeResponse.ok) throw new Error(challenge.error || 'QA registration challenge failed.');
      const registrationResponse = await fetch(`${qaApi}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'Visual QA',
          email,
          password,
          challengeId: challenge.challengeId,
          verificationCode: '123456',
        }),
      });
      const registration = await registrationResponse.json();
      if (!registrationResponse.ok) throw new Error(registration.error || 'QA registration failed.');
      token = registration.token;
    }
  }
  if (!token) throw new Error('QA_AUTH_TOKEN or a loopback --qa-api is required.');

  const chrome = args.chrome || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const url = args.url || 'http://127.0.0.1:5173/#studio';
  const width = Math.max(360, Number(args.width) || 1440);
  const height = Math.max(500, Number(args.height) || 1000);
  const playSeconds = Math.max(0, Number(args['play-seconds']) || 0);
  const surface = ['lesson', 'admin-characters'].includes(args.surface) ? args.surface : 'keyboard';
  const demonstrate = args.demonstrate === 'true';
  const openHelp = args['open-help'] === 'true';
  const requestedTeacher = String(args.teacher || '').trim();
  const requestedConversationMode = String(args['conversation-mode'] || '').trim();
  const startSession = args['start-session'] === 'true';
  const voiceOnly = args['voice-only'] === 'true';
  const output = path.resolve(args.output || 'qa-teacher-hands.png');
  const profile = await fsp.mkdtemp(path.join(os.tmpdir(), 'polymath-teacher-qa-'));
  const child = spawn(chrome, [
    '--headless=new',
    '--disable-gpu',
    '--autoplay-policy=no-user-gesture-required',
    '--no-first-run',
    '--no-default-browser-check',
    '--hide-scrollbars',
    '--remote-debugging-port=0',
    `--user-data-dir=${profile}`,
    'about:blank',
  ], { stdio: 'ignore', windowsHide: true });

  let session;
  try {
    const activePort = await waitForFile(path.join(profile, 'DevToolsActivePort'));
    const port = Number(activePort.split(/\r?\n/)[0]);
    session = new CdpSession(await pageTarget(port));
    await session.connect();
    await session.send('Page.enable');
    await session.send('Runtime.enable');
    if (qaApi) {
      await session.send('Page.addScriptToEvaluateOnNewDocument', {
        source: `(() => {
          const qaApi = ${JSON.stringify(qaApi)};
          window.__polymathQaApi = qaApi;
          const nativeFetch = window.fetch.bind(window);
          window.fetch = (input, init) => {
            const original = typeof input === 'string' ? input : input?.url;
            if (!original) return nativeFetch(input, init);
            let parsed;
            try { parsed = new URL(original, window.location.href); } catch { return nativeFetch(input, init); }
            const isolatedRoutes = ['/api/auth/', '/api/virtual-lessons', '/api/virtual-teachers'];
            if (!isolatedRoutes.some((prefix) => parsed.pathname.startsWith(prefix))) {
              return nativeFetch(input, init);
            }
            const redirected = new URL(qaApi + parsed.pathname + parsed.search);
            return nativeFetch(typeof input === 'string' ? redirected.href : new Request(redirected.href, input), init);
          };
        })();`,
      });
    }
    await session.send('Emulation.setDeviceMetricsOverride', {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: width < 700,
    });
    await session.send('Page.navigate', { url });
    await delay(900);
    await session.send('Runtime.evaluate', {
      expression: `localStorage.setItem('polymath_musician_auth_token', ${JSON.stringify(token)}); localStorage.setItem('polymath-teacher-hands-v2', 'true'); location.reload();`,
    });
    await delay(1800);
    if (surface === 'admin-characters') {
      await session.send('Runtime.evaluate', {
        expression: `location.hash = 'admin-database'`,
      });
      await delay(1100);
      const opened = await session.send('Runtime.evaluate', {
        expression: `(() => {
          const button = [...document.querySelectorAll('.admin-section-nav button')]
            .find((item) => /virtual teachers/i.test(item.textContent));
          button?.click();
          return Boolean(button);
        })()`,
        returnByValue: true,
      });
      if (!opened.result?.value) throw new Error('The Virtual teachers admin section was not found.');
      await delay(650);
      await session.send('Runtime.evaluate', {
        expression: `document.querySelector('.admin-character-manager')?.scrollIntoView({ block: 'start' })`,
      });
      await delay(250);
      const adminAudit = await session.send('Runtime.evaluate', {
        expression: `(() => ({
          documentOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          characterRows: document.querySelectorAll('.admin-character-row').length,
          hasMinimumAge: [...document.querySelectorAll('.admin-character-form label')].some((item) => /minimum age/i.test(item.textContent)),
          hasCharacterPrice: [...document.querySelectorAll('.admin-character-form label')].some((item) => /price per 30 minutes/i.test(item.textContent)),
          hasImageUpload: Boolean(document.querySelector('.admin-character-file-button input[type="file"]')),
          clippedButtons: [...document.querySelectorAll('.admin-character-row-actions button')].filter((button) => button.scrollWidth > button.clientWidth + 1).length,
        }))()`,
        returnByValue: true,
      });
      const audit = adminAudit.result?.value || {};
      if (audit.documentOverflow || audit.characterRows < 5 || !audit.hasMinimumAge || !audit.hasCharacterPrice || !audit.hasImageUpload || audit.clippedButtons) {
        throw new Error(`Virtual teacher admin audit failed: ${JSON.stringify(audit)}`);
      }
      const screenshot = await session.send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: false,
        fromSurface: true,
      });
      await fsp.mkdir(path.dirname(output), { recursive: true });
      await fsp.writeFile(output, Buffer.from(screenshot.data, 'base64'));
      process.stdout.write(`${JSON.stringify({ output, width, height, surface, audit }, null, 2)}\n`);
      return;
    }
    await session.send('Runtime.evaluate', {
      expression: `(() => { const button = [...document.querySelectorAll('button')].find((item) => item.textContent.trim() === 'Learn'); button?.click(); return Boolean(button); })()`,
      returnByValue: true,
    });
    await delay(650);
    if (requestedTeacher) {
      const selection = await session.send('Runtime.evaluate', {
        expression: `(() => {
          const disclosure = [...document.querySelectorAll('.teacher-studio-disclosure')]
            .find((item) => /choose another teacher/i.test(item.querySelector('summary')?.textContent || ''));
          if (disclosure) disclosure.open = true;
          const choice = [...document.querySelectorAll('.teacher-choice')]
            .find((item) => item.querySelector('strong')?.textContent.trim().toLowerCase() === ${JSON.stringify(requestedTeacher.toLowerCase())});
          choice?.click();
          return Boolean(choice);
        })()`,
        returnByValue: true,
      });
      if (!selection.result?.value) throw new Error(`Teacher ${requestedTeacher} was not found.`);
      await delay(200);
      await session.send('Runtime.evaluate', {
        expression: `(() => {
          const confirm = [...document.querySelectorAll('.teacher-age-gate-actions button')]
            .find((item) => /18 or older/i.test(item.textContent));
          confirm?.click();
          return Boolean(confirm);
        })()`,
        returnByValue: true,
      });
      await delay(300);
    }
    if (!voiceOnly) {
      const preparation = await session.send('Runtime.evaluate', {
        expression: `(async () => {
          const unlock = [...document.querySelectorAll('button')]
            .find((item) => /unlock keyboard|try keyboard again/i.test(item.textContent));
          unlock?.click();
          await new Promise((resolve) => setTimeout(resolve, 250));
          const deadline = Date.now() + 15000;
          while (Date.now() < deadline) {
            const shell = document.querySelector('.piano-shell');
            if (shell?.classList.contains('is-ready')) return { ready: true };
            const retry = [...(shell?.querySelectorAll('.piano-preparation button') || [])]
              .find((item) => /try keyboard again/i.test(item.textContent));
            if (retry) return { ready: false, error: true };
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
          return { ready: false, timeout: true };
        })()`,
        awaitPromise: true,
        returnByValue: true,
      });
      if (!preparation.result?.value?.ready) {
        throw new Error(`The real keyboard preparation did not complete: ${JSON.stringify(preparation.result?.value)}`);
      }
    }
    await session.send('Runtime.evaluate', {
      expression: surface === 'lesson'
        ? `(() => { const studio = document.querySelector('.piano-teacher-studio'); const lesson = studio?.querySelector('details'); if (lesson) lesson.open = true; studio?.scrollIntoView({ block: 'start' }); })()`
        : `document.querySelector('.piano-scroll-wrap')?.scrollIntoView({ block: 'center' })`,
    });
    await delay(350);
    if (surface === 'lesson' && requestedConversationMode === 'adult-companion') {
      const mode = await session.send('Runtime.evaluate', {
        expression: `(() => {
          const button = [...document.querySelectorAll('.virtual-lesson-mode button')]
            .find((item) => /flirty companion/i.test(item.textContent));
          button?.click();
          return Boolean(button);
        })()`,
        returnByValue: true,
      });
      if (!mode.result?.value) throw new Error('Adult companion mode was not available for the selected teacher.');
      await delay(250);
      await session.send('Runtime.evaluate', {
        expression: `(() => {
          document.querySelectorAll('.virtual-companion-consent input[type="checkbox"]')
            .forEach((input) => { if (!input.checked) input.click(); });
        })()`,
      });
      await delay(250);
      const consent = await session.send('Runtime.evaluate', {
        expression: `(() => ({
          checked: [...document.querySelectorAll('.virtual-companion-consent input[type="checkbox"]')]
            .filter((input) => input.checked).length,
          total: document.querySelectorAll('.virtual-companion-consent input[type="checkbox"]').length,
        }))()`,
        returnByValue: true,
      });
      if (consent.result?.value?.checked !== consent.result?.value?.total) {
        throw new Error(`Adult companion consent controls did not persist: ${JSON.stringify(consent.result?.value)}`);
      }
    }
    if (surface === 'lesson' && startSession) {
      const startDeadline = Date.now() + 5000;
      let startResult = null;
      while (Date.now() < startDeadline && !startResult?.found && !startResult?.active) {
        const start = await session.send('Runtime.evaluate', {
          expression: `(() => {
            if (document.querySelector('.virtual-lesson-room')) return { active: true };
            const button = [...document.querySelectorAll('.virtual-lesson-purchase-row button')]
              .find((item) => /start private/i.test(item.textContent));
            if (!button || button.disabled) return {
              found: Boolean(button),
              disabled: Boolean(button?.disabled),
              panel: document.querySelector('.virtual-lesson-checkout, .virtual-lesson-gate, .virtual-lesson-loading')?.className || '',
              studio: Boolean(document.querySelector('.piano-teacher-studio')),
              modeButtons: [...document.querySelectorAll('.learn-mode-tabs button')].map((item) => item.textContent.trim()),
              location: window.location.hash,
              qaApi: window.__polymathQaApi || '',
              tokenLength: (localStorage.getItem('polymath_musician_auth_token') || '').length,
              body: document.body.innerText.slice(0, 600),
            };
            button.click();
            return { found: true, disabled: false };
          })()`,
          returnByValue: true,
        });
        startResult = start.result?.value || null;
        if (!startResult?.found && !startResult?.active) await delay(100);
      }
      if (!startResult?.active && (!startResult?.found || startResult?.disabled)) {
        throw new Error(`Private session could not start: ${JSON.stringify(startResult)}`);
      }
      const deadline = Date.now() + 5000;
      let active = Boolean(startResult?.active);
      while (Date.now() < deadline && !active) {
        await delay(100);
        const result = await session.send('Runtime.evaluate', {
          expression: `Boolean(document.querySelector('.virtual-lesson-room'))`,
          returnByValue: true,
        });
        active = Boolean(result.result?.value);
      }
      if (!active) throw new Error('The server did not activate the private session.');
      await delay(250);
      if (voiceOnly) {
        await session.send('Runtime.evaluate', {
          expression: `(() => {
            const button = [...document.querySelectorAll('.virtual-lesson-actions button')]
              .find((item) => /enable teacher voice/i.test(item.textContent));
            button?.click();
            return Boolean(button);
          })()`,
          returnByValue: true,
        });
        await delay(150);
      }
    }
    if (surface === 'lesson' && demonstrate) {
      const command = await session.send('Runtime.evaluate', {
        expression: `(() => { const button = [...document.querySelectorAll('.virtual-lesson-prompts button')].find((item) => /first 5 seconds/i.test(item.textContent)); button?.click(); return Boolean(button); })()`,
        returnByValue: true,
      });
      if (!command.result?.value) throw new Error('The five-second demonstration control was not found.');
      await delay(900);
    }
    if (playSeconds > 0) {
      await session.send('Runtime.evaluate', {
        expression: `(() => {
          const play = [...document.querySelectorAll('button')]
            .find((item) => item.textContent.trim() === 'Play');
          play?.click();
          return Boolean(play);
        })()`,
        returnByValue: true,
      });
      await delay(playSeconds * 1000);
      await session.send('Runtime.evaluate', {
        expression: surface === 'lesson'
          ? `document.querySelector('.piano-teacher-studio')?.scrollIntoView({ block: 'start' })`
          : `document.querySelector('.piano-scroll-wrap')?.scrollIntoView({ block: 'center' })`,
      });
      await delay(180);
    }
    if (openHelp) {
      const help = await session.send('Runtime.evaluate', {
        expression: `(() => {
          const button = document.querySelector('.support-assistant-trigger');
          button?.click();
          return Boolean(button);
        })()`,
        returnByValue: true,
      });
      if (!help.result?.value) throw new Error('Signed-in Help trigger was not found.');
      await delay(250);
    }
    if (surface === 'lesson' && startSession) {
      await session.send('Runtime.evaluate', {
        expression: `document.querySelector('.virtual-teacher-live-stage')?.scrollIntoView({ block: 'center' })`,
      });
      await delay(220);
    }
    const visualAudit = await session.send('Runtime.evaluate', {
      expression: `(() => {
        const deck = document.querySelector('.keyboard-deck')?.getBoundingClientRect();
        const hands = [...document.querySelectorAll('.teacher-main-hand-photo')]
          .map((node) => {
            const rect = node.getBoundingClientRect();
            return {
              side: node.classList.contains('teacher-main-hand-left') ? 'left' : 'right',
              state: node.classList.contains('is-pressing')
                ? 'pressing'
                : node.classList.contains('is-upcoming') ? 'upcoming' : 'rest',
              left: Math.round(rect.left),
              top: Math.round(rect.top),
              right: Math.round(rect.right),
              bottom: Math.round(rect.bottom),
              overlapsKeys: Boolean(deck && rect.top < deck.bottom && rect.bottom > deck.top),
            };
          });
        return {
          handCount: hands.length,
          bothHandsOverlapMainKeyboard: hands.length === 2 && hands.every((hand) => hand.overlapsKeys),
          documentOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          demonstrationPlaying: [...document.querySelectorAll('button')].some((item) => item.textContent.trim() === 'Pause'),
          teacherReplyCount: document.querySelectorAll('.teacher-message-teacher').length,
          teacherLocked: Boolean(document.querySelector('.teacher-session-lock')),
          lockedTeacherId: document.querySelector('.teacher-session-lock')?.dataset.lockedTeacherId || '',
          teacherRosterAvailable: Boolean(document.querySelector('.teacher-roster')),
          liveTeacherStage: Boolean(document.querySelector('.virtual-teacher-live-stage')),
          liveTeacherStageId: document.querySelector('.virtual-teacher-live-stage')?.dataset.teacherId || '',
          speechMouthLayer: Boolean(document.querySelector('.teacher-speech-mouth-window')),
          songSyncedHandCamera: Boolean(document.querySelector('.virtual-teacher-live-stage .teacher-hand-camera')),
          voiceControl: [...document.querySelectorAll('.virtual-lesson-actions button')]
            .find((item) => /teacher voice/i.test(item.textContent))?.textContent.trim() || '',
          speedLabel: document.querySelector('.dock-speed span')?.textContent?.trim() || '',
          supportOpen: Boolean(document.querySelector('.support-assistant-panel')),
          supportAllowance: document.querySelector('.support-assistant-panel header small')?.textContent?.trim() || '',
          hands,
        };
      })()`,
      returnByValue: true,
    });
    if (!voiceOnly && !visualAudit.result?.value?.bothHandsOverlapMainKeyboard) {
      throw new Error(`Teacher hands failed main-keyboard overlap audit: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    if (demonstrate && !visualAudit.result?.value?.demonstrationPlaying) {
      throw new Error(`The paid teacher command did not start main-piano playback: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    if (demonstrate && !/1\.00/.test(visualAudit.result?.value?.speedLabel || '')) {
      throw new Error(`A normal demonstration changed the learner's speed unexpectedly: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    if (startSession && (
      !visualAudit.result?.value?.teacherLocked
      || visualAudit.result?.value?.teacherRosterAvailable
      || !visualAudit.result?.value?.liveTeacherStage
      || !visualAudit.result?.value?.speechMouthLayer
      || !visualAudit.result?.value?.songSyncedHandCamera
      || visualAudit.result?.value?.lockedTeacherId !== visualAudit.result?.value?.liveTeacherStageId
    )) {
      throw new Error(`Paid teacher identity or synchronized stage audit failed: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    if (startSession && !/teacher voice on|enable teacher voice/i.test(visualAudit.result?.value?.voiceControl || '')) {
      throw new Error(`Teacher voice recovery control was not available: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    if (openHelp && (!visualAudit.result?.value?.supportOpen || !/\d+\/7 left today|Unlimited Help/.test(visualAudit.result?.value?.supportAllowance || ''))) {
      throw new Error(`Signed-in Help allowance was not visible: ${JSON.stringify(visualAudit.result?.value)}`);
    }
    const screenshot = await session.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false,
      fromSurface: true,
    });
    await fsp.mkdir(path.dirname(output), { recursive: true });
    await fsp.writeFile(output, Buffer.from(screenshot.data, 'base64'));
    process.stdout.write(`${JSON.stringify({ output, width, height, playSeconds, surface, demonstrate, audit: visualAudit.result.value }, null, 2)}\n`);
  } finally {
    session?.close();
    child.kill();
    await delay(150);
    const resolvedProfile = path.resolve(profile);
    const tempRoot = `${path.resolve(os.tmpdir())}${path.sep}`;
    if (resolvedProfile.startsWith(tempRoot)) {
      await fsp.rm(resolvedProfile, { recursive: true, force: true, maxRetries: 3 });
    }
  }
}

main().catch((error) => {
  process.stderr.write(`Teacher-hands capture failed: ${error.message}\n`);
  process.exitCode = 1;
});
