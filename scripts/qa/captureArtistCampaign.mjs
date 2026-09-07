import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '..', '..');
const qaRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-campaign-qa-'));
const port = 3197;
const baseUrl = `http://127.0.0.1:${port}`;
const chromeCandidates = process.platform === 'win32'
  ? [
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ]
  : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
const chrome = chromeCandidates.find((candidate) => fs.existsSync(candidate));
if (!chrome) throw new Error('Chrome or Edge is required for campaign screenshot QA.');

function connectToChrome(webSocketUrl) {
  if (typeof WebSocket !== 'function') {
    throw new Error('Campaign screenshot QA requires Node.js 22 or newer for its built-in WebSocket client.');
  }
  const socket = new WebSocket(webSocketUrl);
  const pending = new Map();
  let requestId = 0;
  const opened = new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', () => reject(new Error('Could not connect to Chrome DevTools.')), { once: true });
  });
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(String(event.data));
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message || 'Chrome DevTools command failed.'));
    else resolve(message.result || {});
  });
  return {
    opened,
    async send(method, params = {}, sessionId = undefined) {
      await opened;
      requestId += 1;
      const message = { id: requestId, method, params };
      if (sessionId) message.sessionId = sessionId;
      return new Promise((resolve, reject) => {
        pending.set(requestId, { resolve, reject });
        socket.send(JSON.stringify(message));
      });
    },
    close() {
      socket.close();
    },
  };
}

async function waitForChromeDebugger(processHandle) {
  return new Promise((resolve, reject) => {
    let diagnostics = '';
    const timer = setTimeout(() => reject(new Error(`Chrome DevTools did not start. ${diagnostics}`)), 10000);
    processHandle.stderr.on('data', (chunk) => {
      diagnostics += chunk.toString();
      const match = diagnostics.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (!match) return;
      clearTimeout(timer);
      resolve(match[1]);
    });
    processHandle.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`Chrome exited before QA began (code ${code}). ${diagnostics}`));
    });
  });
}

async function captureResponsivePage({
  name,
  width,
  height,
  url,
  outputPath,
  localStorage = {},
  readySelector = '.artist-challenge:not(.is-loading)',
  afterReadyExpression = '',
  finalSelector = '',
}) {
  const browser = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
    '--remote-debugging-port=0',
    `--user-data-dir=${path.join(qaRoot, `chrome-${name}`)}`,
    'about:blank',
  ], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  let cdp;
  try {
    const debuggerUrl = await waitForChromeDebugger(browser);
    cdp = connectToChrome(debuggerUrl);
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width,
      height,
      screenWidth: width,
      screenHeight: height,
      deviceScaleFactor: 1,
      mobile: width <= 680,
    }, sessionId);
    await cdp.send('Emulation.setTouchEmulationEnabled', {
      enabled: width <= 680,
      maxTouchPoints: width <= 680 ? 5 : 1,
    }, sessionId);
    const storageEntries = Object.entries(localStorage);
    if (storageEntries.length > 0) {
      await cdp.send('Page.navigate', { url: baseUrl }, sessionId);
      for (let attempt = 0; attempt < 30; attempt += 1) {
        const baseReady = await cdp.send('Runtime.evaluate', {
          expression: "document.readyState === 'complete'",
          returnByValue: true,
        }, sessionId);
        if (baseReady.result?.value) break;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      await cdp.send('Runtime.evaluate', {
        expression: `(() => {
          const entries = ${JSON.stringify(storageEntries)};
          entries.forEach(([key, value]) => window.localStorage.setItem(key, value));
        })()`,
      }, sessionId);
    }
    const destinationUrl = storageEntries.length > 0
      ? (() => {
          const refreshed = new URL(url);
          refreshed.searchParams.set('_visual_qa', name);
          return refreshed.toString();
        })()
      : url;
    await cdp.send('Page.navigate', { url: destinationUrl }, sessionId);

    let pageReady = false;
    for (let attempt = 0; attempt < 50; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      const readiness = await cdp.send('Runtime.evaluate', {
        expression: `document.readyState === 'complete' && Boolean(document.querySelector(${JSON.stringify(readySelector)}))`,
        returnByValue: true,
      }, sessionId);
      if (readiness.result?.value) {
        pageReady = true;
        break;
      }
    }
    if (!pageReady) {
      const diagnostics = await cdp.send('Runtime.evaluate', {
        expression: `({
          href: window.location.href,
          routeText: document.body?.innerText?.slice(0, 500) || '',
          hasToken: Boolean(window.localStorage.getItem('polymath_musician_auth_token')),
        })`,
        returnByValue: true,
      }, sessionId);
      throw new Error(`${name} did not finish loading ${readySelector} for visual QA. ${JSON.stringify(diagnostics.result?.value || {})}`);
    }
    if (afterReadyExpression) {
      await cdp.send('Runtime.evaluate', { expression: afterReadyExpression }, sessionId);
    }
    if (finalSelector) {
      let finalReady = false;
      for (let attempt = 0; attempt < 30; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        const readiness = await cdp.send('Runtime.evaluate', {
          expression: `Boolean(document.querySelector(${JSON.stringify(finalSelector)}))`,
          returnByValue: true,
        }, sessionId);
        if (readiness.result?.value) {
          finalReady = true;
          break;
        }
      }
      if (!finalReady) throw new Error(`${name} did not reach ${finalSelector} for visual QA.`);
      await new Promise((resolve) => setTimeout(resolve, 350));
    }

    const metricsResult = await cdp.send('Runtime.evaluate', {
      expression: `(() => {
        const challenge = document.querySelector('.artist-challenge');
        const challengeRect = challenge?.getBoundingClientRect();
        const livesInHorizontalScroller = (element) => {
          let current = element.parentElement;
          while (current && current !== document.body) {
            const overflow = getComputedStyle(current).overflowX;
            if (overflow === 'auto' || overflow === 'scroll') return true;
            current = current.parentElement;
          }
          return false;
        };
        const overflowing = [...document.querySelectorAll('body *')]
          .filter((element) => (
            element.getBoundingClientRect().right > window.innerWidth + 1
            && !livesInHorizontalScroller(element)
          ))
          .slice(0, 12)
          .map((element) => ({
            tag: element.tagName.toLowerCase(),
            className: typeof element.className === 'string' ? element.className : '',
            right: Math.round(element.getBoundingClientRect().right),
          }));
        return {
          innerWidth: window.innerWidth,
          clientWidth: document.documentElement.clientWidth,
          scrollWidth: document.documentElement.scrollWidth,
          challengeLeft: challengeRect ? Math.round(challengeRect.left) : null,
          challengeRight: challengeRect ? Math.round(challengeRect.right) : null,
          overflowing,
        };
      })()`,
      returnByValue: true,
    }, sessionId);
    const capture = await cdp.send('Page.captureScreenshot', {
      format: 'png',
      fromSurface: true,
      captureBeyondViewport: false,
    }, sessionId);
    fs.writeFileSync(outputPath, Buffer.from(capture.data, 'base64'));
    return metricsResult.result?.value || {};
  } finally {
    cdp?.close();
    if (browser.exitCode === null) {
      browser.kill();
      await Promise.race([
        once(browser, 'exit'),
        new Promise((resolve) => setTimeout(resolve, 2000)),
      ]);
    }
  }
}

const backend = spawn(process.execPath, ['server/server.js'], {
  cwd: root,
  windowsHide: true,
  stdio: ['ignore', 'pipe', 'pipe'],
  env: {
    ...process.env,
    POLYMATH_DATA_DIR: path.join(qaRoot, 'data'),
    NODE_ENV: 'production',
    PORT: String(port),
    CLIENT_ORIGIN: baseUrl,
    CLIENT_ORIGINS: baseUrl,
    ADMIN_EMAILS: 'visual-admin@example.test',
    ADMIN_PASSWORD: 'VisualCampaignPassword123',
    REGISTRATION_OTP_TEST_CODE: '123456',
    MUSCRIPTOR_ENABLED: 'false',
    DATABASE_URL: '',
    ARTIFACT_S3_BUCKET: '',
  },
});

let backendErrors = '';
backend.stderr.on('data', (chunk) => { backendErrors += chunk.toString(); });

async function waitForHealth() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(`${baseUrl}/api/health`);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Temporary server did not start. ${backendErrors}`);
}

async function jsonRequest(route, options = {}) {
  const response = await fetch(`${baseUrl}${route}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(`${route} failed (${response.status}): ${data.error || JSON.stringify(data)}`);
  return data;
}

try {
  await waitForHealth();
  const registration = await jsonRequest('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      identifier: 'visual-admin@example.test',
      password: 'VisualCampaignPassword123',
    }),
  });
  await jsonRequest('/api/auth/change-password', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${registration.token}`,
    },
    body: JSON.stringify({ password: 'VisualCampaignPassword456' }),
  });
  const song = {
    title: 'Midnight Visual Check',
    composer: 'QA Artist',
    bpm: 92,
    performance: { preserveScoreDurations: true, preserveScoreTiming: true },
    notes: [
      { note: 'A2', time: 2, duration: 1, velocity: 0.5, hand: 'left' },
      { note: 'C3', time: 12, duration: 2.4, velocity: 0.62, hand: 'left' },
      { note: 'G4', time: 12, duration: 0.75, velocity: 0.86, hand: 'right' },
      { note: 'A4', time: 13, duration: 0.75, velocity: 0.88, hand: 'right' },
      { note: 'C5', time: 14, duration: 1.2, velocity: 0.91, hand: 'right' },
      { note: 'F3', time: 15.5, duration: 2.2, velocity: 0.66, hand: 'left' },
      { note: 'E5', time: 16, duration: 1.1, velocity: 0.9, hand: 'right' },
      { note: 'D5', time: 17.5, duration: 1.2, velocity: 0.86, hand: 'right' },
      { note: 'C5', time: 19, duration: 2.2, velocity: 0.93, hand: 'right' },
      { note: 'C7', time: 40, duration: 1, velocity: 0.9, hand: 'right' },
    ],
    pedals: [{ time: 12, down: true }, { time: 15.2, down: false }, { time: 15.5, down: true }, { time: 21.5, down: false }],
  };
  const form = new FormData();
  const fields = {
    artist: 'QA Artist',
    title: 'Midnight Visual Check',
    slug: 'qa-artist-visual-check',
    referralCode: 'QAVISUAL',
    hook: 'Can you beat the artist at their own chorus?',
    description: 'A fast, human-verified piano challenge built for one satisfying first win.',
    status: 'published',
    previewStartSeconds: 12,
    previewDurationSeconds: 20,
    qaScore: 94,
    challengeScore: 88,
    rightsConfirmed: true,
    rightsHolder: 'QA fixture only',
    rightsBasis: 'Automated local visual test fixture',
    artistApproved: true,
    humanVerified: true,
    verificationNotes: 'Pitch, timing, holds, dynamics, and pedal checked for the isolated QA fixture.',
    affiliatePercent: 20,
  };
  Object.entries(fields).forEach(([key, value]) => form.append(key, String(value)));
  form.append('song', new Blob([JSON.stringify(song)], { type: 'application/json' }), 'challenge.json');
  const campaignResponse = await fetch(`${baseUrl}/api/admin/artist-campaigns`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${registration.token}` },
    body: form,
  });
  const campaignData = await campaignResponse.json();
  if (!campaignResponse.ok) throw new Error(`Campaign fixture failed: ${JSON.stringify(campaignData)}`);

  const campaignUrl = `${baseUrl}/c/${campaignData.campaign.slug}?ref=${campaignData.campaign.referralCode}`;
  const socialResponse = await fetch(campaignUrl);
  const socialHtml = await socialResponse.text();
  if (!socialResponse.ok
      || !socialHtml.includes('<meta property="og:title"')
      || !socialHtml.includes('<link rel="canonical"')
      || !socialHtml.includes(campaignData.campaign.title)) {
    throw new Error('The production campaign route did not render its social preview metadata.');
  }
  const captures = [
    ['mobile', 390, 844],
    ['desktop', 1440, 1000],
  ];
  const screenshots = {};
  const viewportMetrics = {};
  for (const [name, width, height] of captures) {
    const screenshot = path.join(qaRoot, `campaign-${name}.png`);
    viewportMetrics[name] = await captureResponsivePage({
      name,
      width,
      height,
      url: campaignUrl,
      outputPath: screenshot,
    });
    screenshots[name] = screenshot;
  }
  const privatePreviewScreenshot = path.join(qaRoot, 'campaign-private-preview-mobile.png');
  viewportMetrics.privatePreviewMobile = await captureResponsivePage({
    name: 'private-preview-mobile',
    width: 390,
    height: 844,
    url: `${baseUrl}/#studio?try=learn&campaign=${campaignData.campaign.slug}&adminPreview=1`,
    outputPath: privatePreviewScreenshot,
    localStorage: { polymath_musician_auth_token: registration.token },
  });
  screenshots.privatePreviewMobile = privatePreviewScreenshot;
  const adminScreenshot = path.join(qaRoot, 'campaign-admin-growth.png');
  viewportMetrics.adminGrowth = await captureResponsivePage({
    name: 'admin-growth',
    width: 1440,
    height: 1200,
    url: `${baseUrl}/#admin-database`,
    outputPath: adminScreenshot,
    localStorage: { polymath_musician_auth_token: registration.token },
    readySelector: '.admin-console',
    afterReadyExpression: `(() => {
      const growth = [...document.querySelectorAll('.admin-section-nav button')]
        .find((button) => button.textContent.includes('Growth'));
      growth?.click();
    })()`,
    finalSelector: '.campaign-launchpad',
  });
  screenshots.adminGrowth = adminScreenshot;
  const adminFormScreenshot = path.join(qaRoot, 'campaign-admin-form-mobile.png');
  viewportMetrics.adminFormMobile = await captureResponsivePage({
    name: 'admin-form-mobile',
    width: 390,
    height: 1000,
    url: `${baseUrl}/#admin-database`,
    outputPath: adminFormScreenshot,
    localStorage: { polymath_musician_auth_token: registration.token },
    readySelector: '.admin-console',
    afterReadyExpression: `(() => {
      const growth = [...document.querySelectorAll('.admin-section-nav button')]
        .find((button) => button.textContent.includes('Growth'));
      growth?.click();
      window.setTimeout(() => {
        const create = [...document.querySelectorAll('button')]
          .find((button) => button.textContent.trim() === 'New campaign');
        create?.click();
      }, 150);
    })()`,
    finalSelector: '.campaign-form',
  });
  screenshots.adminFormMobile = adminFormScreenshot;
  process.stdout.write(`${JSON.stringify({
    qaRoot,
    campaignStatus: campaignData.campaign.status,
    qaScore: campaignData.campaign.verification.qaScore,
    socialPreview: 'passed',
    screenshots,
    viewportMetrics,
  }, null, 2)}\n`);
} finally {
  if (backend.exitCode === null) {
    backend.kill();
    await Promise.race([
      once(backend, 'exit'),
      new Promise((resolve) => setTimeout(resolve, 3000)),
    ]);
  }
}
