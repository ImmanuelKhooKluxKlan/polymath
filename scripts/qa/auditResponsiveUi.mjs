import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const outputRoot = path.join(root, 'artifacts', 'qa', 'responsive-ui');
const baseUrl = String(process.env.POLYMATH_QA_BASE_URL || 'http://127.0.0.1:5173').replace(/\/$/, '');
const qaRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-responsive-qa-'));

const chromeCandidates = process.platform === 'win32'
  ? [
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ]
  : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
const chrome = chromeCandidates.find((candidate) => fs.existsSync(candidate));
if (!chrome) throw new Error('Chrome or Edge is required for responsive UI QA.');

const viewports = [
  { id: 'foldable-phone-portrait', width: 280, height: 653, mobile: true },
  { id: 'small-phone-portrait', width: 320, height: 568, mobile: true },
  { id: 'small-phone-landscape', width: 568, height: 320, mobile: true },
  { id: 'modern-phone-portrait', width: 390, height: 844, mobile: true },
  { id: 'modern-phone-landscape', width: 844, height: 390, mobile: true },
  { id: 'small-tablet-portrait', width: 600, height: 960, mobile: true },
  { id: 'tablet-landscape', width: 1024, height: 768, mobile: true },
  { id: 'half-laptop', width: 683, height: 768, mobile: false },
];

const routes = [
  ['studio', 'piano'],
  ['guitar', 'guitar'],
  ['ensemble', 'instruments'],
  ['published-songs', 'composers'],
  ['find-teacher', 'learn'],
  ['community', 'community'],
  ['payment', 'subscriptions'],
  ['account', 'account'],
];

function connectToChrome(webSocketUrl) {
  if (typeof WebSocket !== 'function') throw new Error('Responsive UI QA requires Node.js 22 or newer.');
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
      return new Promise((resolve, reject) => {
        pending.set(requestId, { resolve, reject });
        socket.send(JSON.stringify({ id: requestId, method, params, ...(sessionId ? { sessionId } : {}) }));
      });
    },
    close() { socket.close(); },
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
      reject(new Error(`Chrome exited before responsive QA began (code ${code}). ${diagnostics}`));
    });
  });
}

async function waitForPage(cdp, sessionId) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    const ready = await cdp.send('Runtime.evaluate', {
      expression: `document.readyState === 'complete'
        && Boolean(document.querySelector('.app-shell'))
        && !document.querySelector('.route-loading')`,
      returnByValue: true,
    }, sessionId);
    if (ready.result?.value) {
      await new Promise((resolve) => setTimeout(resolve, 450));
      return;
    }
  }
  throw new Error('The page did not finish rendering for responsive QA.');
}

async function inspectPage(cdp, sessionId, viewport) {
  const evaluation = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const visible = (element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) !== 0
          && rect.width > 0 && rect.height > 0;
      };
      const descriptor = (element) => {
        const text = (element.innerText || element.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim();
        return {
          tag: element.tagName.toLowerCase(),
          className: typeof element.className === 'string' ? element.className.slice(0, 140) : '',
          text: text.slice(0, 100),
        };
      };
      const intendedScroller = (element) => {
        let current = element.parentElement;
        while (current && current !== document.body) {
          const overflowX = getComputedStyle(current).overflowX;
          if (overflowX === 'auto' || overflowX === 'scroll') return true;
          current = current.parentElement;
        }
        return false;
      };
      const all = [...document.querySelectorAll('body *')].filter(visible);
      const controls = all.filter((element) => element.matches('button, summary, a, input, select, textarea'));
      const clippedControls = controls
        .filter((element) => {
          const label = (element.innerText || element.getAttribute('aria-label') || element.title || '').trim();
          return label && (element.scrollWidth > element.clientWidth + 2 || element.scrollHeight > element.clientHeight + 2);
        })
        .slice(0, 25)
        .map((element) => ({
          ...descriptor(element),
          client: [element.clientWidth, element.clientHeight],
          scroll: [element.scrollWidth, element.scrollHeight],
        }));
      const outsideViewport = all
        .filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.right > viewportWidth + 2 && !intendedScroller(element);
        })
        .slice(0, 25)
        .map((element) => ({ ...descriptor(element), right: Math.round(element.getBoundingClientRect().right) }));
      const oversizedControls = controls
        .filter((element) => {
          if (element.matches('.piano-key, .guitar-string button')) return false;
          const rect = element.getBoundingClientRect();
          const text = (element.innerText || '').trim();
          return rect.height > Math.min(104, viewportHeight * 0.24)
            || (text.length < 42 && rect.width > viewportWidth * 0.94 && rect.height > 76);
        })
        .slice(0, 25)
        .map((element) => {
          const rect = element.getBoundingClientRect();
          return { ...descriptor(element), size: [Math.round(rect.width), Math.round(rect.height)] };
        });
      const largeText = all
        .filter((element) => {
          const size = Number.parseFloat(getComputedStyle(element).fontSize);
          const isHeading = element.matches('h1, h2, h3');
          const limit = isHeading ? (viewportHeight <= 600 ? 38 : 46) : 22;
          return size > limit && (element.innerText || '').trim();
        })
        .slice(0, 25)
        .map((element) => ({
          ...descriptor(element),
          fontSize: getComputedStyle(element).fontSize,
        }));
      const controlMetric = (element) => {
        if (!element || !visible(element)) return null;
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return {
          ...descriptor(element),
          width: Math.round(rect.width * 10) / 10,
          height: Math.round(rect.height * 10) / 10,
          fontSize: Number.parseFloat(style.fontSize),
          padding: [style.paddingTop, style.paddingRight, style.paddingBottom, style.paddingLeft],
        };
      };
      const landscapeButtonReference = controlMetric(document.querySelector('.piano-core-start-actions button'));
      const playbackButtons = [...document.querySelectorAll('.transport-dock .transport-button-group button, .guitar-transport-dock .transport-button-group button')]
        .map(controlMetric)
        .filter(Boolean);
      return {
        viewport: [viewportWidth, viewportHeight],
        page: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
        bodyHorizontalOverflow: document.documentElement.scrollWidth > viewportWidth + 2,
        clippedControls,
        outsideViewport,
        oversizedControls,
        largeText,
        landscapeButtonReference,
        playbackButtons,
      };
    })()`,
    returnByValue: true,
  }, sessionId);
  return { ...evaluation.result?.value, requestedViewport: [viewport.width, viewport.height] };
}

async function main() {
  fs.rmSync(outputRoot, { recursive: true, force: true });
  fs.mkdirSync(outputRoot, { recursive: true });
  const browser = spawn(chrome, [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
    '--remote-debugging-port=0', `--user-data-dir=${qaRoot}`, 'about:blank',
  ], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  let cdp;
  const report = {};
  try {
    cdp = connectToChrome(await waitForChromeDebugger(browser));
    for (const viewport of viewports) {
      report[viewport.id] = {};
      for (const [route, label] of routes) {
        const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
        const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
        await cdp.send('Page.enable', {}, sessionId);
        await cdp.send('Runtime.enable', {}, sessionId);
        await cdp.send('Emulation.setDeviceMetricsOverride', {
          width: viewport.width,
          height: viewport.height,
          screenWidth: viewport.width,
          screenHeight: viewport.height,
          deviceScaleFactor: 1,
          mobile: viewport.mobile,
        }, sessionId);
        await cdp.send('Emulation.setTouchEmulationEnabled', {
          enabled: viewport.mobile,
          maxTouchPoints: viewport.mobile ? 5 : 1,
        }, sessionId);
        await cdp.send('Page.navigate', { url: `${baseUrl}/#${route}` }, sessionId);
        await waitForPage(cdp, sessionId);
        report[viewport.id][route] = await inspectPage(cdp, sessionId, viewport);
        const screenshot = await cdp.send('Page.captureScreenshot', {
          format: 'png',
          fromSurface: true,
          captureBeyondViewport: false,
        }, sessionId);
        fs.writeFileSync(
          path.join(outputRoot, `${viewport.id}-${label}.png`),
          Buffer.from(screenshot.data, 'base64'),
        );
        if (viewport.width > viewport.height && viewport.height <= 600 && (route === 'studio' || route === 'guitar')) {
          const dockSelector = route === 'studio' ? '.transport-dock' : '.guitar-transport-dock';
          await cdp.send('Runtime.evaluate', {
            expression: `document.querySelector('${dockSelector}')?.scrollIntoView({ block: 'center' })`,
          }, sessionId);
          await new Promise((resolve) => setTimeout(resolve, 150));
          const controlsScreenshot = await cdp.send('Page.captureScreenshot', {
            format: 'png',
            fromSurface: true,
            captureBeyondViewport: false,
          }, sessionId);
          fs.writeFileSync(
            path.join(outputRoot, `${viewport.id}-${label}-playback-controls.png`),
            Buffer.from(controlsScreenshot.data, 'base64'),
          );
        }
        await cdp.send('Target.closeTarget', { targetId });
      }
    }
    fs.writeFileSync(path.join(outputRoot, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    const failures = [];
    for (const [viewportId, pages] of Object.entries(report)) {
      for (const [route, result] of Object.entries(pages)) {
        if (result.bodyHorizontalOverflow || result.clippedControls.length || result.outsideViewport.length || result.oversizedControls.length) {
          failures.push({
            viewport: viewportId,
            route,
            bodyHorizontalOverflow: result.bodyHorizontalOverflow,
            clippedControls: result.clippedControls.length,
            outsideViewport: result.outsideViewport.length,
            oversizedControls: result.oversizedControls.length,
            largeText: result.largeText.length,
          });
        }
      }
    }
    console.log(JSON.stringify({ outputRoot, failures, report }, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally {
    cdp?.close();
    if (browser.exitCode === null) {
      browser.kill();
      await Promise.race([once(browser, 'exit'), new Promise((resolve) => setTimeout(resolve, 2000))]);
    }
    fs.rmSync(qaRoot, { recursive: true, force: true });
  }
}

await main();
