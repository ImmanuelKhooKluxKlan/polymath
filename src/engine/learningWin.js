const DEFAULT_BASE_URL = 'https://polymathmusician67.com/';

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function safeText(value, fallback, maximum = 120) {
  const text = String(value || '').trim().replace(/\s+/g, ' ');
  return (text || fallback).slice(0, maximum);
}

function safeScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(clamp(number, 0, 100)) : 0;
}

function publicSongKey(songKey) {
  const value = String(songKey || '').trim();
  return value.startsWith('free:') && value.length <= 240 ? value : '';
}

function shareUrl(baseUrl, score, songKey) {
  let url;
  try {
    url = new URL(baseUrl || DEFAULT_BASE_URL, DEFAULT_BASE_URL);
  } catch {
    url = new URL(DEFAULT_BASE_URL);
  }
  url.search = '';
  const params = new URLSearchParams({ try: 'learn', score: String(score) });
  const shareableSong = publicSongKey(songKey);
  if (shareableSong) params.set('song', shareableSong);
  url.hash = `studio?${params.toString()}`;
  return url.toString();
}

function resultLabel(score) {
  if (score >= 90) return 'Performance-ready';
  if (score >= 75) return 'Strong progress';
  if (score >= 55) return 'Building momentum';
  return 'First win complete';
}

export function buildLearningWin({
  report,
  song,
  songKey,
  level,
  momentum,
  baseUrl,
} = {}) {
  const score = safeScore(report?.score);
  const matchedCount = Math.max(0, Math.round(Number(report?.matchedCount) || 0));
  const expectedCount = Math.max(matchedCount, Math.round(Number(report?.expectedCount) || 0));
  const title = safeText(song?.title, 'My piano song', 90);
  const artist = safeText(song?.artist || song?.composer, 'Polymath arrangement', 70);
  const stage = safeText(level?.shortLabel || level?.label, 'Piano lesson', 50);
  const streakDays = Math.max(0, Math.round(Number(momentum?.streakDays) || 0));
  const url = shareUrl(baseUrl, score, songKey);
  const noteProof = expectedCount > 0 ? `${matchedCount}/${expectedCount} notes matched` : `${stage} complete`;
  const streakProof = streakDays > 1 ? ` · ${streakDays}-day streak` : '';

  return {
    score,
    title,
    artist,
    stage,
    matchedCount,
    expectedCount,
    streakDays,
    resultLabel: resultLabel(score),
    challengeLabel: `Can you beat ${score}?`,
    shareTitle: `${title}: ${score}/100 on Polymath Musician`,
    shareText: `I scored ${score}/100 learning “${title}” on Polymath Musician — ${noteProof}${streakProof}. Can you beat my score?`,
    url,
    filename: `polymath-piano-win-${score}.png`,
  };
}

function roundedRectangle(context, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + safeRadius, y);
  context.lineTo(x + width - safeRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  context.lineTo(x + width, y + height - safeRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  context.lineTo(x + safeRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  context.lineTo(x, y + safeRadius);
  context.quadraticCurveTo(x, y, x + safeRadius, y);
  context.closePath();
}

function fitText(context, text, maximumWidth, startingSize, minimumSize = 28) {
  let size = startingSize;
  while (size > minimumSize) {
    context.font = `800 ${size}px system-ui, -apple-system, BlinkMacSystemFont, sans-serif`;
    if (context.measureText(text).width <= maximumWidth) break;
    size -= 2;
  }
  return size;
}

export async function createLearningWinCard(win) {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = 1200;
  canvas.height = 630;
  const context = canvas.getContext('2d');
  if (!context) return null;

  const background = context.createLinearGradient(0, 0, 1200, 630);
  background.addColorStop(0, '#090c25');
  background.addColorStop(0.55, '#17154a');
  background.addColorStop(1, '#421957');
  context.fillStyle = background;
  context.fillRect(0, 0, 1200, 630);

  const glow = context.createRadialGradient(980, 80, 20, 980, 80, 430);
  glow.addColorStop(0, 'rgba(217, 79, 198, 0.46)');
  glow.addColorStop(1, 'rgba(217, 79, 198, 0)');
  context.fillStyle = glow;
  context.fillRect(0, 0, 1200, 630);

  context.fillStyle = 'rgba(255, 255, 255, 0.055)';
  roundedRectangle(context, 54, 50, 1092, 530, 34);
  context.fill();
  context.strokeStyle = 'rgba(176, 157, 255, 0.24)';
  context.lineWidth = 2;
  context.stroke();

  const badge = context.createLinearGradient(72, 70, 172, 170);
  badge.addColorStop(0, '#75ead0');
  badge.addColorStop(1, '#a78aff');
  context.fillStyle = badge;
  roundedRectangle(context, 80, 78, 86, 86, 24);
  context.fill();
  context.fillStyle = '#090c25';
  context.font = '900 31px system-ui, sans-serif';
  context.textAlign = 'center';
  context.fillText('PM', 123, 133);
  context.textAlign = 'left';

  context.fillStyle = '#85f0dd';
  context.font = '800 25px system-ui, sans-serif';
  context.fillText('POLYMATH MUSICIAN · PIANO WIN', 190, 112);
  context.fillStyle = '#d6d0ef';
  context.font = '600 22px system-ui, sans-serif';
  context.fillText(`${win.stage} · ${win.resultLabel}`, 190, 148);

  const scoreGradient = context.createLinearGradient(82, 210, 340, 500);
  scoreGradient.addColorStop(0, '#826aff');
  scoreGradient.addColorStop(1, '#e452c6');
  context.fillStyle = scoreGradient;
  roundedRectangle(context, 80, 210, 270, 278, 38);
  context.fill();
  context.fillStyle = '#ffffff';
  context.font = '900 118px system-ui, sans-serif';
  context.textAlign = 'center';
  context.fillText(String(win.score), 215, 368);
  context.font = '800 27px system-ui, sans-serif';
  context.fillText('OUT OF 100', 215, 417);
  context.textAlign = 'left';

  context.fillStyle = '#ffffff';
  const titleSize = fitText(context, win.title, 690, 58, 34);
  context.font = `800 ${titleSize}px system-ui, sans-serif`;
  context.fillText(win.title, 405, 270, 690);
  context.fillStyle = '#bdb6da';
  context.font = '600 29px system-ui, sans-serif';
  context.fillText(win.artist, 405, 314, 690);

  context.fillStyle = '#f4f1ff';
  context.font = '800 31px system-ui, sans-serif';
  const noteProof = win.expectedCount > 0
    ? `${win.matchedCount}/${win.expectedCount} notes matched`
    : `${win.stage} complete`;
  context.fillText(noteProof, 405, 390);
  if (win.streakDays > 1) {
    context.fillStyle = '#84e8ff';
    context.font = '750 26px system-ui, sans-serif';
    context.fillText(`${win.streakDays}-day practice streak`, 405, 433);
  }
  context.fillStyle = '#ffffff';
  context.font = '850 32px system-ui, sans-serif';
  context.fillText(win.challengeLabel, 405, 493);

  context.fillStyle = '#aaa4ca';
  context.font = '650 21px system-ui, sans-serif';
  context.fillText('polymathmusician67.com', 80, 548);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png', 0.92));
  if (!blob) return null;
  return typeof window.File === 'function'
    ? new window.File([blob], win.filename, { type: 'image/png' })
    : blob;
}

async function copyShareText(win) {
  const text = `${win.shareText}\n${win.url}`;
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const input = document.createElement('textarea');
  input.value = text;
  input.setAttribute('readonly', '');
  input.style.position = 'fixed';
  input.style.opacity = '0';
  document.body.appendChild(input);
  input.select();
  document.execCommand('copy');
  input.remove();
}

export async function shareLearningWin(win) {
  try {
    if (navigator.share) {
      const card = await createLearningWinCard(win).catch(() => null);
      const files = card && typeof window.File === 'function' && card instanceof window.File ? [card] : [];
      if (files.length && navigator.canShare?.({ files })) {
        await navigator.share({ title: win.shareTitle, text: win.shareText, url: win.url, files });
      } else {
        await navigator.share({ title: win.shareTitle, text: win.shareText, url: win.url });
      }
      return 'shared';
    }
    await copyShareText(win);
    return 'copied';
  } catch (error) {
    if (error?.name === 'AbortError') return 'cancelled';
    await copyShareText(win);
    return 'copied';
  }
}
