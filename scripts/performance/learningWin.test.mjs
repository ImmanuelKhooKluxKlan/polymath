import assert from 'node:assert/strict';
import test from 'node:test';
import { buildLearningWin } from '../../src/engine/learningWin.js';

test('a measured result becomes a concise challenge with proof', () => {
  const win = buildLearningWin({
    report: { score: 87.6, matchedCount: 18, expectedCount: 20 },
    song: { title: 'Neon C Major Warmup', composer: 'Polymath Musician' },
    songKey: 'free:Neon C Major Warmup:Polymath Musician',
    level: { shortLabel: 'Stage 1' },
    momentum: { streakDays: 4 },
    baseUrl: 'https://polymathmusician67.com/#studio',
  });

  assert.equal(win.score, 88);
  assert.match(win.shareText, /18\/20 notes matched/);
  assert.match(win.shareText, /4-day streak/);
  assert.match(win.url, /try=learn/);
  assert.match(win.url, /score=88/);
  assert.match(win.url, /song=free%3A/);
});

test('private song identifiers never leave the learner share link', () => {
  const win = buildLearningWin({
    report: { score: 73, matchedCount: 8, expectedCount: 12 },
    song: { title: 'My private draft' },
    songKey: 'personal:secret-record-id',
    baseUrl: 'http://127.0.0.1:5173/#studio',
  });

  assert.equal(new URL(win.url).hash.includes('personal%3Asecret-record-id'), false);
  assert.match(win.url, /score=73/);
});

test('unsafe score and text values are bounded for a stable card', () => {
  const win = buildLearningWin({
    report: { score: 500, matchedCount: -5, expectedCount: 2 },
    song: { title: 'A'.repeat(200), artist: 'B'.repeat(200) },
    level: {},
    baseUrl: 'not a url',
  });

  assert.equal(win.score, 100);
  assert.equal(win.title.length, 90);
  assert.equal(win.artist.length, 70);
  assert.equal(win.matchedCount, 0);
  assert.equal(win.expectedCount, 2);
  assert.equal(new URL(win.url).origin, 'https://polymathmusician67.com');
});
