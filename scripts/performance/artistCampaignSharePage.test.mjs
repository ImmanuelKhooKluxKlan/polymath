/* global Request, Response */

import assert from 'node:assert/strict';
import test from 'node:test';
import { onRequest } from '../../functions/c/[slug].js';

const APP_SHELL = '<!doctype html><html><head><title>Polymath Musician</title></head><body><div id="root"></div></body></html>';

function contextFor(url, { method = 'GET', slug = 'qa-artist-song' } = {}) {
  return {
    request: new Request(url, { method }),
    params: { slug },
    env: {
      ASSETS: {
        fetch: async () => new Response(APP_SHELL, {
          headers: { 'Content-Type': 'text/html; charset=UTF-8' },
        }),
      },
    },
  };
}

test('the Cloudflare share route renders safe campaign metadata and the React target', async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = '';
  globalThis.fetch = async (url) => {
    requestedUrl = String(url);
    return Response.json({
      campaign: {
        live: true,
        slug: 'qa-artist-song',
        title: 'Safe <Song>',
        artist: 'Artist & Friends',
        hook: 'Can you beat this chorus?',
        referralCode: 'ARTIST20',
        preview: { durationSeconds: 20 },
        coverUrl: '/api/artist-campaigns/qa-artist-song/cover',
      },
    });
  };
  try {
    const response = await onRequest(contextFor(
      'https://polymathmusician67.com/c/qa-artist-song?score=88&ref=ARTIST20',
    ));
    const html = await response.text();
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('x-polymath-route'), 'artist-campaign-share');
    assert.match(requestedUrl, /api\.polymathmusician67\.com\/api\/artist-campaigns\/qa-artist-song$/);
    assert.match(html, /<meta property="og:title"/);
    assert.match(html, /Safe &lt;Song&gt; by Artist &amp; Friends/);
    assert.doesNotMatch(html, /Safe <Song>/);
    assert.match(html, /rel="canonical" href="https:\/\/polymathmusician67\.com\/c\/qa-artist-song"/);
    assert.match(html, /studio\?try=learn&amp;campaign=qa-artist-song|studio\?try=learn&campaign=qa-artist-song/);
    assert.match(html, /score=88/);
    assert.match(html, /ref=ARTIST20/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('an unavailable campaign is not indexable but still opens the in-app unavailable state', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ error: 'Not live' }, { status: 404 });
  try {
    const response = await onRequest(contextFor('https://polymathmusician67.com/c/qa-artist-song'));
    const html = await response.text();
    assert.equal(response.status, 404);
    assert.match(html, /noindex,nofollow/);
    assert.match(html, /campaign=qa-artist-song/);
    assert.doesNotMatch(html, /og:title/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('the campaign edge route rejects mutation methods', async () => {
  const response = await onRequest(contextFor(
    'https://polymathmusician67.com/c/qa-artist-song',
    { method: 'POST' },
  ));
  assert.equal(response.status, 405);
  assert.equal(response.headers.get('allow'), 'GET, HEAD');
});
