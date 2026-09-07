/* global AbortController, Headers, Response, URL, URLSearchParams, clearTimeout, fetch, setTimeout */

const API_ORIGIN = 'https://api.polymathmusician67.com';

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeSlug(value) {
  const slug = String(value || '').trim().toLowerCase();
  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) ? slug.slice(0, 64) : '';
}

function safeScore(value) {
  if (value === null || value === undefined || value === '') return null;
  const score = Number(value);
  return Number.isFinite(score) ? Math.max(0, Math.min(100, Math.round(score))) : null;
}

function safeReferral(value) {
  const referral = String(value || '').trim().toUpperCase();
  return /^[A-Z0-9_-]{3,32}$/.test(referral) ? referral : '';
}

function challengeTarget(slug, requestUrl, campaign = null) {
  const incoming = new URL(requestUrl);
  const params = new URLSearchParams({ try: 'learn', campaign: slug });
  const score = safeScore(incoming.searchParams.get('score'));
  const requestedReferral = safeReferral(incoming.searchParams.get('ref'));
  const canonicalReferral = safeReferral(campaign?.referralCode);
  if (score !== null) params.set('score', String(score));
  if (requestedReferral && requestedReferral === canonicalReferral) params.set('ref', canonicalReferral);
  return `/#studio?${params.toString()}`;
}

async function appHtml(context) {
  const assetUrl = new URL('/', context.request.url);
  assetUrl.search = '';
  assetUrl.hash = '';
  const response = await context.env.ASSETS.fetch(assetUrl);
  if (!response.ok) throw new Error('The Polymath application shell is unavailable.');
  return { html: await response.text(), headers: new Headers(response.headers) };
}

function injectHead(html, title, tags) {
  const withTitle = /<title>[\s\S]*?<\/title>/i.test(html)
    ? html.replace(/<title>[\s\S]*?<\/title>/i, `<title>${escapeHtml(title)}</title>`)
    : html;
  return withTitle.includes('</head>')
    ? withTitle.replace('</head>', `${tags.join('\n')}\n</head>`)
    : `${tags.join('\n')}\n${withTitle}`;
}

function pageResponse(shell, html, status, cacheControl) {
  const headers = shell.headers;
  headers.set('Content-Type', 'text/html; charset=UTF-8');
  headers.set('Cache-Control', cacheControl);
  headers.set('X-Content-Type-Options', 'nosniff');
  headers.set('X-Polymath-Route', 'artist-campaign-share');
  headers.delete('Content-Length');
  headers.delete('ETag');
  return new Response(html, { status, headers });
}

async function fetchCampaign(slug) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    return await fetch(`${API_ORIGIN}/api/artist-campaigns/${encodeURIComponent(slug)}`, {
      headers: { Accept: 'application/json' },
      redirect: 'follow',
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function renderSharePage(context) {
  const slug = safeSlug(context.params?.slug);
  const shell = await appHtml(context);
  if (!slug) {
    const title = 'Challenge unavailable | Polymath Musician';
    const html = injectHead(shell.html, title, [
      '<meta name="robots" content="noindex,nofollow">',
    ]);
    return pageResponse(shell, html, 404, 'public, max-age=15, must-revalidate');
  }

  let apiResponse;
  try {
    apiResponse = await fetchCampaign(slug);
  } catch {
    const target = challengeTarget(slug, context.request.url);
    const html = injectHead(shell.html, 'Challenge temporarily unavailable | Polymath Musician', [
      '<meta name="robots" content="noindex,nofollow">',
      `<script>history.replaceState(null,"",${JSON.stringify(target).replace(/</g, '\\u003c')});</script>`,
    ]);
    return pageResponse(shell, html, 503, 'private, no-store');
  }

  let campaign = null;
  if (apiResponse.ok) {
    try {
      campaign = (await apiResponse.json())?.campaign || null;
    } catch {
      campaign = null;
    }
  }
  const target = challengeTarget(slug, context.request.url, campaign);
  if (!apiResponse.ok || !campaign?.live) {
    const status = apiResponse.status === 404 ? 404 : 503;
    const html = injectHead(shell.html, 'Challenge unavailable | Polymath Musician', [
      '<meta name="robots" content="noindex,nofollow">',
      `<script>history.replaceState(null,"",${JSON.stringify(target).replace(/</g, '\\u003c')});</script>`,
    ]);
    return pageResponse(
      shell,
      html,
      status,
      status === 404 ? 'public, max-age=15, must-revalidate' : 'private, no-store',
    );
  }

  const origin = new URL(context.request.url).origin;
  const canonical = new URL(`/c/${encodeURIComponent(slug)}`, origin).toString();
  const cover = campaign.coverUrl ? new URL(campaign.coverUrl, API_ORIGIN).toString() : '';
  const title = `${campaign.title} by ${campaign.artist} | Polymath challenge`;
  const description = campaign.hook
    || `Play a ${campaign.preview?.durationSeconds || 20}-second piano challenge and share your score.`;
  const tags = [
    `<meta name="description" content="${escapeHtml(description)}">`,
    `<link rel="canonical" href="${escapeHtml(canonical)}">`,
    '<meta property="og:type" content="website">',
    '<meta property="og:site_name" content="Polymath Musician">',
    `<meta property="og:title" content="${escapeHtml(title)}">`,
    `<meta property="og:description" content="${escapeHtml(description)}">`,
    `<meta property="og:url" content="${escapeHtml(canonical)}">`,
    ...(cover ? [
      `<meta property="og:image" content="${escapeHtml(cover)}">`,
      `<meta property="og:image:alt" content="${escapeHtml(`${campaign.title} by ${campaign.artist}`)}">`,
    ] : []),
    '<meta name="twitter:card" content="summary_large_image">',
    `<meta name="twitter:title" content="${escapeHtml(title)}">`,
    `<meta name="twitter:description" content="${escapeHtml(description)}">`,
    ...(cover ? [`<meta name="twitter:image" content="${escapeHtml(cover)}">`] : []),
    `<script>history.replaceState(null,"",${JSON.stringify(target).replace(/</g, '\\u003c')});</script>`,
  ];
  return pageResponse(
    shell,
    injectHead(shell.html, title, tags),
    200,
    'public, max-age=15, must-revalidate, stale-while-revalidate=60',
  );
}

export async function onRequest(context) {
  if (!['GET', 'HEAD'].includes(context.request.method)) {
    return new Response('Method not allowed', {
      status: 405,
      headers: { Allow: 'GET, HEAD', 'Cache-Control': 'no-store' },
    });
  }
  const response = await renderSharePage(context);
  if (context.request.method === 'HEAD') {
    return new Response(null, { status: response.status, headers: response.headers });
  }
  return response;
}
