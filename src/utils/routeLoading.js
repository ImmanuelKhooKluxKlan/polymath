const DEFAULT_ROUTE_RETRY_DELAYS_MS = Object.freeze([1200, 2400, 4800, 7200, 9600]);

function wait(milliseconds) {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, Math.max(0, Number(milliseconds) || 0));
  });
}

export function isRetryableRouteLoadError(error) {
  const name = String(error?.name || '');
  const message = String(error?.message || error || '');
  return name === 'ChunkLoadError'
    || /failed to fetch dynamically imported module/i.test(message)
    || /error loading dynamically imported module/i.test(message)
    || /importing a module script failed/i.test(message)
    || /failed to load module script/i.test(message)
    || /loading (?:css )?chunk [^ ]+ failed/i.test(message);
}

export async function loadRouteWithRetry(
  importer,
  { retryDelaysMs = DEFAULT_ROUTE_RETRY_DELAYS_MS, waitFor = wait } = {},
) {
  if (typeof importer !== 'function') throw new TypeError('A route importer is required.');
  const delays = Array.isArray(retryDelaysMs) ? retryDelaysMs : DEFAULT_ROUTE_RETRY_DELAYS_MS;

  for (let attempt = 0; ; attempt += 1) {
    try {
      return await importer();
    } catch (error) {
      if (!isRetryableRouteLoadError(error) || attempt >= delays.length) throw error;
      await waitFor(Math.max(0, Number(delays[attempt]) || 0));
    }
  }
}

export function routeLoadingProgress(elapsedMilliseconds) {
  const elapsed = Math.max(0, Number(elapsedMilliseconds) || 0);
  const eased = 7 + (87 * (1 - Math.exp(-elapsed / 22000)));
  return Math.min(92, Math.round(eased));
}

export function routeLoadingDetail(elapsedMilliseconds) {
  const elapsed = Math.max(0, Number(elapsedMilliseconds) || 0);
  if (elapsed < 8000) return 'Preparing the page and its controls.';
  if (elapsed < 30000) return 'A slower connection is okay. This page is still opening.';
  return 'Still loading safely. You can leave this page open.';
}
