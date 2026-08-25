const configuredAssetOrigin = String(import.meta.env.VITE_ASSET_BASE_URL || '').trim().replace(/\/+$/, '');
const configuredAssetRelease = String(import.meta.env.VITE_ASSET_RELEASE || '').trim().replace(/^\/+|\/+$/g, '');

function localBaseUrl() {
  const baseUrl = String(import.meta.env.BASE_URL || '/');
  return baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
}

/**
 * Builds a URL for a public, cacheable asset.
 *
 * Local development keeps using Vite's public directory. Production can point
 * at a versioned Cloudflare R2 custom domain, for example:
 * https://audio.example.com/v1/samples/iowa-mf/A4.wav
 */
export function publicAssetUrl(relativePath) {
  const cleanPath = String(relativePath || '').replace(/^\/+/, '');
  if (!configuredAssetOrigin) return `${localBaseUrl()}/${cleanPath}`;

  const releasePrefix = configuredAssetRelease ? `${configuredAssetRelease}/` : '';
  return `${configuredAssetOrigin}/${releasePrefix}${cleanPath}`;
}

export function relativeAssetUrl(baseUrl, relativePath) {
  return new URL(String(relativePath || ''), new URL(baseUrl, window.location.href)).href;
}

export const ASSET_DELIVERY = Object.freeze({
  remote: Boolean(configuredAssetOrigin),
  origin: configuredAssetOrigin,
  release: configuredAssetRelease,
});
