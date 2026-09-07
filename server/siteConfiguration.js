const crypto = require('crypto');

class SiteConfigurationError extends Error {
  constructor(message, status = 400, code = 'SITE_CONFIGURATION_INVALID') {
    super(message);
    this.name = 'SiteConfigurationError';
    this.status = status;
    this.code = code;
  }
}

const NAVIGATION_GROUPS = Object.freeze(['primary', 'more']);

const NAVIGATION_REGISTRY = Object.freeze([
  Object.freeze({ id: 'studio', path: '#studio', defaultLabel: 'Piano', defaultGroup: 'primary', defaultOrder: 10, access: 'public' }),
  Object.freeze({ id: 'create-music', path: '#create-music', defaultLabel: 'Create Music', defaultGroup: 'primary', defaultOrder: 20, access: 'public' }),
  Object.freeze({ id: 'guitar', path: '#guitar', defaultLabel: 'Guitar', defaultGroup: 'primary', defaultOrder: 30, access: 'public' }),
  Object.freeze({ id: 'ensemble', path: '#ensemble', defaultLabel: 'Instruments', defaultGroup: 'primary', defaultOrder: 40, access: 'public' }),
  Object.freeze({ id: 'published-songs', path: '#published-songs', defaultLabel: 'Composers', defaultGroup: 'more', defaultOrder: 50, access: 'public' }),
  Object.freeze({ id: 'find-teacher', path: '#find-teacher', defaultLabel: 'Find Teacher', defaultGroup: 'more', defaultOrder: 60, access: 'public' }),
  Object.freeze({ id: 'band', path: '#band', defaultLabel: 'Band', defaultGroup: 'more', defaultOrder: 70, access: 'public' }),
  Object.freeze({ id: 'your-songs', path: '#your-songs', defaultLabel: 'Your Songs', defaultGroup: 'more', defaultOrder: 80, access: 'signed-in' }),
]);

const REGISTRY_BY_ID = new Map(NAVIGATION_REGISTRY.map((item) => [item.id, item]));
const MAX_HISTORY = 200;

function cleanText(value, maximum, fallback = '') {
  const cleaned = String(value ?? '')
    .replace(/[<>]/g, '')
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return (cleaned || fallback).slice(0, maximum);
}

function integer(value, minimum, maximum, fallback) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function defaultNavigation() {
  return NAVIGATION_REGISTRY.map((item) => ({
    id: item.id,
    label: item.defaultLabel,
    group: item.defaultGroup,
    order: item.defaultOrder,
    visible: true,
  }));
}

function defaultSiteConfiguration() {
  return {
    revision: 1,
    brand: {
      name: 'Polymath',
      suffix: 'Musician',
    },
    announcement: {
      enabled: false,
      text: '',
      tone: 'info',
    },
    navigation: defaultNavigation(),
    updatedAt: null,
    updatedBy: null,
  };
}

function normalizeNavigation(rawNavigation) {
  const rawItems = Array.isArray(rawNavigation) ? rawNavigation : [];
  const rawById = new Map();
  for (const item of rawItems) {
    if (!item || typeof item !== 'object' || !REGISTRY_BY_ID.has(String(item.id || ''))) continue;
    const id = String(item.id);
    if (!rawById.has(id)) rawById.set(id, item);
  }

  return NAVIGATION_REGISTRY.map((definition) => {
    const raw = rawById.get(definition.id) || {};
    const group = NAVIGATION_GROUPS.includes(raw.group) ? raw.group : definition.defaultGroup;
    return {
      id: definition.id,
      label: cleanText(raw.label, 32, definition.defaultLabel),
      group,
      order: integer(raw.order, 0, 1000, definition.defaultOrder),
      visible: raw.visible !== false,
    };
  }).sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
}

function normalizeSiteConfiguration(rawConfiguration) {
  const defaults = defaultSiteConfiguration();
  const raw = rawConfiguration && typeof rawConfiguration === 'object' && !Array.isArray(rawConfiguration)
    ? rawConfiguration
    : {};
  const brand = raw.brand && typeof raw.brand === 'object' && !Array.isArray(raw.brand) ? raw.brand : {};
  const announcement = raw.announcement && typeof raw.announcement === 'object' && !Array.isArray(raw.announcement)
    ? raw.announcement
    : {};
  return {
    revision: integer(raw.revision, 1, Number.MAX_SAFE_INTEGER, defaults.revision),
    brand: {
      name: cleanText(brand.name, 24, defaults.brand.name),
      suffix: cleanText(brand.suffix, 24, defaults.brand.suffix),
    },
    announcement: {
      enabled: announcement.enabled === true,
      text: cleanText(announcement.text, 180),
      tone: ['info', 'success', 'warning'].includes(announcement.tone) ? announcement.tone : 'info',
    },
    navigation: normalizeNavigation(raw.navigation),
    updatedAt: raw.updatedAt || null,
    updatedBy: raw.updatedBy || null,
  };
}

function publicSiteConfiguration(db) {
  const configuration = normalizeSiteConfiguration(db?.siteConfiguration);
  return {
    revision: configuration.revision,
    brand: configuration.brand,
    announcement: configuration.announcement,
    navigation: configuration.navigation.map((item) => {
      const definition = REGISTRY_BY_ID.get(item.id);
      return {
        ...item,
        access: definition.access,
      };
    }),
  };
}

function navigationRegistryForAdmin() {
  return NAVIGATION_REGISTRY.map((item) => ({ ...item }));
}

function validateNavigationInput(rawNavigation) {
  if (!Array.isArray(rawNavigation)) {
    throw new SiteConfigurationError('Navigation must contain the complete page list.');
  }
  const ids = rawNavigation.map((item) => String(item?.id || ''));
  const unknown = ids.filter((id) => !REGISTRY_BY_ID.has(id));
  if (unknown.length) {
    throw new SiteConfigurationError(`Unknown navigation page: ${unknown[0]}. Permanent page IDs cannot be added here.`);
  }
  if (new Set(ids).size !== ids.length) {
    throw new SiteConfigurationError('A navigation page can appear only once.');
  }
  const missing = NAVIGATION_REGISTRY.find((definition) => !ids.includes(definition.id));
  if (missing) {
    throw new SiteConfigurationError(`Navigation is missing the permanent ${missing.id} page.`);
  }
}

function summarizeChanges(before, after) {
  const changes = [];
  function add(field, previous, next) {
    if (JSON.stringify(previous) === JSON.stringify(next)) return;
    changes.push({ field, from: previous, to: next });
  }
  add('Brand name', before.brand.name, after.brand.name);
  add('Brand suffix', before.brand.suffix, after.brand.suffix);
  add('Announcement enabled', before.announcement.enabled, after.announcement.enabled);
  add('Announcement text', before.announcement.text, after.announcement.text);
  add('Announcement tone', before.announcement.tone, after.announcement.tone);

  const beforeById = new Map(before.navigation.map((item) => [item.id, item]));
  for (const item of after.navigation) {
    const previous = beforeById.get(item.id);
    add(`${item.id} label`, previous?.label, item.label);
    add(`${item.id} menu`, previous?.group, item.group);
    add(`${item.id} visibility`, previous?.visible, item.visible);
    add(`${item.id} order`, previous?.order, item.order);
  }
  return changes.slice(0, 50);
}

function updateSiteConfiguration(db, payload, actorId, now = new Date()) {
  if (!db || typeof db !== 'object') throw new SiteConfigurationError('Site storage is unavailable.', 500);
  const before = normalizeSiteConfiguration(db.siteConfiguration);
  const expectedRevision = Number(payload?.revision);
  if (!Number.isInteger(expectedRevision)) {
    throw new SiteConfigurationError('Refresh the editor before saving; its revision is missing.');
  }
  if (expectedRevision !== before.revision) {
    throw new SiteConfigurationError(
      'Another administrator changed the site. Reload the newest version before saving.',
      409,
      'SITE_CONFIGURATION_CONFLICT',
    );
  }
  validateNavigationInput(payload.navigation);

  const candidate = normalizeSiteConfiguration({
    revision: before.revision + 1,
    brand: payload.brand,
    announcement: payload.announcement,
    navigation: payload.navigation,
    updatedAt: now.toISOString(),
    updatedBy: actorId,
  });
  if (candidate.announcement.enabled && !candidate.announcement.text) {
    throw new SiteConfigurationError('Write an announcement before turning it on.');
  }
  const primaryCount = candidate.navigation.filter((item) => item.visible && item.group === 'primary').length;
  if (primaryCount > 5) {
    throw new SiteConfigurationError('Keep at most five visible pages in the main menu. Move the others into More.');
  }

  const changes = summarizeChanges(before, candidate);
  if (!changes.length) return { configuration: before, event: null };
  db.siteConfiguration = candidate;
  if (!Array.isArray(db.siteConfigurationEvents)) db.siteConfigurationEvents = [];
  const event = {
    id: `site_change_${crypto.randomUUID()}`,
    revision: candidate.revision,
    actorId,
    createdAt: candidate.updatedAt,
    changes,
  };
  db.siteConfigurationEvents.push(event);
  if (db.siteConfigurationEvents.length > MAX_HISTORY) {
    db.siteConfigurationEvents.splice(0, db.siteConfigurationEvents.length - MAX_HISTORY);
  }
  return { configuration: candidate, event };
}

function adminSiteConfiguration(db) {
  return {
    configuration: normalizeSiteConfiguration(db?.siteConfiguration),
    registry: navigationRegistryForAdmin(),
    history: (Array.isArray(db?.siteConfigurationEvents) ? db.siteConfigurationEvents : [])
      .slice(-20)
      .reverse(),
  };
}

module.exports = {
  NAVIGATION_GROUPS,
  NAVIGATION_REGISTRY,
  SiteConfigurationError,
  adminSiteConfiguration,
  defaultSiteConfiguration,
  normalizeSiteConfiguration,
  publicSiteConfiguration,
  updateSiteConfiguration,
};
