export const DEFAULT_SITE_CONFIGURATION = Object.freeze({
  revision: 1,
  brand: Object.freeze({ name: 'Polymath', suffix: 'Musician' }),
  announcement: Object.freeze({ enabled: false, text: '', tone: 'info' }),
  navigation: Object.freeze([
    Object.freeze({ id: 'studio', label: 'Piano', group: 'primary', order: 10, visible: true, access: 'public' }),
    Object.freeze({ id: 'create-music', label: 'Create Music', group: 'primary', order: 20, visible: true, access: 'public' }),
    Object.freeze({ id: 'guitar', label: 'Guitar', group: 'primary', order: 30, visible: true, access: 'public' }),
    Object.freeze({ id: 'ensemble', label: 'Instruments', group: 'primary', order: 40, visible: true, access: 'public' }),
    Object.freeze({ id: 'published-songs', label: 'Composers', group: 'more', order: 50, visible: true, access: 'public' }),
    Object.freeze({ id: 'find-teacher', label: 'Find Teacher', group: 'more', order: 60, visible: true, access: 'public' }),
    Object.freeze({ id: 'band', label: 'Band', group: 'more', order: 70, visible: true, access: 'public' }),
    Object.freeze({ id: 'your-songs', label: 'Your Songs', group: 'more', order: 80, visible: true, access: 'signed-in' }),
  ]),
});

function cleanLabel(value, fallback) {
  const label = String(value || '').replace(/[<>]/g, '').replace(/\s+/g, ' ').trim().slice(0, 32);
  return label || fallback;
}

export function normalizePublicSiteConfiguration(raw) {
  const source = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const rawNavigation = Array.isArray(source.navigation) ? source.navigation : [];
  const rawById = new Map(rawNavigation.map((item) => [String(item?.id || ''), item]));
  const brand = source.brand && typeof source.brand === 'object' ? source.brand : {};
  const announcement = source.announcement && typeof source.announcement === 'object' ? source.announcement : {};
  return {
    revision: Number.isInteger(source.revision) ? source.revision : DEFAULT_SITE_CONFIGURATION.revision,
    brand: {
      name: cleanLabel(brand.name, DEFAULT_SITE_CONFIGURATION.brand.name).slice(0, 24),
      suffix: cleanLabel(brand.suffix, DEFAULT_SITE_CONFIGURATION.brand.suffix).slice(0, 24),
    },
    announcement: {
      enabled: announcement.enabled === true && Boolean(String(announcement.text || '').trim()),
      text: String(announcement.text || '').replace(/[<>]/g, '').replace(/\s+/g, ' ').trim().slice(0, 180),
      tone: ['info', 'success', 'warning'].includes(announcement.tone) ? announcement.tone : 'info',
    },
    navigation: DEFAULT_SITE_CONFIGURATION.navigation.map((fallback) => {
      const item = rawById.get(fallback.id) || {};
      return {
        id: fallback.id,
        label: cleanLabel(item.label, fallback.label),
        group: ['primary', 'more'].includes(item.group) ? item.group : fallback.group,
        order: Number.isFinite(Number(item.order)) ? Math.max(0, Math.min(1000, Math.floor(Number(item.order)))) : fallback.order,
        visible: item.visible !== false,
        access: item.access === 'signed-in' || fallback.access === 'signed-in' ? 'signed-in' : 'public',
      };
    }).sort((left, right) => left.order - right.order || left.id.localeCompare(right.id)),
  };
}
