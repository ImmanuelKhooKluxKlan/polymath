export const LIVE_PRODUCT_FEATURES = Object.freeze({
  band: false,
  createMusic: false,
  instrumentLearning: false,
});

const HIDDEN_PUBLIC_PAGES = new Set([
  ...(!LIVE_PRODUCT_FEATURES.band ? ['band'] : []),
  ...(!LIVE_PRODUCT_FEATURES.createMusic ? ['create-music'] : []),
  ...(!LIVE_PRODUCT_FEATURES.instrumentLearning ? ['teacher-ar', 'teacher-projection'] : []),
]);

export function resolveLivePage(page) {
  const requested = String(page || 'studio').trim() || 'studio';
  if (requested === 'learn') return 'find-teacher';
  if (HIDDEN_PUBLIC_PAGES.has(requested)) return 'studio';
  return requested;
}

export function applyLiveNavigationPolicy(navigation = []) {
  return navigation.map((item) => {
    if (item.id === 'find-teacher') return { ...item, label: 'Learn', visible: true };
    if (HIDDEN_PUBLIC_PAGES.has(item.id)) return { ...item, visible: false };
    return item;
  });
}
