import { useEffect, useMemo, useState } from 'react';
import { apiRequest } from '../services/api.js';
import '../adminControlCenter.css';

function editableSnapshot(configuration) {
  if (!configuration) return '';
  return JSON.stringify({
    brand: configuration.brand,
    announcement: configuration.announcement,
    navigation: configuration.navigation,
  });
}

function resequence(items) {
  const ordered = ['primary', 'more'].flatMap((group) => (
    items
      .filter((item) => item.group === group)
      .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id))
  ));
  return ordered.map((item, index) => ({ ...item, order: (index + 1) * 10 }));
}

function shortChange(change) {
  if (String(change.field).endsWith(' visibility')) return `${change.field}: ${change.to ? 'shown' : 'hidden'}`;
  return `${change.field}: ${String(change.to ?? 'empty')}`;
}

function formatWhen(value) {
  if (!value) return 'Never';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? 'Unknown time' : parsed.toLocaleString();
}

export default function SiteControlCenter() {
  const [saved, setSaved] = useState(null);
  const [draft, setDraft] = useState(null);
  const [registry, setRegistry] = useState([]);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Loading website controls...');
  const [statusKind, setStatusKind] = useState('neutral');
  const [navigationSearch, setNavigationSearch] = useState('');

  async function loadConfiguration() {
    setBusy(true);
    setStatus('Loading the newest website settings...');
    setStatusKind('neutral');
    try {
      const data = await apiRequest('/api/admin/site-configuration');
      setSaved(data.configuration);
      setDraft(data.configuration);
      setRegistry(data.registry || []);
      setHistory(data.history || []);
      setStatus('All website controls are up to date.');
      setStatusKind('success');
    } catch (error) {
      setStatus(error.message);
      setStatusKind('error');
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    loadConfiguration();
  }, []);

  const hasChanges = editableSnapshot(draft) !== editableSnapshot(saved);
  useEffect(() => {
    function warnBeforeLeave(event) {
      if (!hasChanges) return;
      event.preventDefault();
      event.returnValue = '';
    }
    window.addEventListener('beforeunload', warnBeforeLeave);
    return () => window.removeEventListener('beforeunload', warnBeforeLeave);
  }, [hasChanges]);

  const registryById = useMemo(() => new Map(registry.map((item) => [item.id, item])), [registry]);
  const orderedNavigation = useMemo(
    () => resequence(draft?.navigation || []),
    [draft?.navigation],
  );
  const filteredNavigation = useMemo(() => {
    const query = navigationSearch.trim().toLowerCase();
    if (!query) return orderedNavigation;
    return orderedNavigation.filter((item) => {
      const definition = registryById.get(item.id);
      return [item.label, item.id, definition?.path, definition?.access]
        .some((value) => String(value || '').toLowerCase().includes(query));
    });
  }, [navigationSearch, orderedNavigation, registryById]);
  const primaryPreview = orderedNavigation.filter((item) => item.visible && item.group === 'primary');
  const morePreview = orderedNavigation.filter((item) => item.visible && item.group === 'more');
  const hiddenCount = orderedNavigation.filter((item) => !item.visible).length;

  function updateDraft(section, changes) {
    setDraft((current) => ({
      ...current,
      [section]: { ...current[section], ...changes },
    }));
  }

  function updateNavigation(id, changes) {
    setDraft((current) => ({
      ...current,
      navigation: resequence(current.navigation.map((item) => (
        item.id === id ? { ...item, ...changes } : item
      ))),
    }));
  }

  function moveNavigation(id, direction) {
    setDraft((current) => {
      const source = resequence(current.navigation);
      const selected = source.find((item) => item.id === id);
      const groupItems = source.filter((item) => item.group === selected?.group);
      const index = groupItems.findIndex((item) => item.id === id);
      const targetIndex = index + direction;
      if (!selected || targetIndex < 0 || targetIndex >= groupItems.length) return current;
      [groupItems[index], groupItems[targetIndex]] = [groupItems[targetIndex], groupItems[index]];
      const groupOrder = new Map(groupItems.map((item, itemIndex) => [item.id, itemIndex]));
      return {
        ...current,
        navigation: resequence(source.sort((left, right) => {
          if (left.group !== selected.group || right.group !== selected.group) return 0;
          return groupOrder.get(left.id) - groupOrder.get(right.id);
        }).map((item) => (
          item.group === selected.group
            ? { ...item, order: (groupOrder.get(item.id) + 1) * 10 }
            : item
        ))),
      };
    });
  }

  function resetNavigationItem(id) {
    const definition = registryById.get(id);
    if (!definition) return;
    updateNavigation(id, {
      label: definition.defaultLabel,
      group: definition.defaultGroup,
      order: definition.defaultOrder,
      visible: true,
    });
  }

  async function saveConfiguration(event) {
    event.preventDefault();
    if (!draft || !hasChanges) return;
    setBusy(true);
    setStatus('Publishing website settings...');
    setStatusKind('neutral');
    try {
      const data = await apiRequest('/api/admin/site-configuration', {
        method: 'PUT',
        body: JSON.stringify({ ...draft, navigation: resequence(draft.navigation) }),
      });
      setSaved(data.configuration);
      setDraft(data.configuration);
      setRegistry(data.registry || []);
      setHistory(data.history || []);
      setStatus(data.message);
      setStatusKind('success');
      window.dispatchEvent(new window.CustomEvent('polymath-site-configuration-updated', {
        detail: data.configuration,
      }));
    } catch (error) {
      setStatus(error.message);
      setStatusKind('error');
    } finally {
      setBusy(false);
    }
  }

  if (!draft) {
    return (
      <div className='site-control-loading' role='status'>
        <span className='site-control-spinner' aria-hidden='true' />
        <strong>{status}</strong>
        {statusKind === 'error' && <button className='ghost' type='button' onClick={loadConfiguration}>Try again</button>}
      </div>
    );
  }

  return (
    <form className='site-control-center' onSubmit={saveConfiguration}>
      <header className='site-control-heading'>
        <div>
          <p className='eyebrow'>Website editor</p>
          <h2>Site &amp; navigation</h2>
          <p>Rename and arrange what visitors see. Permanent URLs and access rules stay protected.</p>
        </div>
        <div className='site-control-actions'>
          <span className={`site-save-state ${hasChanges ? 'dirty' : ''}`}>{hasChanges ? 'Unsaved changes' : 'Saved'}</span>
          <button className='ghost' type='button' disabled={busy} onClick={loadConfiguration}>Reload</button>
          <button className='primary' type='submit' disabled={busy || !hasChanges}>{busy ? 'Saving...' : 'Publish changes'}</button>
        </div>
      </header>

      <div className='site-control-overview-grid'>
        <section className='site-control-card site-identity-card'>
          <div className='site-control-card-heading'>
            <span className='site-control-card-icon' aria-hidden='true'>Aa</span>
            <div><h3>Site identity</h3><p>Change the name shown in the top navigation.</p></div>
          </div>
          <div className='site-control-two-fields'>
            <label className='field'>Main name<input required maxLength='24' value={draft.brand.name} onChange={(event) => updateDraft('brand', { name: event.target.value })} /></label>
            <label className='field'>Second line<input required maxLength='24' value={draft.brand.suffix} onChange={(event) => updateDraft('brand', { suffix: event.target.value })} /></label>
          </div>
        </section>

        <section className='site-control-card site-announcement-editor'>
          <div className='site-control-card-heading'>
            <span className='site-control-card-icon' aria-hidden='true'>!</span>
            <div><h3>Visitor announcement</h3><p>Show one short message above the website.</p></div>
          </div>
          <label className='site-control-switch'><input type='checkbox' checked={draft.announcement.enabled} onChange={(event) => updateDraft('announcement', { enabled: event.target.checked })} /><span>Show announcement</span></label>
          <label className='field'>Message<input maxLength='180' placeholder='New lessons are now available.' value={draft.announcement.text} onChange={(event) => updateDraft('announcement', { text: event.target.value })} /></label>
          <label className='field'>Style<select value={draft.announcement.tone} onChange={(event) => updateDraft('announcement', { tone: event.target.value })}><option value='info'>Information</option><option value='success'>Good news</option><option value='warning'>Important</option></select></label>
        </section>
      </div>

      <section className='site-control-card site-navigation-editor'>
        <header className='site-navigation-heading'>
          <div><h3>Navigation menu</h3><p>Visitors only see pages marked visible. Hidden pages keep their permanent URL.</p></div>
          <label className='site-navigation-search'><span>Find a page</span><input type='search' value={navigationSearch} onChange={(event) => setNavigationSearch(event.target.value)} placeholder='Search label or URL' /></label>
        </header>
        <div className='site-navigation-labels' aria-hidden='true'>
          <span>Order</span><span>Visitor label</span><span>Menu</span><span>Visibility</span><span>Protected URL</span><span>Reset</span>
        </div>
        <div className='site-navigation-list'>
          {filteredNavigation.map((item) => {
            const definition = registryById.get(item.id);
            const groupItems = orderedNavigation.filter((candidate) => candidate.group === item.group);
            const groupIndex = groupItems.findIndex((candidate) => candidate.id === item.id);
            return (
              <article className={`site-navigation-row ${item.visible ? '' : 'hidden-page'}`} key={item.id}>
                <div className='site-navigation-order' aria-label={`Move ${item.label}`}>
                  <button type='button' disabled={groupIndex <= 0} aria-label={`Move ${item.label} up`} onClick={() => moveNavigation(item.id, -1)}>↑</button>
                  <button type='button' disabled={groupIndex === groupItems.length - 1} aria-label={`Move ${item.label} down`} onClick={() => moveNavigation(item.id, 1)}>↓</button>
                </div>
                <label className='field site-navigation-name'><span>Visitor label</span><input required maxLength='32' value={item.label} onChange={(event) => updateNavigation(item.id, { label: event.target.value })} /></label>
                <label className='field site-navigation-group'><span>Menu</span><select value={item.group} onChange={(event) => updateNavigation(item.id, { group: event.target.value })}><option value='primary'>Main menu</option><option value='more'>More menu</option></select></label>
                <label className='site-control-switch site-navigation-visibility'><input type='checkbox' checked={item.visible} onChange={(event) => updateNavigation(item.id, { visible: event.target.checked })} /><span>{item.visible ? 'Shown' : 'Hidden'}</span></label>
                <div className='site-locked-route'>
                  <code>{definition?.path || `#${item.id}`}</code>
                  <small>{definition?.access === 'signed-in' ? 'Sign-in required' : 'Public page'} · locked</small>
                </div>
                <button className='ghost compact-action' type='button' onClick={() => resetNavigationItem(item.id)}>Reset</button>
              </article>
            );
          })}
          {!filteredNavigation.length && <div className='site-navigation-empty'>No page matches “{navigationSearch}”.</div>}
        </div>
      </section>

      <div className='site-control-lower-grid'>
        <section className='site-control-card site-menu-preview'>
          <div className='site-control-card-heading'>
            <span className='site-control-card-icon' aria-hidden='true'>👁</span>
            <div><h3>Live menu preview</h3><p>Preview the structure before publishing.</p></div>
          </div>
          {draft.announcement.enabled && draft.announcement.text && <div className={`site-preview-announcement ${draft.announcement.tone}`}>{draft.announcement.text}</div>}
          <div className='site-preview-browser'>
            <div className='site-preview-browser-bar'><i /><i /><i /><span>polymathmusician67.com</span></div>
            <div className='site-preview-menu'>
              <strong><b>{draft.brand.name.charAt(0)}{draft.brand.suffix.charAt(0)}</b><span>{draft.brand.name}<small>{draft.brand.suffix}</small></span></strong>
              <nav>{primaryPreview.map((item) => <span key={item.id}>{item.label}</span>)}<span className='preview-more'>More ({morePreview.length + 1})</span></nav>
            </div>
          </div>
          <div className='site-preview-summary'><span>{primaryPreview.length} main</span><span>{morePreview.length} in More</span><span>{hiddenCount} hidden</span></div>
        </section>

        <section className='site-control-card site-change-history'>
          <div className='site-control-card-heading'>
            <span className='site-control-card-icon' aria-hidden='true'>↺</span>
            <div><h3>Recent changes</h3><p>Latest administrator publishing activity.</p></div>
          </div>
          <div className='site-history-list'>
            {history.slice(0, 6).map((event) => (
              <details key={event.id}>
                <summary><span><strong>Revision {event.revision}</strong><small>{event.actor?.name || event.actor?.email || 'Administrator'}</small></span><time>{formatWhen(event.createdAt)}</time></summary>
                <ul>{event.changes.map((change, index) => <li key={`${change.field}-${index}`}>{shortChange(change)}</li>)}</ul>
              </details>
            ))}
            {!history.length && <p className='site-history-empty'>The first published change will appear here.</p>}
          </div>
        </section>
      </div>

      <footer className={`site-control-status ${statusKind}`} role='status'>
        <span aria-hidden='true'>{statusKind === 'error' ? '!' : statusKind === 'success' ? '✓' : '•'}</span>
        <p>{status}</p>
        <small>Revision {draft.revision} · last changed {formatWhen(draft.updatedAt)}</small>
      </footer>
    </form>
  );
}
