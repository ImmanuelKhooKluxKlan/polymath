import { useEffect, useMemo, useState } from 'react';
import { INSTRUMENTS, instrumentLabel } from '../data/instruments.js';
import { apiRequest, fileToBase64 } from '../services/api.js';
import { announceFeaturedSongsUpdated } from '../services/featuredSongs.js';
import { userFacingError } from '../utils/userFacingError.js';
import TaskProgress from './TaskProgress.jsx';

const EMPTY_DRAFT = Object.freeze({
  title: '',
  artist: '',
  instrument: 'piano',
  sortOrder: '',
  active: true,
  rightsConfirmed: false,
});

function fileSizeLabel(bytes) {
  const size = Math.max(0, Number(bytes) || 0);
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 ** 2).toFixed(1)} MB`;
}

export default function FeaturedSongAdmin() {
  const [songs, setSongs] = useState([]);
  const [draft, setDraft] = useState(EMPTY_DRAFT);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Loading available songs...');
  const [editingId, setEditingId] = useState('');
  const [editDraft, setEditDraft] = useState(null);

  const activeCount = useMemo(
    () => songs.filter((song) => song.active !== false).length,
    [songs],
  );

  async function refresh() {
    const data = await apiRequest('/api/admin/featured-songs');
    setSongs(Array.isArray(data.songs) ? data.songs : []);
    return data;
  }

  useEffect(() => {
    let cancelled = false;
    apiRequest('/api/admin/featured-songs')
      .then((data) => {
        if (cancelled) return;
        setSongs(Array.isArray(data.songs) ? data.songs : []);
        setStatus('');
      })
      .catch((error) => {
        if (!cancelled) setStatus(userFacingError(error));
      });
    return () => { cancelled = true; };
  }, []);

  async function addSong(event) {
    event.preventDefault();
    if (!file) {
      setStatus('Choose a ready-to-play JSON or MIDI file.');
      return;
    }
    if (!draft.rightsConfirmed) {
      setStatus('Confirm publishing rights before adding this song.');
      return;
    }
    setBusy(true);
    setStatus('');
    try {
      await apiRequest('/api/admin/featured-songs', {
        method: 'POST',
        body: JSON.stringify({
          ...draft,
          sortOrder: draft.sortOrder === '' ? undefined : Number(draft.sortOrder),
          filename: file.name,
          contentBase64: await fileToBase64(file),
        }),
      });
      await refresh();
      announceFeaturedSongsUpdated();
      setDraft(EMPTY_DRAFT);
      setFile(null);
      const input = document.getElementById('featured-song-file');
      if (input) input.value = '';
      setStatus('Song added to the public available-song library.');
    } catch (error) {
      setStatus(userFacingError(error, 'This song could not be added.'));
    } finally {
      setBusy(false);
    }
  }

  async function updateSong(songId, changes, successMessage) {
    setBusy(true);
    setStatus('');
    try {
      await apiRequest(`/api/admin/featured-songs/${encodeURIComponent(songId)}`, {
        method: 'PATCH',
        body: JSON.stringify(changes),
      });
      await refresh();
      announceFeaturedSongsUpdated();
      setEditingId('');
      setEditDraft(null);
      setStatus(successMessage);
    } catch (error) {
      setStatus(userFacingError(error, 'This available song could not be updated.'));
    } finally {
      setBusy(false);
    }
  }

  function beginEdit(song) {
    setEditingId(song.id);
    setEditDraft({
      title: song.title,
      artist: song.artist || '',
      instrument: song.instrument,
      sortOrder: song.sortOrder ?? 0,
    });
  }

  async function saveEdit(event) {
    event.preventDefault();
    if (!editingId || !editDraft) return;
    await updateSong(editingId, {
      ...editDraft,
      sortOrder: Number(editDraft.sortOrder),
    }, 'Song details updated.');
  }

  async function deleteSong(song) {
    if (!window.confirm(`Delete “${song.title}” from every user's available songs?`)) return;
    setBusy(true);
    setStatus('');
    try {
      await apiRequest(`/api/admin/featured-songs/${encodeURIComponent(song.id)}`, {
        method: 'DELETE',
      });
      await refresh();
      announceFeaturedSongsUpdated();
      setStatus('Song deleted from the public library.');
    } catch (error) {
      setStatus(userFacingError(error, 'This available song could not be deleted.'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="featured-song-admin">
      <header className="admin-section-heading">
        <div>
          <p className="eyebrow">Public starter catalogue</p>
          <h2>Available songs</h2>
          <p>Add free ready-to-play songs to Piano, Guitar, or one specific instrument.</p>
        </div>
        <span className="status-pill">{activeCount} live · {songs.length} total</span>
      </header>

      <form className="admin-form-card featured-song-upload-form" onSubmit={addSong}>
        <div>
          <h3>Add an available song</h3>
          <p className="muted">The catalogue stays lightweight. The music file downloads only after a listener selects it.</p>
        </div>
        <div className="admin-form-grid featured-song-form-grid">
          <label className="field">Song title
            <input value={draft.title} maxLength="160" onChange={(event) => setDraft({ ...draft, title: event.target.value })} placeholder="Read from JSON when blank" />
          </label>
          <label className="field">Artist / composer
            <input value={draft.artist} maxLength="120" onChange={(event) => setDraft({ ...draft, artist: event.target.value })} placeholder="Artist name" />
          </label>
          <label className="field">Show under instrument
            <select value={draft.instrument} onChange={(event) => setDraft({ ...draft, instrument: event.target.value })}>
              {INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}
            </select>
          </label>
          <label className="field">Display order
            <input type="number" value={draft.sortOrder} onChange={(event) => setDraft({ ...draft, sortOrder: event.target.value })} placeholder="Automatic" />
          </label>
          <label className="field">Ready-to-play file
            <input id="featured-song-file" type="file" accept=".json,.mid,.midi,application/json,audio/midi,audio/x-midi" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          </label>
        </div>
        <div className="featured-song-publish-checks">
          <label className="rights-check"><input type="checkbox" checked={draft.active} onChange={(event) => setDraft({ ...draft, active: event.target.checked })} /><span>Publish immediately</span></label>
          <label className="rights-check"><input type="checkbox" checked={draft.rightsConfirmed} onChange={(event) => setDraft({ ...draft, rightsConfirmed: event.target.checked })} /><span>I confirm Polymath may publish this file to all users.</span></label>
        </div>
        <button className="primary" type="submit" disabled={busy || !file || !draft.rightsConfirmed}>Add to available songs</button>
        {busy && <TaskProgress compact label="Adding song safely..." />}
      </form>

      {status && <p className="form-status" role="status">{status}</p>}

      <div className="featured-song-admin-list">
        {!songs.length && !busy && <div className="empty-state">No administrator-added songs yet. Built-in starter songs remain available.</div>}
        {songs.map((song) => (
          <article className={`featured-song-admin-row ${song.active === false ? 'is-hidden' : ''}`} key={song.id}>
            {editingId === song.id ? (
              <form className="featured-song-edit-form" onSubmit={saveEdit}>
                <label className="field">Title<input required maxLength="160" value={editDraft.title} onChange={(event) => setEditDraft({ ...editDraft, title: event.target.value })} /></label>
                <label className="field">Artist<input maxLength="120" value={editDraft.artist} onChange={(event) => setEditDraft({ ...editDraft, artist: event.target.value })} /></label>
                <label className="field">Instrument<select value={editDraft.instrument} onChange={(event) => setEditDraft({ ...editDraft, instrument: event.target.value })}>{INSTRUMENTS.map((instrument) => <option key={instrument.id} value={instrument.id}>{instrument.label}</option>)}</select></label>
                <label className="field">Order<input type="number" value={editDraft.sortOrder} onChange={(event) => setEditDraft({ ...editDraft, sortOrder: event.target.value })} /></label>
                <div className="button-row"><button className="primary compact-action" type="submit" disabled={busy}>Save</button><button className="ghost compact-action" type="button" onClick={() => { setEditingId(''); setEditDraft(null); }}>Cancel</button></div>
              </form>
            ) : (
              <>
                <div className="featured-song-admin-copy">
                  <span className="status-pill">{song.active === false ? 'Hidden' : 'Live'}</span>
                  <div><strong>{song.title}</strong><small>{song.artist || 'No artist'} · {instrumentLabel(song.instrument)}</small></div>
                  <small>{song.format} · {fileSizeLabel(song.size)} · order {song.sortOrder}</small>
                </div>
                <div className="featured-song-admin-actions">
                  <button className="ghost compact-action" type="button" disabled={busy} onClick={() => beginEdit(song)}>Edit</button>
                  <button className="ghost compact-action" type="button" disabled={busy} onClick={() => updateSong(song.id, { active: song.active === false }, song.active === false ? 'Song published.' : 'Song hidden from users.')}>{song.active === false ? 'Publish' : 'Hide'}</button>
                  <button className="ghost compact-action danger-outline" type="button" disabled={busy} onClick={() => deleteSong(song)}>Delete</button>
                </div>
              </>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
