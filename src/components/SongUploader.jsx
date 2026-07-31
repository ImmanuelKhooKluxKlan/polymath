import { useState } from 'react';
import { parseLyricsAndChordsToSong } from '../utils/chordParser.js';
import { parseUploadedSongFile, downloadSongTemplate } from '../utils/songParser.js';
import MusicUploadPanel from './MusicUploadPanel.jsx';

const SUPPORTED_READY_EXTENSIONS = /\.(json|csv|mid|midi|musicxml|xml)$/i;

export default function SongUploader({ onUpload, user, setUser, onNavigate }) {
  const [status, setStatus] = useState('Ready for a playable sheet or song/style folder.');

  async function loadReadySheet(file) {
    const song = await parseUploadedSongFile(file);
    onUpload(song);
    return song;
  }

  async function handleNoteFolder(event) {
    const files = [...(event.target.files || [])];
    const file = files.find((candidate) => SUPPORTED_READY_EXTENSIONS.test(candidate.name));

    if (!file) {
      setStatus('That folder does not contain a supported ready-to-play JSON, CSV, MIDI, or MusicXML file.');
      event.target.value = '';
      return;
    }

    try {
      const folderName = String(file.webkitRelativePath || '').split('/').filter(Boolean)[0] || 'Uploaded folder';
      setStatus(`Reading ${folderName}/${file.name}…`);
      const song = await parseUploadedSongFile(file);
      onUpload({
        ...song,
        sourceFolderName: folderName,
        youtubeSearchQuery: song.youtubeSearchQuery || folderName,
      });
      setStatus(`Loaded ${song.title} from ${folderName}. YouTube comparison will search that style automatically.`);
    } catch (error) {
      setStatus(error.message || 'Folder import failed.');
    } finally {
      event.target.value = '';
    }
  }

  async function handleLyricsUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsedSong = parseLyricsAndChordsToSong(text);
      if (!parsedSong.notes.length) throw new Error('No chords found. Try a lyrics file with chords such as C, G, Am, or F.');
      onUpload(parsedSong);
      setStatus(`Loaded chord sheet · ${parsedSong.notes.length} chord notes.`);
    } catch (error) {
      setStatus(error.message || 'Chord import failed.');
    } finally {
      event.target.value = '';
    }
  }

  return (
    <section className="uploader-card">
      <div>
        <p className="eyebrow">Music sheet tools</p>
        <h2>Play or translate your sheet</h2>
        <p className="muted">Use plain-language upload options. Ready-to-play sheets open immediately; PDFs are translated separately.</p>
      </div>

      <MusicUploadPanel
        user={user}
        setUser={setUser}
        onNavigate={onNavigate}
        instrument="piano"
        onReadyFile={loadReadySheet}
      />

      <label className="upload-folder-button">
        <input
          type="file"
          accept=".json,.csv,.mid,.midi,.musicxml,.xml"
          multiple
          webkitdirectory=""
          directory=""
          onChange={handleNoteFolder}
        />
        Choose a ready-to-play song/style folder
      </label>

      <p className="import-status">{status}</p>

      <button className="ghost full" type="button" onClick={downloadSongTemplate}>
        Download ready-to-play CSV template
      </button>

      <div className="upload-divider" />

      <div>
        <p className="eyebrow">Quick chord demo</p>
        <h2>Lyrics / Chord Sheet</h2>
        <p className="muted">Upload a text file with chords such as C, G, Am, F, or Dm7 and the piano will play them.</p>
      </div>

      <label className="upload-box compact">
        <input type="file" accept=".txt,.lyrics,.chords" onChange={handleLyricsUpload} />
        <span>Choose lyrics / chord file</span>
      </label>
    </section>
  );
}
