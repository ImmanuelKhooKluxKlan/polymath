import { parseUploadedSongFile } from '../utils/songParser.js';
import MusicUploadPanel from './MusicUploadPanel.jsx';

export default function SongUploader({ onUpload, user, setUser, onNavigate }) {
  async function loadReadySheet(file) {
    const song = await parseUploadedSongFile(file);
    onUpload(song);
    return song;
  }

  return (
    <section className='uploader-card'>
      <header className='uploader-heading'>
        <p className='eyebrow'>Music source</p>
        <h2>Add music</h2>
        <p className='muted'>Choose one starting point.</p>
      </header>
      <MusicUploadPanel
        user={user}
        setUser={setUser}
        onNavigate={onNavigate}
        instrument='piano'
        onReadyFile={loadReadySheet}
      />
    </section>
  );
}
