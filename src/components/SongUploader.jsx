import { parseUploadedSongFile } from '../utils/songParser.js';
import MusicUploadPanel from './MusicUploadPanel.jsx';

export default function SongUploader({ onUpload, user, setUser, onNavigate }) {
  async function loadReadySheet(file) {
    const song = await parseUploadedSongFile(file);
    onUpload(song);
    return song;
  }

  return (
    <section className="uploader-card">
      <MusicUploadPanel
        user={user}
        setUser={setUser}
        onNavigate={onNavigate}
        instrument="piano"
        onReadyFile={loadReadySheet}
      />
    </section>
  );
}
