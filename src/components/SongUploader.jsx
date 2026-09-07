import { parseUploadedSongFile } from '../utils/songParser.js';
import MusicUploadPanel from './MusicUploadPanel.jsx';
import MusicChoiceDisclosure from './MusicChoiceDisclosure.jsx';

export default function SongUploader({ onUpload, user, setUser, onNavigate, expanded, onToggle, onPersonalSongSaved }) {
  async function loadReadySheet(file, { commit = true, prepared = null } = {}) {
    const song = prepared || await parseUploadedSongFile(file);
    if (!commit) return song;
    onUpload(song);
    return song;
  }

  return (
    <section className='uploader-card'>
      <MusicChoiceDisclosure
        id="piano-choose-music"
        title="Upload music sheet or video"
        expanded={expanded}
        onToggle={onToggle}
      >
        <MusicUploadPanel
          user={user}
          setUser={setUser}
          onNavigate={onNavigate}
          instrument='piano'
          onReadyFile={loadReadySheet}
          onPersonalSongSaved={onPersonalSongSaved}
        />
      </MusicChoiceDisclosure>
    </section>
  );
}
