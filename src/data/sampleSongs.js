import { Midi } from '@tonejs/midi';
const FREE_PIANO_SONGS = [
  {
    url: 'songs/BS.mid',
    title: 'Blank Space',
    composer: 'Taylor Swift',
  },
  {
    url: 'songs/Enchanted.midi',
    title: 'Enchanted',
    composer: 'Taylor Swift',
  },
  {
    url: 'songs/LS.mid',
    title: 'Love story',
    composer: 'Taylor Swift',
  },
  {
    url: 'songs/WANGBT.mid',
    title: 'We Are Never Getting Back Together',
    composer: 'Taylor Swift',
  },
    {
    url: 'songs/TWILY.mid',
    title: 'The way i loved you',
    composer: 'Taylor Swift',
  },
  {
    url: 'songs/stayStayStayTaylorSwift.json',
    title: 'Stay Stay Stay',
    composer: 'Taylor Swift',
  },
];

function songUrl(relativeUrl) {
  const baseUrl = import.meta.env.BASE_URL || '/';
  const cleanBaseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const cleanRelativeUrl = relativeUrl.replace(/^\/+/, '');

  return `${cleanBaseUrl}${cleanRelativeUrl}`;
}

function isMidiFile(url = '') {
  return /\.(mid|midi)(?:\?.*)?$/i.test(url);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function convertMidiToSong(arrayBuffer, entry) {
  const midi = new Midi(arrayBuffer);

  const notes = midi.tracks
    .flatMap((track, trackIndex) =>
      track.notes.map((midiNote, noteIndex) => ({
        id: `midi-${trackIndex}-${noteIndex}`,
        note: midiNote.name,
        time: Number(midiNote.time.toFixed(6)),
        duration: Math.max(
          0.035,
          Number(midiNote.duration.toFixed(6)),
        ),
        velocity: clamp(
          Number(midiNote.velocity.toFixed(4)),
          0.05,
          1,
        ),

        // Notes below middle C are treated as left hand.
        hand: midiNote.midi < 60 ? 'left' : 'right',
      })),
    )
    .sort((firstNote, secondNote) => {
      if (firstNote.time !== secondNote.time) {
        return firstNote.time - secondNote.time;
      }

      return firstNote.note.localeCompare(secondNote.note);
    });

  if (!notes.length) {
    throw new Error(`${entry.title} does not contain playable MIDI notes.`);
  }

  const firstTempo = midi.header.tempos?.[0];

  return {
    title: entry.title || midi.name || 'Untitled MIDI song',
    composer: entry.composer || 'Unknown composer',
    bpm: Math.round(firstTempo?.bpm || 120),

    instrument: 'piano',
    libraryType: 'free',
    readyToPlay: true,
    sourceType: 'midi',

    performance: {
      preserveScoreDurations: true,
      defaultAutoplayReleaseSeconds: 0.56,
      sameKeyRetriggerGapSeconds: 0.025,
    },

    notes,
  };
}

async function loadSongEntry(entry) {
  const url = songUrl(entry.url);
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Could not load ${entry.title}. HTTP ${response.status}.`,
    );
  }

  if (isMidiFile(entry.url)) {
    const arrayBuffer = await response.arrayBuffer();
    return convertMidiToSong(arrayBuffer, entry);
  }

  const song = await response.json();

  if (!Array.isArray(song.notes)) {
    throw new Error(`${entry.title} does not contain a notes array.`);
  }

  return {
    ...song,
    title: entry.title || song.title,
    composer:
      entry.composer ||
      song.composer ||
      song.artist ||
      'Unknown composer',

    instrument: 'piano',
    libraryType: 'free',
    readyToPlay: true,
    sourceType: 'json',
  };
}

export async function loadFeaturedSongs() {
  const results = await Promise.allSettled(
    FREE_PIANO_SONGS.map((entry) => loadSongEntry(entry)),
  );

  const loaded = [];
  const failed = [];

  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      loaded.push(result.value);
      return;
    }

    const entry = FREE_PIANO_SONGS[index];

    failed.push({
      title: entry.title,
      url: entry.url,
      reason: result.reason,
    });

    console.error(
      `Failed to load "${entry.title}" from "${entry.url}":`,
      result.reason,
    );
  });

  if (!loaded.length) {
    throw new Error('Could not load the free piano song library.');
  }

  if (failed.length) {
    console.warn(
      `${failed.length} free song file(s) failed to load.`,
      failed,
    );
  }

  return loaded;
}
export const sampleSongs = [
  {
    title: 'Neon C Major Warmup',
    composer: 'Polymath Musician',
    bpm: 88,
    instrument: 'piano',
    libraryType: 'free',
    performance: {
      preserveScoreDurations: true,
      defaultAutoplayReleaseSeconds: 0.56,
    },
    notes: [
      { note: 'C4', time: 0, duration: 0.45, velocity: 0.8, hand: 'left' },
      { note: 'E4', time: 0.45, duration: 0.45, velocity: 0.8, hand: 'right' },
      { note: 'G4', time: 0.9, duration: 0.45, velocity: 0.8, hand: 'right' },
      { note: 'C5', time: 1.35, duration: 0.7, velocity: 0.9, hand: 'right' },
      { note: 'G4', time: 2.1, duration: 0.45, velocity: 0.75, hand: 'right' },
      { note: 'E4', time: 2.55, duration: 0.45, velocity: 0.75, hand: 'right' },
      { note: 'C4', time: 3, duration: 0.8, velocity: 0.85, hand: 'left' },
      { note: 'C3', time: 3, duration: 0.8, velocity: 0.75, hand: 'left' },
      { note: 'F4', time: 4, duration: 0.45, velocity: 0.8, hand: 'right' },
      { note: 'A4', time: 4.45, duration: 0.45, velocity: 0.8, hand: 'right' },
      { note: 'C5', time: 4.9, duration: 0.45, velocity: 0.85, hand: 'right' },
      { note: 'A4', time: 5.35, duration: 0.45, velocity: 0.8, hand: 'right' },
      { note: 'F4', time: 5.8, duration: 0.8, velocity: 0.75, hand: 'right' },
      { note: 'F3', time: 5.8, duration: 0.8, velocity: 0.7, hand: 'left' },
      { note: 'G3', time: 7, duration: 0.6, velocity: 0.75, hand: 'left' },
      { note: 'B4', time: 7, duration: 0.6, velocity: 0.85, hand: 'right' },
      { note: 'C5', time: 7.75, duration: 1.2, velocity: 0.95, hand: 'right' },
      { note: 'C4', time: 7.75, duration: 1.2, velocity: 0.8, hand: 'left' },
    ],
  },
  {
    title: 'Repeated Key Human Test',
    composer: 'Polymath Musician',
    bpm: 80,
    instrument: 'piano',
    libraryType: 'free',
    performance: {
      preserveScoreDurations: true,
      defaultAutoplayReleaseSeconds: 0.5,
      sameKeyRetriggerGapSeconds: 0.025,
    },
    notes: [
      { note: 'C4', time: 0, duration: 2, velocity: 0.86, hand: 'right' },
      { note: 'C4', time: 1, duration: 0.8, velocity: 0.9, hand: 'right' },
      { note: 'E4', time: 1, duration: 1.4, velocity: 0.82, hand: 'right' },
      { note: 'G4', time: 1, duration: 1.4, velocity: 0.84, hand: 'right' },
      { note: 'C3', time: 1.2, duration: 2.2, velocity: 0.75, hand: 'left' },
      { note: 'C4', time: 2.05, duration: 1.4, velocity: 0.88, hand: 'right' },
      { note: 'G3', time: 3, duration: 1.7, velocity: 0.76, hand: 'left' },
      { note: 'B3', time: 3.15, duration: 0.6, velocity: 0.78, hand: 'right' },
      { note: 'D4', time: 3.45, duration: 0.8, velocity: 0.82, hand: 'right' },
      { note: 'G4', time: 3.8, duration: 1.1, velocity: 0.88, hand: 'right' },
    ],
  },
];