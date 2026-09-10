import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applySongStyle,
  applySongBlueprint,
  buildSongArrangement,
  CREATE_MUSIC_STYLES,
  CREATE_MUSIC_VOICES,
  createBlankMusicProject,
  lyricsAsText,
  lyricsFromText,
} from '../../src/engine/songCreationEngine.js';

function blueprint() {
  return {
    title: 'City after rain',
    genre: 'Pop',
    mood: 'Hopeful',
    energy: 'medium',
    bpm: 108,
    key: 'D major',
    timeSignature: '4/4',
    chordDegrees: ['I', 'V', 'vi', 'IV'],
    structure: ['intro', 'verse-1', 'chorus', 'outro'],
    lyrics: {
      'verse-1': ['Rain moves slowly on the glass', 'I let the old night pass'],
      chorus: ['Turn the lights on', 'I can begin again'],
    },
    vocalCoach: { comfortableRange: 'D3-A4', practiceSteps: ['Clap, hum, sing, perform.'] },
  };
}

test('song blueprint becomes a deterministic playable karaoke arrangement', () => {
  const project = createBlankMusicProject();
  project.brief.idea = 'Beginning again after a rainy night';
  const arrangedProject = applySongBlueprint(project, blueprint());
  const first = buildSongArrangement(arrangedProject);
  const second = buildSongArrangement(arrangedProject);

  assert.equal(first.bpm, 108);
  assert.equal(first.key, 'D major');
  assert.ok(first.duration > 0);
  assert.ok(first.tracks.length >= 3);
  assert.ok(first.lyricCues.length >= 4);
  assert.ok(first.notes.length > 0);
  assert.deepEqual(first.notes, second.notes);
  assert.deepEqual(first.chords, second.chords);
  first.tracks.flatMap((track) => track.events).forEach((event) => {
    assert.ok(Number.isFinite(event.time) && event.time >= 0);
    assert.ok(Number.isFinite(event.duration) && event.duration >= 0.05);
    assert.ok(Number.isInteger(event.midi) && event.midi >= 0 && event.midi <= 127);
    assert.ok(event.velocity > 0 && event.velocity <= 1);
  });
});

test('editable section text round-trips into structured lyrics', () => {
  const text = '[verse 1]\nOne line\nSecond line\n\n[chorus]\nSing this';
  const lyrics = lyricsFromText(text);
  assert.deepEqual(lyrics['verse-1'], ['One line', 'Second line']);
  assert.deepEqual(lyrics.chorus, ['Sing this']);
  assert.match(lyricsAsText({ lyrics }), /\[chorus\]/);
});

test('one-tap styles change real arrangement decisions rather than only the label', () => {
  const blank = createBlankMusicProject();
  const acoustic = applySongStyle(blank, 'catchy-acoustic');
  const ballad = applySongStyle(blank, 'intimate-piano');
  const acousticArrangement = buildSongArrangement(acoustic);
  const balladArrangement = buildSongArrangement(ballad);

  assert.equal(CREATE_MUSIC_STYLES.length, 6);
  assert.equal(acoustic.brief.genre, 'Acoustic pop');
  assert.ok(acoustic.brief.instruments.includes('acoustic-guitar'));
  assert.equal(ballad.brief.bpm, 76);
  assert.equal(acousticArrangement.stylePreset, 'catchy-acoustic');
  assert.equal(balladArrangement.stylePreset, 'intimate-piano');
  assert.notDeepEqual(acousticArrangement.notes, balladArrangement.notes);
});

test('arranger v2 creates phrase space, musical piano motion, and chord-change pedalling', () => {
  const project = applySongBlueprint(createBlankMusicProject(), blueprint());
  const arrangement = buildSongArrangement(project);
  const piano = arrangement.tracks.find((track) => track.id === 'harmony-piano');
  const melody = arrangement.tracks.find((track) => track.id === 'guide-melody');

  assert.equal(arrangement.provenance.engine, 'polymath-musical-arranger-v002');
  assert.equal(arrangement.readyToPlayFormat, 'polymath-arrangement-v2');
  assert.ok(piano.events.length > arrangement.chords.length * 4);
  assert.ok(piano.events.some((event) => event.hand === 'left'));
  assert.ok(piano.events.some((event) => event.hand === 'right'));
  assert.equal(arrangement.pedals.length, arrangement.chords.length * 2);
  assert.equal(arrangement.provenance.pedalPolicy, 'repedal-each-harmony-change');

  arrangement.lyricCues.forEach((cue) => {
    const phraseEvents = melody.events.filter((event) => event.time >= cue.start && event.time < cue.end);
    if (!phraseEvents.length) return;
    const finalEnd = Math.max(...phraseEvents.map((event) => event.time + event.duration));
    assert.ok(finalEnd <= cue.end + 0.001);
    assert.ok(cue.end - finalEnd >= 0.035);
  });

  const byPitch = new Map();
  arrangement.notes.forEach((event) => {
    const previous = byPitch.get(event.midi);
    if (previous) assert.ok(event.time - previous.time >= 0.0649);
    byPitch.set(event.midi, event);
  });
});

test('selecting a guide singer changes the melodic timbre carried by the arrangement', () => {
  const project = createBlankMusicProject();
  project.brief.singerVoice = 'rich-baritone';
  const arrangement = buildSongArrangement(project);
  const melody = arrangement.tracks.find((track) => track.id === 'guide-melody');

  assert.equal(CREATE_MUSIC_VOICES.length, 4);
  assert.equal(arrangement.singerVoice, 'rich-baritone');
  assert.equal(melody.instrument, 'vocal-rich');
  assert.match(melody.label, /baritone/i);
});

test('every singer profile declares a usable and distinct rehearsal range', () => {
  assert.deepEqual(
    CREATE_MUSIC_VOICES.map((voice) => voice.defaultRange),
    ['C4-G5', 'G3-D5', 'C3-G4', 'G2-D4'],
  );
});
