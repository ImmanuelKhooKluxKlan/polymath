import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applySongBlueprint,
  buildSongArrangement,
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
