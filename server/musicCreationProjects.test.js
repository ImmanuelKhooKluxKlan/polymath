'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  createProject,
  deleteProject,
  findProject,
  listProjects,
  updateProject,
} = require('./musicCreationProjects');

function project(title = 'First light') {
  return {
    title,
    status: 'draft',
    brief: {
      idea: 'Finding courage on a quiet train ride.',
      genre: 'Pop',
      mood: 'Hopeful',
      bpm: 104,
      key: 'C major',
      timeSignature: '4/4',
      stylePreset: 'catchy-acoustic',
      singerVoice: 'warm-alto',
      instruments: ['piano', 'drums'],
      structure: ['verse-1', 'chorus'],
    },
    lyrics: { 'verse-1': ['Rain against the window'], chorus: ['I can begin again'] },
  };
}

function arrangement() {
  return {
    title: 'First light',
    bpm: 104,
    key: 'C major',
    timeSignature: '4/4',
    duration: 12,
    stylePreset: 'catchy-acoustic',
    styleLabel: 'Catchy acoustic',
    singerVoice: 'warm-alto',
    singerVoiceLabel: 'Warm alto',
    notes: [{ note: 'C4', midi: 60, time: 0, duration: 1, velocity: 0.75, articulation: 'legato' }],
    tracks: [{ id: 'piano', label: 'Piano', instrument: 'piano', events: [] }],
    provenance: { engine: 'polymath-musical-arranger-v002', pedalPolicy: 'repedal-each-harmony-change' },
  };
}

test('music projects are private, revision-safe, and removable', () => {
  const db = { musicCreationProjects: [] };
  const created = createProject(db, 'user-a', project(), arrangement());
  assert.match(created.id, /^music-project-/);
  assert.equal(created.revision, 1);
  assert.equal(findProject(db, 'user-b', created.id), null);
  assert.equal(listProjects(db, 'user-a').length, 1);
  assert.equal(listProjects(db, 'user-a')[0].arrangement, undefined);
  assert.equal(created.brief.stylePreset, 'catchy-acoustic');
  assert.equal(created.brief.singerVoice, 'warm-alto');
  assert.equal(created.arrangement.styleLabel, 'Catchy acoustic');
  assert.equal(created.arrangement.singerVoiceLabel, 'Warm alto');
  assert.equal(created.arrangement.notes[0].articulation, 'legato');
  assert.equal(created.arrangement.provenance.pedalPolicy, 'repedal-each-harmony-change');

  const updated = updateProject(db, 'user-a', created.id, project('New title'), arrangement(), 1);
  assert.equal(updated.title, 'New title');
  assert.equal(updated.revision, 2);
  assert.throws(
    () => updateProject(db, 'user-a', created.id, project(), arrangement(), 1),
    (error) => error.status === 409,
  );

  const removed = deleteProject(db, 'user-a', created.id);
  assert.equal(removed.id, created.id);
  assert.equal(db.musicCreationProjects.length, 0);
});

test('arrangement validation clamps unsafe values and normalizes time signatures', () => {
  const db = { musicCreationProjects: [] };
  const created = createProject(db, 'user-a', project(), {
    ...arrangement(),
    bpm: 900,
    timeSignature: 'not-a-meter',
    notes: [{ note: 'C99', midi: 999, time: -2, duration: -4, velocity: 8 }],
  });
  assert.equal(created.arrangement.bpm, 220);
  assert.equal(created.arrangement.timeSignature, '4/4');
  assert.deepEqual(
    Object.fromEntries(['midi', 'time', 'duration', 'velocity'].map((key) => [key, created.arrangement.notes[0][key]])),
    { midi: 127, time: 0, duration: 0.03, velocity: 1 },
  );
});
