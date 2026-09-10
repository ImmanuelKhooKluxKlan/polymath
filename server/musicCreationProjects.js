'use strict';

const crypto = require('crypto');

function clean(value, maximum = 500) {
  return String(value || '').trim().slice(0, maximum);
}

function clampNumber(value, minimum, maximum, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

function cleanLines(value, maxItems = 16, maxChars = 120) {
  return Array.isArray(value) ? value.map((item) => clean(item, maxChars)).filter(Boolean).slice(0, maxItems) : [];
}

function sanitizeLyrics(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value).slice(0, 20).flatMap(([key, lines]) => {
    const section = clean(key, 40).toLowerCase().replace(/[^a-z0-9-]+/g, '-');
    const safeLines = cleanLines(lines);
    return section && safeLines.length ? [[section, safeLines]] : [];
  }));
}

function sanitizeEvent(event) {
  if (!event || typeof event !== 'object') return null;
  const time = clampNumber(event.time, 0, 7200, 0);
  const duration = clampNumber(event.audioDuration ?? event.duration, 0.03, 120, 0.35);
  const midi = Math.round(clampNumber(event.midi, 0, 127, 60));
  return {
    note: clean(event.note, 8),
    midi,
    time,
    duration,
    audioDuration: duration,
    velocity: clampNumber(event.velocity, 0.01, 1, 0.75),
    hand: clean(event.hand, 10),
    scoreRole: clean(event.scoreRole, 30),
    lyric: clean(event.lyric, 40),
    word: clean(event.word, 80),
    percussion: clean(event.percussion, 30),
    articulation: clean(event.articulation, 30),
    ...(Number.isInteger(event.stringIndex) ? { stringIndex: Math.max(0, Math.min(5, event.stringIndex)) } : {}),
    ...(Number.isFinite(Number(event.fret)) ? { fret: clampNumber(event.fret, 0, 24, 0) } : {}),
  };
}

function sanitizeArrangement(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const tracks = Array.isArray(value.tracks) ? value.tracks.slice(0, 12).map((track, index) => ({
    id: clean(track.id || `track-${index + 1}`, 60),
    label: clean(track.label || `Track ${index + 1}`, 80),
    instrument: clean(track.instrument || 'piano', 50),
    role: clean(track.role, 40),
    colour: clean(track.colour, 20),
    events: (Array.isArray(track.events) ? track.events : []).slice(0, 25000).map(sanitizeEvent).filter(Boolean),
  })) : [];
  const totalTrackEvents = tracks.reduce((total, track) => total + track.events.length, 0);
  if (totalTrackEvents > 60000) throw Object.assign(new Error('Arrangement contains too many note events.'), { status: 413 });
  return {
    title: clean(value.title, 120),
    composer: clean(value.composer, 120),
    bpm: Math.round(clampNumber(value.bpm, 40, 220, 100)),
    key: clean(value.key, 20),
    timeSignature: ['4/4', '3/4', '6/8'].includes(value.timeSignature) ? value.timeSignature : '4/4',
    duration: clampNumber(value.duration, 0, 7200, 0),
    stylePreset: clean(value.stylePreset, 50),
    styleLabel: clean(value.styleLabel, 80),
    singerVoice: clean(value.singerVoice, 50),
    singerVoiceLabel: clean(value.singerVoiceLabel, 80),
    notes: (Array.isArray(value.notes) ? value.notes : []).slice(0, 30000).map(sanitizeEvent).filter(Boolean),
    pedals: (Array.isArray(value.pedals) ? value.pedals : []).slice(0, 5000).map((pedal) => ({
      time: clampNumber(pedal.time, 0, 7200, 0), down: Boolean(pedal.down),
    })),
    tracks,
    chords: (Array.isArray(value.chords) ? value.chords : []).slice(0, 3000).map((chord) => ({
      time: clampNumber(chord.time, 0, 7200, 0),
      duration: clampNumber(chord.duration, 0.03, 120, 1),
      degree: clean(chord.degree, 12),
      label: clean(chord.label, 40),
    })),
    sections: (Array.isArray(value.sections) ? value.sections : []).slice(0, 30).map((section, index) => ({
      id: clean(section.id || `section-${index}`, 60),
      type: clean(section.type, 40),
      label: clean(section.label, 60),
      start: clampNumber(section.start, 0, 7200, 0),
      end: clampNumber(section.end, 0, 7200, 0),
      bars: Math.round(clampNumber(section.bars, 0, 256, 0)),
    })),
    lyricCues: (Array.isArray(value.lyricCues) ? value.lyricCues : []).slice(0, 2000).map((cue, index) => ({
      id: clean(cue.id || `cue-${index}`, 60),
      section: clean(cue.section, 40),
      text: clean(cue.text, 120),
      start: clampNumber(cue.start, 0, 7200, 0),
      end: clampNumber(cue.end, 0, 7200, 0),
    })),
    provenance: {
      engine: clean(value.provenance?.engine, 100),
      generatedAt: clean(value.provenance?.generatedAt, 40),
      userLed: value.provenance?.userLed !== false,
      referencePolicy: clean(value.provenance?.referencePolicy, 120),
      pedalPolicy: clean(value.provenance?.pedalPolicy, 100),
    },
  };
}

function sanitizeProject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw Object.assign(new Error('Project data is required.'), { status: 400 });
  }
  const brief = value.brief && typeof value.brief === 'object' ? value.brief : {};
  const coach = value.vocalCoach && typeof value.vocalCoach === 'object' ? value.vocalCoach : {};
  return {
    title: clean(value.title || 'Untitled song', 120),
    status: ['draft', 'arranged', 'rehearsing', 'complete'].includes(value.status) ? value.status : 'draft',
    summary: clean(value.summary, 400),
    brief: {
      idea: clean(brief.idea, 1800),
      genre: clean(brief.genre, 80),
      mood: clean(brief.mood, 80),
      energy: ['low', 'medium', 'high'].includes(brief.energy) ? brief.energy : 'medium',
      bpm: Math.round(clampNumber(brief.bpm, 40, 220, 100)),
      key: clean(brief.key, 20),
      timeSignature: ['3/4', '4/4', '6/8'].includes(brief.timeSignature) ? brief.timeSignature : '4/4',
      vocalRange: clean(brief.vocalRange, 30),
      reference: clean(brief.reference, 300),
      referenceNotes: clean(brief.referenceNotes, 800),
      stylePreset: clean(brief.stylePreset, 50),
      singerVoice: clean(brief.singerVoice, 50),
      instruments: cleanLines(brief.instruments, 8, 50),
      structure: cleanLines(brief.structure, 16, 40),
    },
    lyrics: sanitizeLyrics(value.lyrics),
    vocalCoach: {
      comfortableRange: clean(coach.comfortableRange, 30),
      delivery: cleanLines(coach.delivery, 10, 180),
      breathing: cleanLines(coach.breathing, 10, 180),
      practiceSteps: cleanLines(coach.practiceSteps, 12, 180),
    },
    referenceTraits: cleanLines(value.referenceTraits, 10, 140),
    chordDegrees: cleanLines(value.chordDegrees, 16, 12),
    revisionRequest: clean(value.revisionRequest, 1200),
  };
}

function publicProject(row, { includeArrangement = true } = {}) {
  return {
    id: row.id,
    ...row.project,
    ...(includeArrangement ? { arrangement: row.arrangement || null } : {}),
    revision: Number(row.revision || 1),
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
  };
}

function createProject(db, userId, project, arrangement) {
  const now = new Date().toISOString();
  const row = {
    id: `music-project-${crypto.randomUUID()}`,
    userId,
    project: sanitizeProject(project),
    arrangement: sanitizeArrangement(arrangement),
    revision: 1,
    createdAt: now,
    updatedAt: now,
  };
  db.musicCreationProjects.push(row);
  return publicProject(row);
}

function findProject(db, userId, projectId) {
  return db.musicCreationProjects.find((item) => item.id === projectId && item.userId === userId) || null;
}

function listProjects(db, userId) {
  return db.musicCreationProjects
    .filter((item) => item.userId === userId)
    .sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)))
    .slice(0, 100)
    .map((item) => ({
      id: item.id,
      title: item.project?.title || 'Untitled song',
      status: item.project?.status || 'draft',
      genre: item.project?.brief?.genre || '',
      bpm: Number(item.project?.brief?.bpm || 0),
      key: item.project?.brief?.key || '',
      revision: Number(item.revision || 1),
      createdAt: item.createdAt,
      updatedAt: item.updatedAt,
    }));
}

function updateProject(db, userId, projectId, project, arrangement, expectedRevision) {
  const row = findProject(db, userId, projectId);
  if (!row) throw Object.assign(new Error('Music project not found.'), { status: 404 });
  if (expectedRevision && Number(expectedRevision) !== Number(row.revision || 1)) {
    throw Object.assign(new Error('This project changed in another session. Reload it before saving.'), { status: 409 });
  }
  row.project = sanitizeProject(project);
  row.arrangement = sanitizeArrangement(arrangement);
  row.revision = Number(row.revision || 1) + 1;
  row.updatedAt = new Date().toISOString();
  return publicProject(row);
}

function deleteProject(db, userId, projectId) {
  const index = db.musicCreationProjects.findIndex((item) => item.id === projectId && item.userId === userId);
  if (index < 0) throw Object.assign(new Error('Music project not found.'), { status: 404 });
  const [removed] = db.musicCreationProjects.splice(index, 1);
  return publicProject(removed, { includeArrangement: false });
}

module.exports = {
  createProject,
  deleteProject,
  findProject,
  listProjects,
  publicProject,
  sanitizeArrangement,
  sanitizeProject,
  updateProject,
};
