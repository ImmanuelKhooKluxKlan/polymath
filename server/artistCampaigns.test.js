'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { Midi } = require('@tonejs/midi');
const {
  buildCampaignExcerpt,
  campaignAttribution,
  campaignIsLive,
  campaignPublishProblems,
  normalizeCampaignInput,
  publicArtistCampaign,
} = require('./artistCampaigns');

function validCampaign(overrides = {}) {
  return {
    id: 'artist_campaign_12345678',
    ...normalizeCampaignInput({
      artist: 'Independent Artist',
      title: 'Midnight Practice',
      hook: 'Can you play the chorus?',
      status: 'published',
      referralCode: 'MIDNIGHT20',
      previewStartSeconds: 12,
      previewDurationSeconds: 20,
      qaScore: 91,
      rightsConfirmed: true,
      rightsHolder: 'Independent Artist',
      rightsBasis: 'Written campaign licence dated 2026-09-07',
      artistApproved: true,
      humanVerified: true,
      verificationNotes: 'Pitch, timing, holds, dynamics, and pedal checked by a pianist.',
    }),
    songAssetKey: 'artist-campaigns/challenge.json',
    songFilename: 'challenge.json',
    songFormat: 'JSON',
    publicSongAssetKey: 'artist-campaigns/public.json',
    excerptNoteCount: 2,
    createdAt: '2026-09-07T00:00:00.000Z',
    updatedAt: '2026-09-07T00:00:00.000Z',
    ...overrides,
  };
}

test('campaign publication gate requires rights, approval, human QA, and playable data', () => {
  const invalid = validCampaign({
    songAssetKey: '', rightsConfirmed: false, artistApproved: false,
    humanVerified: false, qaScore: 79,
  });
  const problems = campaignPublishProblems(invalid);
  assert.equal(problems.some((problem) => problem.includes('playable')), true);
  assert.equal(problems.some((problem) => problem.includes('rights')), true);
  assert.equal(problems.some((problem) => problem.includes('artist')), true);
  assert.equal(problems.some((problem) => problem.includes('80/100')), true);
  assert.equal(campaignIsLive(invalid), false);
  assert.equal(campaignPublishProblems(validCampaign()).length, 0);
});

test('the public JSON excerpt excludes every note and private field outside the approved window', () => {
  const source = Buffer.from(JSON.stringify({
    title: 'Private source title',
    unreleasedLyrics: 'must never leave the server',
    bpm: 96,
    notes: [
      { note: 'A3', time: 1, duration: 1, privateLabel: 'outside-before' },
      { note: 'C4', time: 11.5, duration: 1, visualDuration: 0.8, audioDuration: 0.65, hand: 'left' },
      { note: 'E4', time: 13, duration: 0.5, velocity: 0.9, hand: 'right' },
      { note: 'G4', time: 33, duration: 1, privateLabel: 'outside-after' },
    ],
    pedals: [{ time: 11, down: true }, { time: 14, down: false }, { time: 35, down: true }],
  }));
  const excerpt = buildCampaignExcerpt(source, 'JSON', validCampaign());
  const publicSong = JSON.parse(excerpt.buffer.toString('utf8'));

  assert.equal(excerpt.noteCount, 2);
  assert.deepEqual(publicSong.notes.map((note) => [note.note, note.time, note.duration]), [
    ['C4', 0, 0.5],
    ['E4', 1, 0.5],
  ]);
  assert.equal(publicSong.notes[0].visualDuration, 0.3);
  assert.equal(publicSong.notes[0].audioDuration, 0.15);
  assert.deepEqual(publicSong.pedals.map((pedal) => [pedal.time, pedal.down]), [[0, true], [2, false]]);
  assert.equal('unreleasedLyrics' in publicSong, false);
  assert.equal(excerpt.buffer.includes(Buffer.from('outside-before')), false);
  assert.equal(excerpt.buffer.includes(Buffer.from('outside-after')), false);
});

test('a standard MIDI upload becomes a bounded public JSON challenge', () => {
  const midi = new Midi();
  midi.header.setTempo(120);
  const track = midi.addTrack();
  track.addNote({ midi: 60, time: 11, duration: 0.5, velocity: 0.7 });
  track.addNote({ midi: 64, time: 13, duration: 0.75, velocity: 0.8 });
  const excerpt = buildCampaignExcerpt(Buffer.from(midi.toArray()), 'MIDI', validCampaign());
  const publicSong = JSON.parse(excerpt.buffer.toString('utf8'));

  assert.equal(excerpt.noteCount, 1);
  assert.equal(publicSong.notes[0].note, 'E4');
  assert.equal(publicSong.notes[0].time, 1);
  assert.equal(publicSong.readyToPlayFormat, 'polymath-artist-campaign-v1');
});

test('only scheduled, gate-passing published campaigns are public', () => {
  const now = Date.parse('2026-09-07T12:00:00.000Z');
  assert.equal(campaignIsLive(validCampaign(), now), true);
  assert.equal(campaignIsLive(validCampaign({ launchesAt: '2026-09-08T00:00:00.000Z' }), now), false);
  assert.equal(campaignIsLive(validCampaign({ endsAt: '2026-09-07T11:00:00.000Z' }), now), false);
  const publicRecord = publicArtistCampaign(validCampaign(), { now });
  assert.equal(publicRecord.verification.humanVerified, true);
  assert.equal(publicRecord.verification.qaScore, 91);
  assert.equal('rightsBasis' in publicRecord, false);
  assert.equal('songAssetKey' in publicRecord, false);
});

test('checkout attribution accepts only the live campaign and its canonical referral', () => {
  const campaign = validCampaign();
  const db = { artistCampaigns: [campaign] };
  assert.deepEqual(campaignAttribution(db, {
    campaignId: campaign.id,
    campaignSlug: campaign.slug,
    referralCode: 'MIDNIGHT20',
  }), {
    campaignId: campaign.id,
    campaignSlug: campaign.slug,
    referralCode: 'MIDNIGHT20',
  });
  assert.equal(campaignAttribution(db, {
    campaignId: campaign.id,
    referralCode: 'SPOOFED',
  }).referralCode, '');
  assert.deepEqual(campaignAttribution({ artistCampaigns: [{ ...campaign, status: 'paused' }] }, {
    campaignId: campaign.id,
  }), {});
});
