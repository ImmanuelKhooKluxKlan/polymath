import assert from 'node:assert/strict';
import test from 'node:test';
import {
  campaignPlaybackRange,
  campaignShareUrl,
  prepareCampaignSong,
} from '../../src/engine/artistCampaign.js';

const campaign = {
  id: 'artist_campaign_12345678',
  slug: 'independent-artist-midnight',
  artist: 'Independent Artist',
  title: 'Midnight Practice',
  referralCode: 'MIDNIGHT20',
  preview: { startSeconds: 10, durationSeconds: 20 },
};

test('campaign preparation clips and shifts only the approved preview', () => {
  const prepared = prepareCampaignSong({
    title: 'Source',
    performance: { preserveScoreDurations: true },
    notes: [
      {
        note: 'C4', time: 9.5, duration: 1, visualDuration: 0.75,
        audioDuration: 0.6, velocity: 0.7, hand: 'left',
      },
      { note: 'E4', time: 12, duration: 0.5, velocity: 0.8, hand: 'right' },
      { note: 'G4', time: 31, duration: 0.5, velocity: 0.9, hand: 'right' },
    ],
    pedals: [
      { time: 9, down: true },
      { time: 14, down: false },
      { time: 32, down: true },
    ],
  }, campaign);

  assert.equal(prepared.libraryId, `campaign:${campaign.id}`);
  assert.equal(prepared.title, campaign.title);
  assert.equal(prepared.composer, campaign.artist);
  assert.equal(prepared.notes.length, 2);
  assert.equal(prepared.notes[0].time, 0);
  assert.equal(prepared.notes[0].duration, 0.5);
  assert.equal(prepared.notes[0].visualDuration, 0.25);
  assert.equal(prepared.notes[0].audioDuration, 0.1);
  assert.deepEqual(prepared.pedals.map((pedal) => [pedal.time, pedal.down]), [[0, true], [4, false]]);
  assert.equal(campaignPlaybackRange(prepared).duration, 20);
});

test('campaign share links are social-preview paths in production and hash routes locally', () => {
  const publicUrl = new URL(campaignShareUrl('https://polymathmusician67.com/#studio', campaign, 88));
  assert.equal(publicUrl.pathname, `/c/${campaign.slug}`);
  assert.equal(publicUrl.searchParams.get('score'), '88');
  assert.equal(publicUrl.searchParams.get('ref'), 'MIDNIGHT20');
  assert.equal(publicUrl.hash, '');

  const localUrl = new URL(campaignShareUrl('http://localhost:5173/#studio', campaign, 105));
  assert.equal(localUrl.pathname, '/');
  assert.match(localUrl.hash, /campaign=independent-artist-midnight/);
  assert.match(localUrl.hash, /score=100/);
});

test('an empty preview fails before a campaign can reach the piano', () => {
  assert.throws(
    () => prepareCampaignSong({ notes: [{ note: 'C4', time: 80, duration: 1 }] }, campaign),
    /does not contain playable notes/,
  );
});
