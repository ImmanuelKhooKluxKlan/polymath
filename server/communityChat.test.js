'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {
  GLOBAL_ROOM_ID,
  canReadRoom,
  canWriteRoom,
  cleanCommunityText,
  ensureGlobalRoom,
  publicRoom,
  trimRoomMessages,
} = require('./communityChat');

function fixture() {
  return {
    users: [{ id: 'u1', name: 'One' }, { id: 'u2', name: 'Two' }],
    communityRooms: [],
    communityMemberships: [],
    communityMessages: [],
  };
}

test('seeds exactly one global Free Flow room', () => {
  const db = fixture();
  ensureGlobalRoom(db);
  ensureGlobalRoom(db);
  assert.equal(db.communityRooms.length, 1);
  assert.equal(db.communityRooms[0].id, GLOBAL_ROOM_ID);
});

test('private groups are readable and writable only by members', () => {
  const db = fixture();
  const room = { id: 'private', ownerId: 'u1', visibility: 'private' };
  db.communityMemberships.push({ id: 'm1', roomId: room.id, userId: 'u1', role: 'owner' });
  assert.equal(canReadRoom(db, room, 'u1'), true);
  assert.equal(canWriteRoom(db, room, 'u1'), true);
  assert.equal(canReadRoom(db, room, 'u2'), false);
  assert.equal(canWriteRoom(db, room, 'u2'), false);
});

test('room output hides invite codes from non-owners', () => {
  const db = fixture();
  const room = { id: 'r1', name: 'Piano', ownerId: 'u1', visibility: 'private', inviteCode: 'SECRET', createdAt: new Date().toISOString() };
  db.communityRooms.push(room);
  db.communityMemberships.push({ id: 'm1', roomId: room.id, userId: 'u2', role: 'member' });
  assert.equal(publicRoom(room, db, 'u2').inviteCode, undefined);
  assert.equal(publicRoom(room, db, 'u1').inviteCode, 'SECRET');
});

test('bounds messages and trims old room history', () => {
  const db = fixture();
  assert.equal(cleanCommunityText(`  ${'x'.repeat(1400)}  `).length, 1200);
  db.communityMessages = Array.from({ length: 5 }, (_, index) => ({
    id: `m${index}`,
    roomId: 'r1',
    createdAt: new Date(index * 1000).toISOString(),
  }));
  assert.equal(trimRoomMessages(db, 'r1', 3), 2);
  assert.deepEqual(db.communityMessages.map((message) => message.id), ['m2', 'm3', 'm4']);
});
