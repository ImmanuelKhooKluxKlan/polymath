'use strict';

const GLOBAL_ROOM_ID = 'community_free_flow';
const GLOBAL_ROOM = Object.freeze({
  id: GLOBAL_ROOM_ID,
  name: 'Polymath Free Flow',
  topic: 'Meet musicians, share ideas, and talk about what matters to you.',
  visibility: 'global',
  ownerId: 'platform',
  createdAt: '2026-01-01T00:00:00.000Z',
});

function ensureGlobalRoom(db) {
  let room = db.communityRooms.find((candidate) => candidate.id === GLOBAL_ROOM_ID);
  if (!room) {
    room = { ...GLOBAL_ROOM };
    db.communityRooms.unshift(room);
  }
  return room;
}

function membershipFor(db, roomId, userId) {
  return db.communityMemberships.find(
    (membership) => membership.roomId === roomId && membership.userId === userId,
  ) || null;
}

function canReadRoom(db, room, userId) {
  return room?.id === GLOBAL_ROOM_ID
    || room?.visibility === 'public'
    || Boolean(membershipFor(db, room?.id, userId));
}

function canWriteRoom(db, room, userId) {
  return room?.id === GLOBAL_ROOM_ID || Boolean(membershipFor(db, room?.id, userId));
}

function cleanCommunityText(value, maximum = 1200) {
  return String(value || '').replace(/\u0000/g, '').trim().slice(0, maximum);
}

function publicRoom(room, db, viewerId, isAdmin = false) {
  const membership = membershipFor(db, room.id, viewerId);
  const isOwner = room.ownerId === viewerId;
  const memberCount = room.id === GLOBAL_ROOM_ID
    ? null
    : db.communityMemberships.filter((item) => item.roomId === room.id).length;
  const lastMessage = db.communityMessages
    .filter((message) => message.roomId === room.id)
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0];
  return {
    id: room.id,
    name: room.name,
    topic: room.topic || '',
    visibility: room.visibility,
    memberCount,
    joined: room.id === GLOBAL_ROOM_ID || Boolean(membership),
    role: isOwner ? 'owner' : membership?.role || (room.id === GLOBAL_ROOM_ID ? 'member' : null),
    inviteCode: isOwner || isAdmin ? room.inviteCode || null : undefined,
    lastMessageAt: lastMessage?.createdAt || room.createdAt,
    createdAt: room.createdAt,
  };
}

function publicMessage(message, db, viewerId, room, isAdmin = false) {
  const author = db.users.find((user) => user.id === message.userId);
  const viewerMembership = membershipFor(db, room.id, viewerId);
  return {
    id: message.id,
    roomId: message.roomId,
    text: message.text,
    createdAt: message.createdAt,
    editedAt: message.editedAt || null,
    author: {
      userId: message.userId,
      name: author?.name || 'Former member',
      avatarUrl: author?.avatarUrl || '',
    },
    mine: message.userId === viewerId,
    canDelete: isAdmin
      || message.userId === viewerId
      || room.ownerId === viewerId
      || viewerMembership?.role === 'moderator',
  };
}

function trimRoomMessages(db, roomId, maximum) {
  const roomMessages = db.communityMessages
    .filter((message) => message.roomId === roomId)
    .sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
  if (roomMessages.length <= maximum) return 0;
  const removeIds = new Set(roomMessages.slice(0, roomMessages.length - maximum).map((message) => message.id));
  db.communityMessages = db.communityMessages.filter((message) => !removeIds.has(message.id));
  return removeIds.size;
}

module.exports = {
  GLOBAL_ROOM_ID,
  canReadRoom,
  canWriteRoom,
  cleanCommunityText,
  ensureGlobalRoom,
  membershipFor,
  publicMessage,
  publicRoom,
  trimRoomMessages,
};
