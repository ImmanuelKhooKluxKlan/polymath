const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { postProcessMuscriptorResult } = require('./muscriptorPostprocess');
const { createMuscriptorEventCollector } = require('./muscriptorEvents');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-admin-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.ADMIN_EMAILS = 'admin@example.test';
process.env.PAYPAL_ENV = 'sandbox';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.MUSCRIPTOR_REMOTE_URL = '';
process.env.RUNPOD_SERVERLESS_ENDPOINT_ID = '';
process.env.RUNPOD_API_KEY = '';
process.env.RUNPOD_NETWORK_VOLUME_ID = '';
process.env.RUNPOD_S3_REGION = '';
process.env.RUNPOD_S3_ENDPOINT = '';
process.env.RUNPOD_S3_ACCESS_KEY_ID = '';
process.env.RUNPOD_S3_SECRET_ACCESS_KEY = '';
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';

const { app } = require('./server');

test('MuScriptor piano cleanup removes duplicate strikes and impossible overlaps', () => {
  const sourceEnvelope = {
    frameSeconds: 0.1,
    levels: Array.from({ length: 50 }, (_, index) => (
      index < 8 ? 0.01 : index < 18 ? 0.08 : index < 28 ? 0.2 : 0.5
    )),
  };
  const result = postProcessMuscriptorResult({
    title: 'Cleanup fixture',
    notes: [
      { midi: 67, note: 'G4', time: 0, duration: 0.3, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 60, note: 'C4', time: 1, duration: 0.7, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 60, note: 'C4', time: 1.02, duration: 0.5, velocity: 0.78, instrument: 'electric_piano' },
      { midi: 60, note: 'C4', time: 1.06, duration: 0.35, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 60, note: 'C4', time: 1.5, duration: 0.8, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 64, note: 'E4', time: 3, duration: 30, velocity: 0.78, instrument: 'electric_piano' },
    ],
  }, {
    instrument: 'piano',
    sourceEnvelope,
  });

  const c4Notes = result.notes.filter((note) => note.midi === 60);
  assert.equal(result.notes.length, 4);
  assert.equal(c4Notes.length, 2);
  assert.equal(c4Notes[0].instrument, 'acoustic_piano');
  assert.ok(c4Notes[0].time + c4Notes[0].duration < c4Notes[1].time);
  assert.equal(result.notes.find((note) => note.midi === 64).duration, 8);
  assert.ok(result.notes.find((note) => note.midi === 64).velocity
    > result.notes.find((note) => note.midi === 67).velocity);
  assert.deepEqual(result.transcriptionCleanup, {
    version: 2,
    inputNotes: 6,
    outputNotes: 4,
    removedDuplicateNotes: 2,
    removedRapidRetriggers: 2,
    excludedVocalNotes: 0,
    vocalMelodyNotes: 0,
    vocalMelodyGain: 1.18,
    shortenedSameKeyOverlaps: 1,
    cappedImpossibleDurations: 1,
    sourceDynamicsApplied: true,
    duplicateOnsetWindowMs: 75,
    maximumPianoHoldSeconds: 8,
  });
});

test('Full song renders the vocal melody on piano while instrumental mode excludes it', () => {
  const payload = {
    title: 'Vocal fixture',
    notes: [
      { midi: 60, note: 'C4', time: 1, duration: 0.4, velocity: 0.78, instrument: 'voice' },
      { midi: 60, note: 'C4', time: 1.02, duration: 0.3, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 64, note: 'E4', time: 2, duration: 0.4, velocity: 0.78, instrument: 'acoustic_guitar' },
    ],
  };

  const full = postProcessMuscriptorResult(payload, {
    instrument: 'piano',
    playbackMode: 'full',
  });
  const instrumental = postProcessMuscriptorResult(payload, {
    instrument: 'piano',
    playbackMode: 'instrumental',
  });

  assert.equal(full.notes.length, 2);
  assert.equal(full.notes[0].instrument, 'voice');
  assert.equal(full.transcriptionCleanup.vocalMelodyNotes, 1);
  assert.equal(full.transcriptionCleanup.excludedVocalNotes, 0);
  assert.ok(full.notes[0].velocity > instrumental.notes[0].velocity);
  assert.equal(instrumental.notes.length, 2);
  assert.equal(instrumental.notes.some((note) => note.instrument === 'voice'), false);
  assert.equal(instrumental.transcriptionCleanup.excludedVocalNotes, 1);
  assert.equal(instrumental.transcriptionCleanup.vocalMelodyNotes, 0);
});

test('MuScriptor remote events apply detected tempo and onset alignment', () => {
  const progressEvents = [];
  const collector = createMuscriptorEventCollector({
    model: 'large',
    source: 'runpod',
    onProgress: (progress) => progressEvents.push(progress),
  });
  collector.accept({ type: 'progress', completed: 1, total: 4 });
  collector.accept({ type: 'start', index: 7, pitch: 60, start_time: 0.18, instrument: 'acoustic_piano' });
  collector.accept({ type: 'end', start_event_index: 7, end_time: 0.68 });
  collector.accept({
    type: 'transcription_complete',
    beat_grid: { bpm: 92.4, beats_per_bar: 3, first_downbeat: 0.1, onset_delay: 0.08 },
  });

  const result = collector.finish();
  assert.deepEqual(progressEvents, [{ completed: 1, total: 4 }]);
  assert.equal(result.notes[0].time, 0.1);
  assert.equal(result.notes[0].duration, 0.5);
  assert.equal(result.beatGrid.bpm, 92.4);
  assert.equal(result.beatGrid.beatsPerBar, 3);
  assert.equal(result.diagnostics.onsetDelayAppliedSeconds, 0.08);
});

test('MuScriptor remote event collection rejects malformed notes without losing valid notes', () => {
  const collector = createMuscriptorEventCollector({ model: 'large', source: 'runpod' });
  collector.accept({ type: 'end', start_event_index: 99, end_time: 1 });
  collector.accept({ type: 'start', index: 1, pitch: 'bad', start_time: 0, instrument: 'acoustic_piano' });
  collector.accept({ type: 'start', index: 2, pitch: 64, start_time: 1, instrument: 'acoustic_piano' });
  const result = collector.finish();

  assert.equal(result.notes.length, 1);
  assert.equal(result.notes[0].note, 'E4');
  assert.equal(result.notes[0].duration, 0.4);
  assert.equal(result.diagnostics.unmatchedEndEvents, 1);
  assert.equal(result.diagnostics.malformedEvents, 1);
  assert.equal(result.diagnostics.danglingStartEvents, 1);
});

test('MuScriptor piano cleanup preserves non-overlapping rapid repeated notes', () => {
  const result = postProcessMuscriptorResult({
    title: 'Rapid repeat fixture',
    notes: [
      { midi: 60, note: 'C4', time: 1, duration: 0.035, velocity: 0.78, instrument: 'acoustic_piano' },
      { midi: 60, note: 'C4', time: 1.06, duration: 0.035, velocity: 0.78, instrument: 'acoustic_piano' },
    ],
  }, { instrument: 'piano' });

  assert.equal(result.notes.length, 2);
  assert.equal(result.transcriptionCleanup.removedRapidRetriggers, 0);
});

test('admin policies, vouchers, password reset, and hashed sessions persist', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(testDataDir, { recursive: true, force: true });
  });

  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  async function api(pathname, { method = 'GET', token = '', body } = {}) {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const data = await response.json();
    return { status: response.status, data };
  }

  async function register(_pathname, { body }) {
    const channel = body.email ? 'email' : 'phone';
    const challenge = await api('/api/auth/register/otp', {
      method: 'POST',
      body: {
        channel,
        email: channel === 'email' ? body.email : '',
        phone: channel === 'phone' ? body.phone : '',
      },
    });
    assert.equal(challenge.status, 202);
    return api('/api/auth/register', {
      method: 'POST',
      body: {
        ...body,
        challengeId: challenge.data.challengeId,
        verificationCode: '123456',
      },
    });
  }

  async function createListing(token, title) {
    return api('/api/listings', {
      method: 'POST',
      token,
      body: {
        artist: 'Test Artist',
        title,
        instrument: 'piano',
        format: 'JSON',
        priceMcoins: 100,
        description: 'Isolated integration-test listing.',
        filename: `${title.toLowerCase().replace(/\s+/g, '-')}.json`,
        contentBase64: Buffer.from(JSON.stringify({ title, notes: [] })).toString('base64'),
        rightsConfirmed: true,
        feeConfirmed: true,
      },
    });
  }

  const transcriptionCapability = await api('/api/media-transcriptions/capabilities');
  assert.equal(transcriptionCapability.status, 200);
  assert.equal(typeof transcriptionCapability.data.enabled, 'boolean');
  assert.equal(transcriptionCapability.data.model, 'large');
  assert.equal(transcriptionCapability.data.execution, 'local');
  assert.equal(transcriptionCapability.data.maxBytes, null);
  assert.equal(transcriptionCapability.data.license, 'CC-BY-NC-4.0');

  const catalog = await api('/api/catalog');
  assert.equal(catalog.status, 200);
  assert.equal(catalog.data.mcoinsPerUsd, 1);
  assert.deepEqual(catalog.data.translationMcoinCosts, { subscriber: 0.5, free: 2 });
  assert.equal(catalog.data.products.find((item) => item.id === 'polymath-chill-monthly').price, '7.99');
  assert.equal(catalog.data.products.find((item) => item.id === 'polymath-chill-yearly').price, '49.99');
  assert.equal(catalog.data.products.find((item) => item.id === 'polymath-musician-monthly').price, '14.99');
  assert.equal(catalog.data.products.find((item) => item.id === 'polymath-musician-yearly').price, '93.99');
  assert.equal(catalog.data.products.find((item) => item.id === 'mcoins-50').price, '50.00');

  const unauthenticatedTranscription = await api('/api/media-transcriptions', { method: 'POST' });
  assert.equal(unauthenticatedTranscription.status, 401);

  const unverifiedRegistration = await api('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Unverified User',
      email: 'unverified@example.test',
      password: 'UnverifiedPassword123',
    },
  });
  assert.equal(unverifiedRegistration.status, 400);
  assert.match(unverifiedRegistration.data.error, /verify a new code/i);

  const adminRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Test Admin',
      email: 'admin@example.test',
      phone: '+65 8000 0001',
      password: 'AdminPassword123',
    },
  });
  assert.equal(adminRegistration.status, 201);
  assert.equal(adminRegistration.data.user.admin, true);
  assert.equal(adminRegistration.data.user.translationAllowance.plan, 'admin');
  assert.equal(adminRegistration.data.user.translationAllowance.unlimited, true);
  assert.equal(adminRegistration.data.user.translationAllowance.limit, null);
  assert.equal(adminRegistration.data.user.translationAllowance.remaining, null);
  assert.match(adminRegistration.data.user.friend_id, /^user_[a-f0-9]{5}$/);
  const adminToken = adminRegistration.data.token;
  const adminFriendId = adminRegistration.data.user.friend_id;

  const userRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Test Customer',
      email: 'customer@example.test',
      password: 'CustomerPassword123',
    },
  });
  assert.equal(userRegistration.status, 201);
  assert.equal(userRegistration.data.user.translationAllowance.unlimited, false);
  assert.equal(userRegistration.data.user.translationAllowance.plan, 'free');
  assert.equal(userRegistration.data.user.translationAllowance.limit, 0);
  assert.equal(userRegistration.data.user.translationAllowance.remaining, 0);
  assert.equal(userRegistration.data.user.translationAllowance.overageCostMcoins, 2);
  assert.equal(userRegistration.data.user.access.learn, false);
  assert.equal(userRegistration.data.user.access.band, false);
  assert.match(userRegistration.data.user.friend_id, /^user_[a-f0-9]{5}$/);
  const userToken = userRegistration.data.token;
  const userId = userRegistration.data.user.user_id;

  const freeBandAccess = await api('/api/bands', { token: userToken });
  assert.equal(freeBandAccess.status, 403);
  assert.match(freeBandAccess.data.error, /Musician plan/);
  const adminBandAccess = await api('/api/bands', { token: adminToken });
  assert.equal(adminBandAccess.status, 200);

  const sellerRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Test Seller',
      phone: '+65 8000 0004',
      password: 'SellerPassword123',
    },
  });
  assert.equal(sellerRegistration.status, 201);
  const sellerToken = sellerRegistration.data.token;
  const sellerFriendId = sellerRegistration.data.user.friend_id;

  const policyUpdate = await api('/api/admin/policies', {
    method: 'PUT',
    token: adminToken,
    body: {
      registrationEnabled: true,
      minimumSignupAge: 18,
      minimumPasswordLength: 10,
      minimumMarketplacePriceMcoins: 30,
      minimumWithdrawalMcoins: 250,
      welcomeMcoins: 25,
      policyNotice: 'Adults only during this test.',
      supportEmail: 'support@example.test',
    },
  });
  assert.equal(policyUpdate.status, 200);
  assert.equal(policyUpdate.data.policies.minimumSignupAge, 18);

  const underageBlocked = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'No Birth Date',
      email: 'new@example.test',
      phone: '+65 8000 0003',
      password: 'LongPassword123',
    },
  });
  assert.equal(underageBlocked.status, 400);

  const policyCompliantRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Adult Customer',
      email: 'adult@example.test',
      phone: '+65 8000 0005',
      password: 'LongPassword123',
      birthDate: '1990-01-01',
      termsAccepted: true,
    },
  });
  assert.equal(policyCompliantRegistration.status, 201);
  assert.equal(policyCompliantRegistration.data.user.mcoins, 25);
  const adultToken = policyCompliantRegistration.data.token;

  const promotionCreate = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'TEST50',
      name: 'Test wallet voucher',
      kind: 'mcoin_credit',
      value: 50,
      maxRedemptions: 10,
      perUserLimit: 1,
    },
  });
  assert.equal(promotionCreate.status, 201);
  assert.equal(promotionCreate.data.promotion.code, 'TEST50');

  const luckyRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Lucky Adult',
      email: 'lucky-adult@example.test',
      password: 'LongPassword123',
      birthDate: '1990-01-01',
      termsAccepted: true,
      luckyCode: 'test50',
    },
  });
  assert.equal(luckyRegistration.status, 201);
  assert.equal(luckyRegistration.data.user.luckyCodeApplied, true);
  assert.equal(luckyRegistration.data.user.mcoins, 75);

  const redemption = await api('/api/promotions/redeem', {
    method: 'POST',
    token: userToken,
    body: { code: 'test50' },
  });
  assert.equal(redemption.status, 200);
  assert.equal(redemption.data.user.mcoins, 50);

  const duplicateRedemption = await api('/api/promotions/redeem', {
    method: 'POST',
    token: userToken,
    body: { code: 'TEST50' },
  });
  assert.equal(duplicateRedemption.status, 400);

  const couponCreate = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'SHEET60',
      name: 'Test marketplace coupon',
      kind: 'marketplace_fixed',
      value: 60,
      minimumSpendMcoins: 100,
      maxRedemptions: 10,
      perUserLimit: 1,
    },
  });
  assert.equal(couponCreate.status, 201);

  const listingCreate = await api('/api/listings', {
    method: 'POST',
    token: sellerToken,
    body: {
      artist: 'Test Artist',
      title: 'Test Song',
      instrument: 'piano',
      format: 'JSON',
      priceMcoins: 100,
      description: 'Isolated integration-test listing.',
      filename: 'test-song.json',
      contentBase64: Buffer.from(JSON.stringify({ title: 'Test Song', notes: [] })).toString('base64'),
      rightsConfirmed: true,
      feeConfirmed: true,
    },
  });
  assert.equal(listingCreate.status, 201);

  const discountedPurchase = await api(`/api/listings/${listingCreate.data.listing.id}/purchase`, {
    method: 'POST',
    token: userToken,
    body: { promotionCode: 'SHEET60' },
  });
  assert.equal(discountedPurchase.status, 201);
  assert.equal(discountedPurchase.data.purchase.grossMcoins, 100);
  assert.equal(discountedPurchase.data.purchase.promotionDiscountMcoins, 60);
  assert.equal(discountedPurchase.data.purchase.buyerPaidMcoins, 40);
  assert.equal(discountedPurchase.data.user.mcoins, 10);

  const friendVoucherCreate = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'FRIEND90',
      name: 'Test Friend ID voucher',
      kind: 'friend_id_percent',
      value: 90,
      minimumSpendMcoins: 100,
      maxRedemptions: 0,
      perUserLimit: 0,
    },
  });
  assert.equal(friendVoucherCreate.status, 201);
  assert.equal(friendVoucherCreate.data.promotion.kind, 'friend_id_percent');

  const friendListingOne = await createListing(sellerToken, 'Friend Test One');
  const friendListingTwo = await createListing(sellerToken, 'Friend Test Two');
  assert.equal(friendListingOne.status, 201);
  assert.equal(friendListingTwo.status, 201);

  const firstFriendPurchase = await api(`/api/listings/${friendListingOne.data.listing.id}/purchase`, {
    method: 'POST',
    token: userToken,
    body: { friendId: sellerFriendId },
  });
  assert.equal(firstFriendPurchase.status, 201);
  assert.equal(firstFriendPurchase.data.purchase.promotionDiscountMcoins, 90);
  assert.equal(firstFriendPurchase.data.purchase.friendId, sellerFriendId);

  const secondFriendPurchase = await api(`/api/listings/${friendListingTwo.data.listing.id}/purchase`, {
    method: 'POST',
    token: adultToken,
    body: { friendId: sellerFriendId },
  });
  assert.equal(secondFriendPurchase.status, 201);
  assert.equal(secondFriendPurchase.data.purchase.promotionDiscountMcoins, 90);
  assert.equal(secondFriendPurchase.data.purchase.friendId, sellerFriendId);

  const selfReferralBlocked = await api(`/api/listings/${friendListingTwo.data.listing.id}/purchase`, {
    method: 'POST',
    token: adminToken,
    body: { friendId: adminFriendId },
  });
  assert.equal(selfReferralBlocked.status, 400);
  assert.match(selfReferralBlocked.data.error, /not your own/i);

  const reset = await api(`/api/admin/users/${userId}/reset-password`, {
    method: 'POST',
    token: adminToken,
    body: {},
  });
  assert.equal(reset.status, 200);
  assert.match(reset.data.temporaryPassword, /^PM-/);

  const oldSession = await api('/api/auth/me', { token: userToken });
  assert.equal(oldSession.status, 401);

  const temporaryLogin = await api('/api/auth/login', {
    method: 'POST',
    body: { identifier: 'customer@example.test', password: reset.data.temporaryPassword },
  });
  assert.equal(temporaryLogin.status, 200);
  assert.equal(temporaryLogin.data.user.mustChangePassword, true);

  const database = JSON.parse(fs.readFileSync(path.join(testDataDir, 'database.json'), 'utf8'));
  const customer = database.users.find((item) => item.id === userId);
  const policyCompliantUser = database.users.find((item) => item.email === 'adult@example.test');
  assert.ok(customer.passwordHash);
  assert.notEqual(customer.passwordHash, reset.data.temporaryPassword);
  assert.ok(policyCompliantUser.policyAcceptedAt);
  assert.equal(policyCompliantUser.birthDate, undefined);
  assert.ok(database.sessions.every((session) => session.tokenHash && !session.token));
  assert.equal(database.settings.minimumWithdrawalMcoins, 250);
  assert.equal(database.promotions.length, 3);
  assert.equal(database.promotionRedemptions.length, 5);
  assert.equal(database.promotionRedemptions.filter((entry) => entry.friendId === sellerFriendId).length, 2);
  assert.equal(database.passwordResetEvents.length, 1);
  assert.ok(Array.isArray(database.mediaTranscriptionJobs));
});
