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

const {
  app,
  muscriptorConstraints,
  normalizeReadyToPlaySong,
  selectMuscriptorExecution,
} = require('./server');

test('PDF normalization preserves written duration, key hold, release, and pedal provenance', () => {
  const result = normalizeReadyToPlaySong({
    isInstrumentalMusicSheet: true,
    instrument: 'piano',
    bpm: 100,
    timeSignature: { numerator: 4, denominator: 4 },
    notes: [{
      note: 'C4', time: 0, duration: 1, scoreDuration: 1,
      visualDuration: 1, audioDuration: 0.52, releaseSeconds: 0.61,
      velocity: 0.76, hand: 'right', voice: 'right up voice', articulation: 'staccato',
    }],
    pedals: [{
      time: 0.05, down: true, value: 96, source: 'inferred-score-pedaling',
      inferred: true, confidence: 0.58,
    }],
    performance: {
      profile: 'polymath-score-pianist-v1', preserveScoreDurations: true,
      preserveScoreTiming: true, durationFieldPolicy: 'written-key-hold-damper-v1',
    },
    pianoPerformance: {
      voices: 4, restrikesGivenReleaseGap: 3, legatoConnections: 2,
      pedalSource: 'inferred-score-pedaling', pedalEvents: 1,
      writtenAndPhysicalDurationsSeparated: true,
    },
  }, 'piano');
  assert.equal(result.notes[0].scoreDuration, 1);
  assert.equal(result.notes[0].audioDuration, 0.52);
  assert.equal(result.notes[0].releaseSeconds, 0.61);
  assert.equal(result.pedals[0].source, 'inferred-score-pedaling');
  assert.equal(result.pedals[0].inferred, true);
  assert.equal(result.performance.preserveScoreDurations, true);
  assert.equal(result.pianoPerformance.writtenAndPhysicalDurationsSeparated, true);
});

test('RunPod Serverless takes priority over the temporary SSH worker', () => {
  assert.equal(selectMuscriptorExecution({
    serverlessConfigured: true,
    remoteUrl: 'http://127.0.0.1:11111',
  }), 'runpod-serverless');
  assert.equal(selectMuscriptorExecution({
    serverlessConfigured: false,
    remoteUrl: 'http://127.0.0.1:11111',
  }), 'remote-gpu');
  assert.equal(selectMuscriptorExecution({
    serverlessConfigured: false,
    remoteUrl: '',
  }), 'local');
});

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
  assert.equal(result.notes.length, 5);
  assert.equal(c4Notes.length, 3);
  assert.equal(c4Notes[0].instrument, 'acoustic_piano');
  assert.equal(c4Notes.some((note) => note.instrument === 'electric_piano'), true);
  const acousticC4 = c4Notes.filter((note) => note.instrument === 'acoustic_piano');
  assert.ok(acousticC4[0].time + acousticC4[0].duration < acousticC4[1].time);
  assert.equal(result.notes.find((note) => note.midi === 64).duration, 8);
  assert.ok(result.notes.find((note) => note.midi === 64).velocity
    > result.notes.find((note) => note.midi === 67).velocity);
  assert.deepEqual(result.transcriptionCleanup, {
    version: 3,
    duplicateScope: 'same-instrument-and-pitch',
    inputNotes: 6,
    outputNotes: 5,
    removedDuplicateNotes: 1,
    removedRapidRetriggers: 1,
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

  assert.equal(full.notes.length, 3);
  assert.equal(full.notes[0].instrument, 'voice');
  assert.equal(full.transcriptionCleanup.vocalMelodyNotes, 1);
  assert.equal(full.transcriptionCleanup.excludedVocalNotes, 0);
  assert.ok(full.notes.find((note) => note.instrument === 'voice').velocity
    > instrumental.notes.find((note) => note.instrument === 'acoustic_piano').velocity);
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

test('selected piano and guitar receive the complete score before revoicing', () => {
  assert.deepEqual(muscriptorConstraints('piano', 'full'), []);
  assert.deepEqual(muscriptorConstraints('piano', 'instrumental'), []);
  assert.deepEqual(muscriptorConstraints('guitar', 'full'), []);
  assert.deepEqual(muscriptorConstraints('guitar', 'instrumental'), []);
  assert.deepEqual(muscriptorConstraints('electric-guitar', 'full'), []);
  assert.deepEqual(muscriptorConstraints('violin', 'instrumental'), ['violin']);
});

test('selected acoustic guitar keeps MIDI timing while enforcing a playable six-string score', () => {
  const payload = {
    title: 'Guitar arrangement fixture',
    notes: [
      { midi: 28, time: 1, duration: 0.8, velocity: 0.7, instrument: 'acoustic_bass' },
      { midi: 40, time: 1.01, duration: 0.9, velocity: 0.7, instrument: 'acoustic_guitar' },
      { midi: 52, time: 1.012, duration: 0.7, velocity: 0.8, instrument: 'acoustic_guitar' },
      { midi: 55, time: 1.014, duration: 0.7, velocity: 0.8, instrument: 'voice' },
      { midi: 59, time: 1.016, duration: 0.7, velocity: 0.8, instrument: 'acoustic_piano' },
      { midi: 64, time: 1.018, duration: 0.7, velocity: 0.8, instrument: 'acoustic_piano' },
      { midi: 67, time: 1.02, duration: 0.7, velocity: 0.8, instrument: 'acoustic_piano' },
      { midi: 76, time: 1.022, duration: 20, velocity: 0.8, instrument: 'flutes' },
      { midi: 52, time: 1.04, duration: 0.2, velocity: 0.8, instrument: 'acoustic_guitar' },
    ],
  };
  const result = postProcessMuscriptorResult(payload, {
    instrument: 'guitar',
    playbackMode: 'full',
  });

  assert.equal(result.instrumentGroups[0], 'acoustic_guitar');
  assert.equal(result.performance.profile, 'selected-guitar-midi-phrasing-v1');
  assert.ok(result.notes.length <= 6);
  assert.ok(result.notes.every((note) => note.midi >= 40 && note.midi <= 88));
  assert.ok(result.notes.every((note) => note.instrument === 'acoustic_guitar'));
  assert.ok(result.notes.every((note) => note.duration <= 6));
  assert.ok(result.notes.every((note) => note.time === 1));
  assert.ok(result.instrumentArrangement.removedDuplicateNotes >= 1);
  assert.ok(result.instrumentArrangement.removedUnplayableChordNotes >= 1);
});

test('instrumental guitar excludes voice before revoicing the selected instrument', () => {
  const result = postProcessMuscriptorResult({
    notes: [
      { midi: 69, time: 0, duration: 0.5, velocity: 0.8, instrument: 'voice' },
      { midi: 52, time: 0.5, duration: 0.5, velocity: 0.8, instrument: 'acoustic_guitar' },
    ],
  }, { instrument: 'guitar', playbackMode: 'instrumental' });
  assert.deepEqual(result.notes.map((note) => note.midi), [52]);
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

  const health = await api('/api/health');
  assert.equal(health.status, 200);
  assert.deepEqual(health.data, {
    ok: true,
    storage: 'atomic-json',
    artifacts: 'local-disk',
    queue: 'in-process',
    region: 'local',
  });

  const stateHealth = await api('/api/health/state');
  assert.equal(stateHealth.status, 200);
  assert.equal(stateHealth.data.ok, true);
  assert.equal(stateHealth.data.state, 'ready');

  const catalog = await api('/api/catalog');
  assert.equal(catalog.status, 200);
  assert.equal(catalog.data.mcoinsPerUsd, 1);
  assert.equal(catalog.data.withdrawalFeeRate, 0.25);
  assert.equal(catalog.data.marketplaceFeeRate, 0.25);
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
  assert.equal(adminRegistration.data.user.unlimitedMcoins, true);
  assert.equal(adminRegistration.data.user.translationAllowance.plan, 'admin');
  assert.equal(adminRegistration.data.user.translationAllowance.unlimited, true);
  assert.equal(adminRegistration.data.user.translationAllowance.limit, null);
  assert.equal(adminRegistration.data.user.translationAllowance.remaining, null);
  assert.match(adminRegistration.data.user.friend_id, /^user_[a-f0-9]{5}$/);
  const adminToken = adminRegistration.data.token;
  const adminFriendId = adminRegistration.data.user.friend_id;

  const localUploadIntent = await api('/api/artifact-upload-intents', {
    method: 'POST',
    token: adminToken,
    body: {
      purpose: 'score-translation',
      filename: 'test-score.pdf',
      contentType: 'application/pdf',
      size: 100,
    },
  });
  assert.equal(localUploadIntent.status, 200);
  assert.deepEqual(localUploadIntent.data, { direct: false });

  const userRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Test Customer',
      email: 'customer@example.test',
      password: 'CustomerPassword123',
    },
  });
  assert.equal(userRegistration.status, 201);
  assert.equal(userRegistration.data.user.unlimitedMcoins, false);
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

  const unauthorizedGrant = await api(`/api/admin/users/${userId}/mcoins`, {
    method: 'POST',
    token: userToken,
    body: { amountMcoins: 25 },
  });
  assert.equal(unauthorizedGrant.status, 403);

  const mcoinGrant = await api(`/api/admin/users/${userId}/mcoins`, {
    method: 'POST',
    token: adminToken,
    body: { amountMcoins: 125.5 },
  });
  assert.equal(mcoinGrant.status, 200);
  assert.equal(mcoinGrant.data.user.mcoins, 125.5);
  assert.equal(mcoinGrant.data.user.unlimitedMcoins, false);

  const personalSongPayload = {
    title: 'Cloud Library Test',
    composer: 'Test Artist',
    instrument: 'piano',
    notes: [{ note: 'C4', time: 0, duration: 0.5, velocity: 0.8 }],
  };
  const personalSongUpload = await api('/api/ready-sheet-uploads', {
    method: 'POST',
    token: userToken,
    body: {
      filename: 'cloud-library-test.json',
      title: personalSongPayload.title,
      artist: personalSongPayload.composer,
      instrument: 'piano',
      contentBase64: Buffer.from(JSON.stringify(personalSongPayload)).toString('base64'),
    },
  });
  assert.equal(personalSongUpload.status, 201);
  assert.equal(personalSongUpload.data.personalSong.title, 'Cloud Library Test');
  assert.equal(personalSongUpload.data.personalSong.artist, 'Test Artist');
  assert.equal(personalSongUpload.data.personalSong.instrument, 'piano');
  assert.equal(personalSongUpload.data.alreadySaved, false);

  const duplicatePersonalSong = await api('/api/ready-sheet-uploads', {
    method: 'POST',
    token: userToken,
    body: {
      filename: 'cloud-library-test-copy.json',
      title: personalSongPayload.title,
      artist: personalSongPayload.composer,
      instrument: 'piano',
      contentBase64: Buffer.from(JSON.stringify(personalSongPayload)).toString('base64'),
    },
  });
  assert.equal(duplicatePersonalSong.status, 200);
  assert.equal(duplicatePersonalSong.data.alreadySaved, true);
  assert.equal(duplicatePersonalSong.data.personalSong.id, personalSongUpload.data.personalSong.id);

  const personalLibrary = await api('/api/library', { token: userToken });
  assert.equal(personalLibrary.status, 200);
  assert.equal(personalLibrary.data.personalSongs.length, 1);
  assert.equal(personalLibrary.data.personalSongs[0].title, 'Cloud Library Test');
  assert.equal(Object.prototype.hasOwnProperty.call(personalLibrary.data.personalSongs[0], 'assetPath'), false);

  const otherAccountDownload = await api(`/api/personal-songs/${personalSongUpload.data.personalSong.id}/download`, {
    token: adminToken,
  });
  assert.equal(otherAccountDownload.status, 404);

  const personalSongDownload = await api(`/api/personal-songs/${personalSongUpload.data.personalSong.id}/download`, {
    token: userToken,
  });
  assert.equal(personalSongDownload.status, 200);
  assert.equal(personalSongDownload.data.title, 'Cloud Library Test');

  const personalSongDelete = await api(`/api/personal-songs/${personalSongUpload.data.personalSong.id}`, {
    method: 'DELETE',
    token: userToken,
  });
  assert.equal(personalSongDelete.status, 200);
  const emptyPersonalLibrary = await api('/api/library', { token: userToken });
  assert.equal(emptyPersonalLibrary.data.personalSongs.length, 0);

  const legacyOutputFilename = 'legacy-media-output.json';
  const legacyOutput = {
    title: 'Earlier Cloud Translation',
    instrument: 'piano',
    notes: [{ note: 'D4', time: 0, duration: 0.75, velocity: 0.7 }],
  };
  fs.mkdirSync(path.join(testDataDir, 'uploads'), { recursive: true });
  fs.writeFileSync(
    path.join(testDataDir, 'uploads', legacyOutputFilename),
    JSON.stringify(legacyOutput),
  );
  const legacyFixturePath = path.join(testDataDir, 'database.json');
  const legacyFixture = JSON.parse(fs.readFileSync(legacyFixturePath, 'utf8'));
  legacyFixture.mediaTranscriptionJobs.push({
    id: 'media_tx_legacy_cloud_test',
    userId,
    filename: 'earlier-song.mp3',
    title: legacyOutput.title,
    instrument: 'piano',
    outputPath: legacyOutputFilename,
    outputFilename: legacyOutputFilename,
    status: 'completed',
    progress: 100,
    startedAt: new Date(Date.now() - 2000).toISOString(),
    completedAt: new Date(Date.now() - 1000).toISOString(),
  });
  fs.writeFileSync(legacyFixturePath, JSON.stringify(legacyFixture, null, 2));

  const backfilledPersonalLibrary = await api('/api/library', { token: userToken });
  assert.equal(backfilledPersonalLibrary.data.personalSongs.length, 1);
  assert.equal(backfilledPersonalLibrary.data.personalSongs[0].title, 'Earlier Cloud Translation');
  const backfilledSongId = backfilledPersonalLibrary.data.personalSongs[0].id;
  const removeBackfilledSong = await api(`/api/personal-songs/${backfilledSongId}`, {
    method: 'DELETE',
    token: userToken,
  });
  assert.equal(removeBackfilledSong.status, 200);
  const hiddenPersonalLibrary = await api('/api/library', { token: userToken });
  assert.equal(hiddenPersonalLibrary.data.personalSongs.length, 0);
  assert.equal(fs.existsSync(path.join(testDataDir, 'uploads', legacyOutputFilename)), true);

  const firstSubscriptionGrant = await api(`/api/admin/users/${userId}/subscription`, {
    method: 'POST',
    token: adminToken,
    body: { tier: 'musician', interval: 'MONTH' },
  });
  assert.equal(firstSubscriptionGrant.status, 200);
  assert.equal(firstSubscriptionGrant.data.extended, false);
  assert.equal(firstSubscriptionGrant.data.user.subscriptionTier, 'musician');
  assert.equal(firstSubscriptionGrant.data.user.subscriptionInterval, 'MONTH');
  assert.equal(firstSubscriptionGrant.data.user.adminSubscriptionGrant.active, true);
  const firstGrantExpiry = new Date(firstSubscriptionGrant.data.user.adminSubscriptionGrant.expiresAt);

  const renewedSubscriptionGrant = await api(`/api/admin/users/${userId}/subscription`, {
    method: 'POST',
    token: adminToken,
    body: { tier: 'musician', interval: 'MONTH' },
  });
  assert.equal(renewedSubscriptionGrant.status, 200);
  assert.equal(renewedSubscriptionGrant.data.extended, true);
  assert.ok(new Date(renewedSubscriptionGrant.data.user.adminSubscriptionGrant.expiresAt) > firstGrantExpiry);

  const adminUsers = await api('/api/admin/users', { token: adminToken });
  assert.equal(adminUsers.status, 200);
  const managedCustomer = adminUsers.data.rows.find((row) => row.userId === userId);
  const managedAdmin = adminUsers.data.rows.find((row) => row.userId === adminRegistration.data.user.user_id);
  assert.equal(managedCustomer.mcoins, 125.5);
  assert.equal(managedCustomer.subscriptionTier, 'musician');
  assert.equal(managedCustomer.adminSubscriptionGrant.active, true);
  assert.equal(managedAdmin.unlimitedMcoins, true);

  const removedSubscriptionGrant = await api(`/api/admin/users/${userId}/subscription`, {
    method: 'DELETE',
    token: adminToken,
  });
  assert.equal(removedSubscriptionGrant.status, 200);
  assert.equal(removedSubscriptionGrant.data.user.adminSubscriptionGrant, null);
  assert.equal(removedSubscriptionGrant.data.user.subscriptionTier, 'free');

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
      minimumPasswordLength: 1,
      minimumMarketplacePriceMcoins: 30,
      maximumMarketplacePriceMcoins: 100000,
      marketplaceFeePercent: 25,
      listenerRewardsEnabled: true,
      maximumListenerRewardMcoins: 5,
      maximumRewardOutflowPerListingMcoins: 5,
      minimumWithdrawalMcoins: 250,
      maximumWithdrawalMcoins: 250,
      dailyWithdrawalLimitMcoins: 250,
      maximumPendingWithdrawalOutflowMcoins: 187.5,
      withdrawalFeePercent: 25,
      welcomeMcoins: 25,
      policyNotice: 'Adults only during this test.',
      supportEmail: 'support@example.test',
    },
  });
  assert.equal(policyUpdate.status, 200);
  assert.equal(policyUpdate.data.policies.minimumSignupAge, 18);
  assert.equal(policyUpdate.data.policies.minimumPasswordLength, 1);
  assert.equal(policyUpdate.data.policies.maximumRewardOutflowPerListingMcoins, 5);

  const oneCharacterPasswordRegistration = await register('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Admin Policy Minimum Test',
      email: 'one-character-password@example.test',
      password: 'x',
      birthDate: '1990-01-01',
      termsAccepted: true,
    },
  });
  assert.equal(oneCharacterPasswordRegistration.status, 201);

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
      name: 'Test signup percentage',
      kind: 'subscription_percent',
      value: 50,
      maxRedemptions: 10,
      perUserLimit: 1,
    },
  });
  assert.equal(promotionCreate.status, 201);
  assert.equal(promotionCreate.data.promotion.code, 'TEST50');

  const freeMcoinVoucherBlocked = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'FREE500',
      name: 'Blocked wallet credit',
      kind: 'mcoin_credit',
      value: 500,
    },
  });
  assert.equal(freeMcoinVoucherBlocked.status, 400);

  const fixedValueCoupon = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'FIXED50',
      name: 'Fixed Mcoin discount',
      kind: 'marketplace_fixed',
      value: 50,
    },
  });
  assert.equal(fixedValueCoupon.status, 201);
  assert.equal(fixedValueCoupon.data.promotion.value, 50);

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
  assert.equal(luckyRegistration.data.user.mcoins, 25);
  const luckyToken = luckyRegistration.data.token;

  const cashoutFixturePath = path.join(testDataDir, 'database.json');
  const cashoutFixture = JSON.parse(fs.readFileSync(cashoutFixturePath, 'utf8'));
  const luckyCashoutUser = cashoutFixture.users.find((item) => item.email === 'lucky-adult@example.test');
  luckyCashoutUser.mcoins += 300;
  cashoutFixture.ledger.push({
    id: 'ledger_paid_mcoins_fixture',
    userId: luckyCashoutUser.id,
    amount: 300,
    type: 'paypal_mcoin_purchase_fixture',
    note: 'Integration-test paid wallet balance',
    createdAt: new Date().toISOString(),
  });
  fs.writeFileSync(cashoutFixturePath, JSON.stringify(cashoutFixture, null, 2));

  const overMaximumCashout = await api('/api/wallet/withdraw', {
    method: 'POST',
    token: luckyToken,
    body: { amountMcoins: 251, payoutEmail: 'lucky-payout@example.test' },
  });
  assert.equal(overMaximumCashout.status, 400);
  assert.match(overMaximumCashout.data.error, /maximum withdrawal/i);

  const regularAccountCashout = await api('/api/wallet/withdraw', {
    method: 'POST',
    token: luckyToken,
    body: { amountMcoins: 250, payoutEmail: 'lucky-payout@example.test' },
  });
  assert.equal(regularAccountCashout.status, 201);
  assert.equal(regularAccountCashout.data.withdrawal.amountMcoins, 250);
  assert.equal(regularAccountCashout.data.withdrawal.feeRate, 0.25);
  assert.equal(regularAccountCashout.data.withdrawal.feeMcoins, 62.5);
  assert.equal(regularAccountCashout.data.withdrawal.netMcoins, 187.5);
  assert.equal(regularAccountCashout.data.user.mcoins, 75);

  const cashoutWallet = await api('/api/wallet', { token: luckyToken });
  assert.equal(cashoutWallet.status, 200);
  assert.equal(cashoutWallet.data.withdrawalFeeRate, 0.25);
  assert.equal(cashoutWallet.data.withdrawals.length, 1);
  assert.equal(cashoutWallet.data.withdrawals[0].status, 'pending_manual_review');

  const adminWithdrawalQueue = await api('/api/admin/withdrawals', { token: adminToken });
  assert.equal(adminWithdrawalQueue.status, 200);
  assert.equal(adminWithdrawalQueue.data.summary.pendingCount, 1);
  assert.equal(adminWithdrawalQueue.data.summary.pendingNetMcoins, 187.5);
  const paidWithdrawal = await api(`/api/admin/withdrawals/${regularAccountCashout.data.withdrawal.id}`, {
    method: 'PATCH',
    token: adminToken,
    body: { status: 'paid' },
  });
  assert.equal(paidWithdrawal.status, 200);
  assert.equal(paidWithdrawal.data.withdrawal.status, 'paid');

  const retiredWalletRedemption = await api('/api/promotions/redeem', {
    method: 'POST',
    token: userToken,
    body: { code: 'test50' },
  });
  assert.equal(retiredWalletRedemption.status, 410);
  assert.match(retiredWalletRedemption.data.error, /percentage discounts only/i);

  const couponCreate = await api('/api/admin/promotions', {
    method: 'POST',
    token: adminToken,
    body: {
      code: 'SHEET60',
      name: 'Test marketplace coupon',
      kind: 'marketplace_percent',
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
  assert.equal(listingCreate.data.listing.marketplaceFeeRate, 0.25);

  const discountedPurchase = await api(`/api/listings/${listingCreate.data.listing.id}/purchase`, {
    method: 'POST',
    token: luckyToken,
    body: { promotionCode: 'SHEET60' },
  });
  assert.equal(discountedPurchase.status, 201);
  assert.equal(discountedPurchase.data.purchase.grossMcoins, 100);
  assert.equal(discountedPurchase.data.purchase.platformFeeRate, 0.25);
  assert.equal(discountedPurchase.data.purchase.platformFeeMcoins, 25);
  assert.equal(discountedPurchase.data.purchase.sellerEarningsMcoins, 75);
  assert.equal(discountedPurchase.data.purchase.promotionDiscountMcoins, 60);
  assert.equal(discountedPurchase.data.purchase.buyerPaidMcoins, 40);
  assert.equal(discountedPurchase.data.user.mcoins, 35);

  const fixedDiscountPurchase = await api(`/api/listings/${listingCreate.data.listing.id}/purchase`, {
    method: 'POST',
    token: userToken,
    body: { promotionCode: 'FIXED50' },
  });
  assert.equal(fixedDiscountPurchase.status, 201);
  assert.equal(fixedDiscountPurchase.data.purchase.grossMcoins, 100);
  assert.equal(fixedDiscountPurchase.data.purchase.promotionDiscountMcoins, 50);
  assert.equal(fixedDiscountPurchase.data.purchase.buyerPaidMcoins, 50);
  assert.equal(fixedDiscountPurchase.data.purchase.sellerEarningsMcoins, 75);

  const administratorPurchase = await api(`/api/listings/${listingCreate.data.listing.id}/purchase`, {
    method: 'POST',
    token: adminToken,
    body: {},
  });
  assert.equal(administratorPurchase.status, 201);
  assert.equal(administratorPurchase.data.purchase.paymentMethod, 'administrator_unlimited');
  assert.equal(administratorPurchase.data.user.unlimitedMcoins, true);
  assert.equal(administratorPurchase.data.user.mcoins, 0);

  const honestReview = await api(`/api/listings/${listingCreate.data.listing.id}/reviews`, {
    method: 'POST',
    token: luckyToken,
    body: { rating: 2, comment: 'The timing needs more work.' },
  });
  assert.equal(honestReview.status, 201);
  assert.equal(honestReview.data.review.verifiedPurchase, true);
  assert.equal(honestReview.data.summary.averageRating, 2);
  assert.equal(honestReview.data.summary.reviewCount, 1);

  const sellerSelfReview = await api(`/api/listings/${listingCreate.data.listing.id}/reviews`, {
    method: 'POST',
    token: sellerToken,
    body: { rating: 5, comment: 'My own sheet is perfect.' },
  });
  assert.equal(sellerSelfReview.status, 403);

  const sellerDeleteReview = await api(`/api/listings/${listingCreate.data.listing.id}/reviews/${honestReview.data.review.id}`, {
    method: 'DELETE',
    token: sellerToken,
  });
  assert.equal(sellerDeleteReview.status, 403);
  assert.match(sellerDeleteReview.data.error, /permanent/i);

  const publicReviews = await api(`/api/listings/${listingCreate.data.listing.id}/reviews`);
  assert.equal(publicReviews.status, 200);
  assert.equal(publicReviews.data.reviews.length, 1);
  assert.equal(publicReviews.data.reviews[0].comment, 'The timing needs more work.');

  const followedComposer = await api(`/api/composers/${sellerRegistration.data.user.user_id}/follow`, {
    method: 'POST',
    token: adultToken,
  });
  assert.equal(followedComposer.status, 200);
  assert.equal(followedComposer.data.composer.followerCount, 1);
  assert.equal(followedComposer.data.composer.isFollowing, true);

  const composerProfile = await api(`/api/composers/${sellerRegistration.data.user.user_id}`, { token: adultToken });
  assert.equal(composerProfile.status, 200);
  assert.equal(composerProfile.data.composer.averageRating, 2);
  assert.equal(composerProfile.data.composer.ratingCount, 1);
  assert.equal(composerProfile.data.composer.buyerCount, 3);
  assert.deepEqual(composerProfile.data.composer.ranking, {
    ratingPoints: 4,
    audiencePoints: 3,
    totalPoints: 7,
    maximumPoints: 50,
  });
  assert.equal(composerProfile.data.composer.followerCount, 1);
  assert.ok(composerProfile.data.listings.some((listing) => listing.id === listingCreate.data.listing.id));

  const freeListing = await api('/api/listings', {
    method: 'POST',
    token: sellerToken,
    body: {
      artist: 'Test Artist',
      title: 'Free Sheet',
      instrument: 'piano',
      format: 'JSON',
      listingMode: 'free',
      priceMcoins: 0,
      filename: 'free-sheet.json',
      contentBase64: Buffer.from(JSON.stringify({ title: 'Free Sheet', notes: [] })).toString('base64'),
      rightsConfirmed: true,
      feeConfirmed: false,
    },
  });
  assert.equal(freeListing.status, 201);
  assert.equal(freeListing.data.listing.listingMode, 'free');
  const freeClaim = await api(`/api/listings/${freeListing.data.listing.id}/purchase`, {
    method: 'POST',
    token: userToken,
    body: {},
  });
  assert.equal(freeClaim.status, 201);
  assert.equal(freeClaim.data.purchase.buyerPaidMcoins, 0);

  const rewardListing = await api('/api/listings', {
    method: 'POST',
    token: sellerToken,
    body: {
      artist: 'Test Artist',
      title: 'Listener Reward Sheet',
      instrument: 'piano',
      format: 'JSON',
      listingMode: 'listener-reward',
      listenerRewardMcoins: 5,
      filename: 'listener-reward-sheet.json',
      contentBase64: Buffer.from(JSON.stringify({ title: 'Listener Reward Sheet', notes: [] })).toString('base64'),
      rightsConfirmed: true,
      feeConfirmed: false,
    },
  });
  assert.equal(rewardListing.status, 201);
  assert.equal(rewardListing.data.listing.rewardAvailable, true);
  const rewardClaim = await api(`/api/listings/${rewardListing.data.listing.id}/purchase`, {
    method: 'POST',
    token: adultToken,
    body: {},
  });
  assert.equal(rewardClaim.status, 201);
  assert.equal(rewardClaim.data.purchase.paymentMethod, 'listener_reward');
  assert.equal(rewardClaim.data.purchase.listenerRewardMcoins, 5);
  assert.equal(rewardClaim.data.user.mcoins, 30);
  const duplicateRewardClaim = await api(`/api/listings/${rewardListing.data.listing.id}/purchase`, {
    method: 'POST',
    token: adultToken,
    body: {},
  });
  assert.equal(duplicateRewardClaim.status, 200);
  assert.equal(duplicateRewardClaim.data.user.mcoins, 30);
  const exhaustedRewardClaim = await api(`/api/listings/${rewardListing.data.listing.id}/purchase`, {
    method: 'POST',
    token: luckyToken,
    body: {},
  });
  assert.equal(exhaustedRewardClaim.status, 409);
  assert.match(exhaustedRewardClaim.data.error, /paused|exhausted/i);

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
    token: luckyToken,
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
  assert.equal(database.promotions.length, 4);
  assert.equal(database.promotionRedemptions.length, 5);
  assert.equal(database.promotionRedemptions.filter((entry) => entry.friendId === sellerFriendId).length, 2);
  assert.equal(database.passwordResetEvents.length, 1);
  assert.ok(Array.isArray(database.mediaTranscriptionJobs));
});
