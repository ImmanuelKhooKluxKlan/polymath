const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { createMlOperations, evaluationView, parseConfiguration } = require('./mlOperations');

function temporaryRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-mlops-test-'));
}

test('validates conservative candidate configuration limits', () => {
  const parsed = parseConfiguration({
    datasetId: 'phase-2-v001',
    version: 'phase2-v001',
    epochs: 1,
    trainLastLayers: 1,
    learningRate: 0.000002,
  });
  assert.equal(parsed.instrument, 'acoustic_piano');
  assert.equal(parsed.precision, 'bf16');
  assert.throws(() => parseConfiguration({ ...parsed, version: 'original' }), /must look like/);
  assert.throws(() => parseConfiguration({ ...parsed, epochs: 10 }), /1–3/);
  assert.throws(() => parseConfiguration({ ...parsed, learningRate: 0.1 }), /at most/);
});

test('creates append-only drafts and requires rights plus typed confirmation before GPU training', async () => {
  const root = temporaryRoot();
  const submissions = [];
  const runpod = {
    configured: true,
    missing: [],
    storageTargetCount: 2,
    async submitAction(input) {
      submissions.push(input);
      return { id: 'runpod-train-1', status: 'IN_QUEUE' };
    },
  };
  const operations = createMlOperations({ dataRoot: root, runpod });
  try {
    const draft = await operations.createDraft({
      datasetId: 'phase-2-v001',
      version: 'phase2-v001',
      epochs: 1,
      trainLastLayers: 1,
      learningRate: 0.000002,
    }, 'admin@example.test');
    assert.equal(draft.status, 'draft');
    assert.equal(draft.baseCheckpoint.immutable, true);
    assert.match(draft.candidateCheckpoint.label, /muscriptor-tester\/phase2-v001$/);
    await assert.rejects(
      operations.startTraining(draft.id, { confirmVersion: 'wrong', rightsAcknowledged: true }),
      /Type phase2-v001 exactly/,
    );
    await assert.rejects(
      operations.startTraining(draft.id, { confirmVersion: 'phase2-v001', rightsAcknowledged: false }),
      /right to use/,
    );
    const training = await operations.startTraining(draft.id, {
      confirmVersion: 'phase2-v001',
      rightsAcknowledged: true,
    }, 'admin@example.test');
    assert.equal(training.status, 'training');
    assert.equal(training.remote.jobId, 'runpod-train-1');
    assert.equal(submissions[0].action, 'train_piano_candidate');
    assert.equal(submissions[0].rights_acknowledgement, 'I_HAVE_TRAINING_RIGHTS');
    await assert.rejects(operations.createDraft({
      datasetId: 'another-dataset',
      version: 'phase2-v001',
      epochs: 1,
      trainLastLayers: 1,
      learningRate: 0.000002,
    }), /already exists/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('captures RunPod training output without exposing service credentials', async () => {
  const root = temporaryRoot();
  let remoteStatus = {
    status: 'COMPLETED',
    output: {
      version: 'phase3-v001',
      metadata: {
        baseSha256: 'safe-sha',
        baselineValidationLoss: 0.7,
        bestValidationLoss: 0.6,
        trainAudit: { clips: 20, songs: 2 },
        validationAudit: { clips: 10, songs: 1 },
      },
    },
  };
  const runpod = {
    configured: true,
    missing: [],
    storageTargetCount: 1,
    async submitAction() { return { id: 'secretless-job-id' }; },
    async getJobStatus() { return remoteStatus; },
  };
  const operations = createMlOperations({ dataRoot: root, runpod });
  try {
    const draft = await operations.createDraft({
      datasetId: 'phase-3-v001', version: 'phase3-v001', epochs: 1,
      trainLastLayers: 1, learningRate: 0.000002,
    });
    await operations.startTraining(draft.id, { confirmVersion: 'phase3-v001', rightsAcknowledged: true });
    const trained = await operations.refresh(draft.id);
    assert.equal(trained.status, 'trained');
    assert.equal(trained.baseCheckpoint.sha256, 'safe-sha');
    assert.equal(trained.metrics.validationLoss.candidate, 0.6);
    const serialized = JSON.stringify(operations.systemOverview());
    assert.doesNotMatch(serialized, /apiKey|secretAccessKey|Bearer/i);
    remoteStatus = { status: 'FAILED', error: 'fixture failure' };
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('turns frozen decoder output into accuracy and error diffs', () => {
  const view = evaluationView({
    baseline: {
      '50ms': { microF1: 0.2 }, '100ms': { microF1: 0.3 }, '250ms': { microF1: 0.5 },
      diagnostics50ms: {
        matchedNotes: 100, ignoredNotes: 60, falsePositiveNotes: 80, cutOffNotes: 12, overlongNotes: 9,
        onsetOnly: { f1: 0.21 }, onsetAndOffset: { f1: 0.12 }, frame: { f1: 0.5 },
        errorCauses: { octaveSubstitution: 10, timingNearMiss: 20 },
        patternRecognition: { chords: { complete: 4, missed: 8 } },
      },
    },
    candidate: {
      '50ms': { microF1: 0.25 }, '100ms': { microF1: 0.32 }, '250ms': { microF1: 0.52 },
      diagnostics50ms: {
        matchedNotes: 110, ignoredNotes: 50, falsePositiveNotes: 70, cutOffNotes: 14, overlongNotes: 7,
        onsetOnly: { f1: 0.24 }, onsetAndOffset: { f1: 0.13 }, frame: { f1: 0.51 },
        errorCauses: { octaveSubstitution: 8, timingNearMiss: 18 },
        patternRecognition: { chords: { complete: 5, missed: 7 } },
      },
    },
  });
  assert.ok(Math.abs(view.metrics.onsetF1At50ms.relativeChangePercent - 25) < 1e-9);
  assert.deepEqual(view.errorDiff.find((row) => row.label === 'Severe cutoffs'), {
    label: 'Severe cutoffs', baseline: 12, candidate: 14, delta: 2, preferredDirection: 'down',
  });
});
