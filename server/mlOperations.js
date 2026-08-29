const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const REMOTE_PREFIX = 'private/ml-operations/experiments';
const TERMINAL_FAILURES = new Set(['CANCELLED', 'FAILED', 'TIMED_OUT']);
const RIGHTS_ACKNOWLEDGEMENT = 'I_HAVE_TRAINING_RIGHTS';

const TOKEN_SYSTEM = {
  name: 'MuScriptor/MT3-style event vocabulary',
  modelCardinality: 1395,
  sequenceLimit: 2000,
  timeResolutionHz: 100,
  ranges: [
    { label: 'Control', first: 0, last: 2, purpose: 'Padding, end-of-sequence, and unknown events' },
    { label: 'Time shift', first: 3, last: 1003, purpose: 'Move the musical clock in 10 ms steps' },
    { label: 'Pitch', first: 1004, last: 1131, purpose: 'MIDI pitches 0–127 for note on/off events' },
    { label: 'Velocity state', first: 1132, last: 1133, purpose: 'Distinguish note-off from note-on' },
    { label: 'Tie', first: 1134, last: 1134, purpose: 'Continue a held note across a five-second clip boundary' },
    { label: 'Program', first: 1135, last: 1264, purpose: 'Condition notes on an individual instrument family/program' },
    { label: 'Drum', first: 1265, last: 1394, purpose: 'Percussion event/program vocabulary' },
  ],
};

const PIPELINE = [
  { id: 'intake', label: 'Feed evidence', platform: 'Admin browser → AWS API', detail: 'Upload source media, raw model output, and desired MIDI/JSON.' },
  { id: 'align', label: 'Align timelines', platform: 'AWS Node service', detail: 'Fit a piecewise time map; inspect offsets, tempo drift, pauses, pitch matches, and unsafe windows.' },
  { id: 'review', label: 'Human supervision', platform: 'Admin console', detail: 'Accept, reject, or anchor five-second regions. Uncertain labels remain outside training.' },
  { id: 'dataset', label: 'Build dataset', platform: 'Python + FFmpeg', detail: 'Split by song before clipping, create mono 16 kHz WAV clips, then encode reviewed note targets.' },
  { id: 'train', label: 'Change candidate weights', platform: 'RunPod Serverless GPU', detail: 'Start from immutable original weights and update only the selected final transformer block(s), output norm, and output head.' },
  { id: 'evaluate', label: 'Frozen evaluation', platform: 'RunPod Serverless GPU', detail: 'Decode an unseen song and compare onset, offset, frame, pitch-band, chord, cutoff, and repeat errors.' },
  { id: 'decide', label: 'Human promotion gate', platform: 'Admin + listening review', detail: 'A candidate stays research-only unless all required scores and listening checks pass.' },
];

const SEEDED_EXPERIMENT = {
  id: 'phase1-v002',
  schema: 'polymath-ml-experiment-v1',
  name: 'Piano Phase 1 v002',
  version: 'phase1-v002',
  datasetId: 'phase-1-v002',
  instrument: 'acoustic_piano',
  status: 'research-only',
  stage: 'Evaluation complete; promotion rejected',
  readOnly: true,
  createdAt: '2026-08-28T11:44:12.385Z',
  updatedAt: '2026-08-28T11:44:12.385Z',
  baseCheckpoint: {
    label: 'models/original',
    sha256: 'ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397',
    immutable: true,
  },
  candidateCheckpoint: { label: 'models/muscriptor-tester/phase1-v002', promoted: false },
  configuration: {
    epochs: 1,
    trainLastLayers: 1,
    learningRate: 0.000002,
    precision: 'bf16',
    optimizer: 'AdamW',
    gradientAccumulation: 8,
    timingTokenWeight: 1.15,
    noteOffTokenWeight: 1.25,
    eosTokenWeight: 1.2,
  },
  weightScope: {
    changed: ['final transformer block', 'output normalization', 'token output head'],
    frozen: ['audio encoder', 'conditioning provider', 'earlier transformer blocks', 'immutable original checkpoint files'],
    exactTensorDeltaAvailable: false,
    note: 'This older run recorded the trainable scope but not per-tensor delta norms. Future runs record that audit automatically.',
  },
  data: {
    training: { songs: 2, clips: 188, audioSeconds: 940, averageTokens: 152.88, minimumTokens: 15, maximumTokens: 206 },
    validation: { songs: 1, clips: 46, audioSeconds: 230, averageTokens: 123.78, minimumTokens: 14, maximumTokens: 155 },
    splitNote: 'Kiss Me and Back to December trained the model. 22 was frozen and unseen by the optimizer.',
  },
  metrics: {
    validationLoss: { baseline: 0.6274978778, candidate: 0.6247775477, relativeChangePercent: -0.43, preferredDirection: 'down' },
    onsetF1At50ms: { baseline: 0.196643, candidate: 0.209656, relativeChangePercent: 6.62, preferredDirection: 'up' },
    onsetF1At100ms: { baseline: 0.354383, candidate: 0.359029, relativeChangePercent: 1.31, preferredDirection: 'up' },
    onsetF1At250ms: { baseline: 0.539835, candidate: 0.545746, relativeChangePercent: 1.1, preferredDirection: 'up' },
    onsetOffsetF1: { baseline: 0.117744, candidate: 0.123865, relativeChangePercent: 5.2, preferredDirection: 'up' },
    frameF1: { baseline: 0.528557, candidate: 0.527897, relativeChangePercent: -0.12, preferredDirection: 'up' },
  },
  errorDiff: [
    { label: 'Correct matched onsets', baseline: 364, candidate: 388, delta: 24, preferredDirection: 'up' },
    { label: 'Ignored desired notes', baseline: 1140, candidate: 1116, delta: -24, preferredDirection: 'down' },
    { label: 'Extra predicted notes', baseline: 1767, candidate: 1741, delta: -26, preferredDirection: 'down' },
    { label: 'Octave substitutions', baseline: 96, candidate: 86, delta: -10, preferredDirection: 'down' },
    { label: 'Timing near misses', baseline: 595, candidate: 583, delta: -12, preferredDirection: 'down' },
    { label: 'Severe cutoffs', baseline: 67, candidate: 71, delta: 4, preferredDirection: 'down' },
    { label: 'Overlong notes', baseline: 25, candidate: 30, delta: 5, preferredDirection: 'down' },
    { label: 'Complete chords', baseline: 39, candidate: 42, delta: 3, preferredDirection: 'up' },
    { label: 'Missed chords', baseline: 217, candidate: 212, delta: -5, preferredDirection: 'down' },
    { label: 'False repeats ≤200 ms', baseline: 239, candidate: 254, delta: 15, preferredDirection: 'down' },
  ],
  decision: {
    approved: false,
    summary: 'Keep as a research checkpoint. Onset, octave, and chord recovery improved, but sustain, cutoffs, and medium-speed repeats did not improve together.',
    commercialUseAllowed: false,
  },
  audit: [
    { at: '2026-08-28T11:44:12.385Z', actor: 'RunPod worker', action: 'candidate-saved', detail: 'Original SHA-256 remained unchanged.' },
    { at: '2026-08-28T12:15:00.000Z', actor: 'evaluation pipeline', action: 'frozen-evaluation-complete', detail: 'Compared original and candidate on 22.' },
    { at: '2026-08-28T12:25:00.000Z', actor: 'human gate', action: 'promotion-rejected', detail: 'Retained for research because duration/repeat regressions remained.' },
  ],
};

function clean(value) {
  return String(value || '').trim();
}

function safeRecordPath(root, id) {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(id)) return null;
  const resolvedRoot = path.resolve(root);
  const filename = path.resolve(resolvedRoot, `${id}.json`);
  return filename.startsWith(`${resolvedRoot}${path.sep}`) ? filename : null;
}

function atomicJson(filename, value) {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.${crypto.randomUUID()}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filename);
}

function parseConfiguration(body = {}) {
  const datasetId = clean(body.datasetId).toLowerCase();
  const version = clean(body.version).toLowerCase();
  if (!/^[a-z0-9][a-z0-9-]{2,50}$/.test(datasetId)) {
    throw new Error('Dataset ID must use 3–51 lowercase letters, numbers, or hyphens.');
  }
  if (!/^phase\d+-v\d{3,}$/.test(version)) {
    throw new Error('Candidate version must look like phase2-v001.');
  }
  const epochs = Number(body.epochs);
  const trainLastLayers = Number(body.trainLastLayers);
  const learningRate = Number(body.learningRate);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 3) throw new Error('Epochs must be 1–3.');
  if (!Number.isInteger(trainLastLayers) || trainLastLayers < 1 || trainLastLayers > 2) throw new Error('Trainable final layers must be 1 or 2.');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.00001) throw new Error('Learning rate must be above 0 and at most 0.00001.');
  const tokenWeight = (name, fallback) => {
    const value = Number(body[name] ?? fallback);
    if (!Number.isFinite(value) || value < 0.5 || value > 3) throw new Error(`${name} must be between 0.5 and 3.`);
    return value;
  };
  return {
    datasetId,
    version,
    name: clean(body.name).slice(0, 100) || version,
    instrument: 'acoustic_piano',
    epochs,
    trainLastLayers,
    learningRate,
    timingTokenWeight: tokenWeight('timingTokenWeight', 1.15),
    noteOffTokenWeight: tokenWeight('noteOffTokenWeight', 1.25),
    eosTokenWeight: tokenWeight('eosTokenWeight', 1.2),
    precision: 'bf16',
    optimizer: 'AdamW',
  };
}

function relativeChange(baseline, candidate) {
  const before = Number(baseline);
  const after = Number(candidate);
  if (!Number.isFinite(before) || !Number.isFinite(after) || before === 0) return null;
  return (after - before) / Math.abs(before) * 100;
}

function evaluationView(output = {}) {
  const baseline = output.baseline || {};
  const candidate = output.candidate || {};
  const baseDetails = baseline.diagnostics50ms || {};
  const candidateDetails = candidate.diagnostics50ms || {};
  const metric = (before, after, preferredDirection = 'up') => ({
    baseline: before,
    candidate: after,
    relativeChangePercent: relativeChange(before, after),
    preferredDirection,
  });
  const row = (label, before, after, preferredDirection) => ({
    label,
    baseline: before,
    candidate: after,
    delta: Number.isFinite(Number(before)) && Number.isFinite(Number(after)) ? Number(after) - Number(before) : null,
    preferredDirection,
  });
  return {
    metrics: {
      onsetF1At50ms: metric(baseline['50ms']?.microF1, candidate['50ms']?.microF1),
      onsetF1At100ms: metric(baseline['100ms']?.microF1, candidate['100ms']?.microF1),
      onsetF1At250ms: metric(baseline['250ms']?.microF1, candidate['250ms']?.microF1),
      boundaryCorrectedOnsetF1: metric(baseDetails.onsetOnly?.f1, candidateDetails.onsetOnly?.f1),
      onsetOffsetF1: metric(baseDetails.onsetAndOffset?.f1, candidateDetails.onsetAndOffset?.f1),
      frameF1: metric(baseDetails.frame?.f1, candidateDetails.frame?.f1),
    },
    errorDiff: [
      row('Correct matched onsets', baseDetails.matchedNotes, candidateDetails.matchedNotes, 'up'),
      row('Ignored desired notes', baseDetails.ignoredNotes, candidateDetails.ignoredNotes, 'down'),
      row('Extra predicted notes', baseDetails.falsePositiveNotes, candidateDetails.falsePositiveNotes, 'down'),
      row('Octave substitutions', baseDetails.errorCauses?.octaveSubstitution, candidateDetails.errorCauses?.octaveSubstitution, 'down'),
      row('Near-pitch substitutions', baseDetails.errorCauses?.nearPitchSubstitution, candidateDetails.errorCauses?.nearPitchSubstitution, 'down'),
      row('Timing near misses', baseDetails.errorCauses?.timingNearMiss, candidateDetails.errorCauses?.timingNearMiss, 'down'),
      row('Spurious extras', baseDetails.errorCauses?.spuriousExtra, candidateDetails.errorCauses?.spuriousExtra, 'down'),
      row('Severe cutoffs', baseDetails.cutOffNotes, candidateDetails.cutOffNotes, 'down'),
      row('Overlong notes', baseDetails.overlongNotes, candidateDetails.overlongNotes, 'down'),
      row('Complete chords', baseDetails.patternRecognition?.chords?.complete, candidateDetails.patternRecognition?.chords?.complete, 'up'),
      row('Missed chords', baseDetails.patternRecognition?.chords?.missed, candidateDetails.patternRecognition?.chords?.missed, 'down'),
    ].filter((item) => Number.isFinite(item.baseline) && Number.isFinite(item.candidate)),
  };
}

function createMlOperations({ dataRoot, artifactStore = null, runpod }) {
  const recordsRoot = path.join(path.resolve(dataRoot), 'experiments');
  const remoteCacheRoot = path.join(path.resolve(dataRoot), 'remote-cache');
  fs.mkdirSync(recordsRoot, { recursive: true });
  fs.mkdirSync(remoteCacheRoot, { recursive: true });

  function readLocalExperiments() {
    return fs.readdirSync(recordsRoot, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
      .map((entry) => {
        try { return JSON.parse(fs.readFileSync(path.join(recordsRoot, entry.name), 'utf8')); } catch { return null; }
      })
      .filter((record) => record?.schema === 'polymath-ml-experiment-v1');
  }

  async function restoreRemoteExperiments() {
    if (!artifactStore?.remote) return [];
    const keys = (await artifactStore.list(REMOTE_PREFIX)).filter((key) => key.endsWith('.json'));
    const records = [];
    for (const key of keys) {
      try {
        const filename = path.join(remoteCacheRoot, path.basename(key));
        await artifactStore.materialize(key, filename);
        const record = JSON.parse(fs.readFileSync(filename, 'utf8'));
        if (record?.schema === 'polymath-ml-experiment-v1') records.push(record);
      } catch {
        // One partial experiment must not hide the rest of the audit trail.
      }
    }
    return records;
  }

  async function list() {
    let remote = [];
    let storageWarning = '';
    try { remote = await restoreRemoteExperiments(); } catch (error) { storageWarning = error.message || String(error); }
    const records = [...new Map(
      [SEEDED_EXPERIMENT, ...remote, ...readLocalExperiments()]
        .map((record) => [record.id, record]),
    ).values()].sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)));
    return { records, storageWarning };
  }

  async function save(record) {
    const filename = safeRecordPath(recordsRoot, record.id);
    if (!filename) throw new Error('Unsafe experiment identifier.');
    record.updatedAt = new Date().toISOString();
    atomicJson(filename, record);
    if (artifactStore?.remote) {
      try {
        await artifactStore.putFile(`${REMOTE_PREFIX}/${record.id}.json`, filename, 'application/json');
        record.persistence = { remote: true, provider: 'private-s3-compatible', warning: '' };
      } catch (error) {
        // Never encourage a second GPU submission merely because the audit
        // mirror failed after RunPod already accepted the first job.
        record.persistence = {
          remote: false,
          provider: 'local-disk-fallback',
          warning: `Remote audit mirror failed: ${error.message || String(error)}`,
        };
      }
      atomicJson(filename, record);
    }
    return record;
  }

  async function get(id) {
    if (id === SEEDED_EXPERIMENT.id) return structuredClone(SEEDED_EXPERIMENT);
    const { records } = await list();
    return records.find((record) => record.id === id) || null;
  }

  async function createDraft(body, actor = 'administrator') {
    const configuration = parseConfiguration(body);
    const { records } = await list();
    if (records.some((record) => record.version === configuration.version)) {
      throw new Error(`Candidate version already exists: ${configuration.version}. Choose a new version; checkpoints are append-only.`);
    }
    const now = new Date().toISOString();
    const record = {
      id: crypto.randomUUID(),
      schema: 'polymath-ml-experiment-v1',
      name: configuration.name,
      version: configuration.version,
      datasetId: configuration.datasetId,
      instrument: configuration.instrument,
      status: 'draft',
      stage: 'Configuration saved; training has not started',
      readOnly: false,
      createdAt: now,
      updatedAt: now,
      baseCheckpoint: { label: 'models/original', immutable: true },
      candidateCheckpoint: { label: `models/muscriptor-tester/${configuration.version}`, promoted: false },
      configuration,
      weightScope: {
        changed: [`final ${configuration.trainLastLayers} transformer block(s)`, 'output normalization', 'token output head'],
        frozen: ['audio encoder', 'conditioning provider', 'earlier transformer blocks', 'immutable original checkpoint files'],
        exactTensorDeltaAvailable: false,
      },
      data: { datasetId: configuration.datasetId, readiness: 'validated by worker before any optimizer update' },
      metrics: {},
      errorDiff: [],
      decision: { approved: false, commercialUseAllowed: false, summary: 'Research candidate only.' },
      remote: null,
      audit: [{ at: now, actor, action: 'draft-created', detail: 'No weights changed.' }],
    };
    return save(record);
  }

  async function startTraining(id, body, actor = 'administrator') {
    const record = await get(id);
    if (!record || record.readOnly) throw new Error('Editable experiment was not found.');
    if (record.status !== 'draft' && record.status !== 'failed') throw new Error('Only a draft or failed experiment can start training.');
    if (clean(body.confirmVersion) !== record.version) throw new Error(`Type ${record.version} exactly to confirm the new checkpoint.`);
    if (body.rightsAcknowledged !== true) throw new Error('Confirm that you have the right to use this dataset for training.');
    const submitted = await runpod.submitAction({
      action: 'train_piano_candidate',
      dataset_id: record.datasetId,
      version: record.version,
      epochs: record.configuration.epochs,
      train_last_layers: record.configuration.trainLastLayers,
      learning_rate: record.configuration.learningRate,
      timing_token_weight: record.configuration.timingTokenWeight,
      note_off_token_weight: record.configuration.noteOffTokenWeight,
      eos_token_weight: record.configuration.eosTokenWeight,
      rights_acknowledgement: RIGHTS_ACKNOWLEDGEMENT,
    }, { executionTimeout: 60 * 60 * 1000, ttl: 2 * 60 * 60 * 1000 });
    if (!clean(submitted.id)) throw new Error('RunPod did not return a training job ID.');
    record.status = 'training';
    record.stage = 'Queued on RunPod Serverless';
    record.remote = { operation: 'train', jobId: clean(submitted.id), state: clean(submitted.status) || 'IN_QUEUE' };
    record.audit.push({ at: new Date().toISOString(), actor, action: 'training-submitted', detail: `RunPod job ${record.remote.jobId}` });
    return save(record);
  }

  function applyRemoteStatus(record, status) {
    const state = clean(status.status).toUpperCase() || 'UNKNOWN';
    record.remote = { ...record.remote, state, delayTimeMs: Number(status.delayTime) || null };
    record.stage = clean(status.progress) || (state === 'IN_QUEUE' ? 'Waiting for an available GPU worker' : state === 'IN_PROGRESS' ? 'GPU operation in progress' : state);
    if (state === 'COMPLETED') {
      if (record.remote.operation === 'train') {
        record.status = 'trained';
        record.stage = 'Candidate checkpoint saved; frozen evaluation required';
        record.results = { ...(record.results || {}), training: status.output || {} };
        const metadata = status.output?.metadata || {};
        record.baseCheckpoint.sha256 = metadata.baseSha256 || record.baseCheckpoint.sha256;
        record.data = { training: metadata.trainAudit, validation: metadata.validationAudit, datasetId: record.datasetId };
        record.metrics = {
          ...record.metrics,
          validationLoss: {
            baseline: metadata.baselineValidationLoss,
            candidate: metadata.bestValidationLoss,
            preferredDirection: 'down',
          },
        };
        if (metadata.weightDelta) {
          record.weightDelta = metadata.weightDelta;
          record.weightScope = {
            ...record.weightScope,
            exactTensorDeltaAvailable: true,
            note: 'Per-tensor parameter deltas were measured against the immutable base immediately after the best epoch was restored.',
          };
        }
      } else {
        record.status = 'evaluated';
        record.stage = 'Frozen evaluation complete; human review required';
        record.results = { ...(record.results || {}), evaluation: status.output || {} };
        const view = evaluationView(status.output || {});
        record.metrics = { ...record.metrics, ...view.metrics };
        record.errorDiff = view.errorDiff;
      }
      record.audit.push({ at: new Date().toISOString(), actor: 'RunPod worker', action: `${record.remote.operation}-completed`, detail: 'Result captured in the private experiment audit.' });
      record.remote = null;
    } else if (TERMINAL_FAILURES.has(state)) {
      record.status = 'failed';
      record.stage = clean(status.error || status.output?.error) || `RunPod job ${state.toLowerCase()}`;
      record.audit.push({ at: new Date().toISOString(), actor: 'RunPod worker', action: `${record.remote.operation}-${state.toLowerCase()}`, detail: record.stage });
      record.remote = null;
    }
  }

  async function refresh(id) {
    const record = await get(id);
    if (!record) throw new Error('Experiment not found.');
    if (!record.remote?.jobId) return record;
    const status = await runpod.getJobStatus(record.remote.jobId);
    applyRemoteStatus(record, status);
    return save(record);
  }

  async function startEvaluation(id, actor = 'administrator') {
    let record = await get(id);
    if (!record || record.readOnly) throw new Error('Editable experiment was not found.');
    if (record.status !== 'trained' && record.status !== 'evaluated') throw new Error('Train the candidate successfully before evaluation.');
    if (record.remote) throw new Error('This experiment already has a running operation.');
    const submitted = await runpod.submitAction({
      action: 'evaluate_piano_candidate',
      dataset_id: record.datasetId,
      version: record.version,
      instrument: record.instrument,
    }, { executionTimeout: 60 * 60 * 1000, ttl: 2 * 60 * 60 * 1000 });
    if (!clean(submitted.id)) throw new Error('RunPod did not return an evaluation job ID.');
    record.status = 'evaluating';
    record.stage = 'Frozen comparison queued on RunPod Serverless';
    record.remote = { operation: 'evaluate', jobId: clean(submitted.id), state: clean(submitted.status) || 'IN_QUEUE' };
    record.audit.push({ at: new Date().toISOString(), actor, action: 'evaluation-submitted', detail: `RunPod job ${record.remote.jobId}` });
    return save(record);
  }

  async function cancel(id, actor = 'administrator') {
    const record = await get(id);
    if (!record?.remote?.jobId || record.readOnly) throw new Error('No cancellable experiment job was found.');
    await runpod.cancelJob(record.remote.jobId);
    record.audit.push({ at: new Date().toISOString(), actor, action: 'job-cancel-requested', detail: `RunPod job ${record.remote.jobId}` });
    record.status = 'cancelled';
    record.stage = 'Cancellation requested';
    record.remote = null;
    return save(record);
  }

  return {
    cancel,
    createDraft,
    get,
    list,
    refresh,
    save,
    startEvaluation,
    startTraining,
    systemOverview(extra = {}) {
      return {
        schema: 'polymath-ml-operations-overview-v1',
        pipeline: PIPELINE,
        tokenSystem: TOKEN_SYSTEM,
        platforms: [
          { name: 'Administrator browser', role: 'Review, configure, and approve. Secrets never enter the browser.' },
          { name: 'AWS API', role: 'Authenticates admins, archives history, aligns labels, and submits guarded GPU jobs.' },
          { name: 'Private S3-compatible artifact storage', role: 'Persists audit records and supervision artifacts across server restarts.' },
          { name: 'RunPod Serverless', role: 'Loads the 1.3B checkpoint on a GPU, trains candidates, and runs frozen evaluation.' },
          { name: 'RunPod network volume', role: 'Keeps original, versioned tester checkpoints, prepared clips, and full private evaluations.' },
        ],
        safety: {
          originalImmutable: true,
          appendOnlyCandidateVersions: true,
          productionPromotionFromConsole: false,
          serverSideSecretsOnly: true,
          commercialUseAllowed: false,
        },
        runpod: {
          configured: Boolean(runpod?.configured),
          missingConfigurationNames: runpod?.missing || [],
          storageTargetCount: runpod?.storageTargetCount || 0,
        },
        ...extra,
      };
    },
  };
}

module.exports = {
  PIPELINE,
  RIGHTS_ACKNOWLEDGEMENT,
  SEEDED_EXPERIMENT,
  TOKEN_SYSTEM,
  createMlOperations,
  evaluationView,
  parseConfiguration,
};
