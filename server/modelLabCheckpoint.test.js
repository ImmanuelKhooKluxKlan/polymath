const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { buildModelLabCheckpoints, createModelLab } = require('./modelLab');

test('Model Lab uses the shared live inference checkpoint instead of its legacy setting', () => {
  const dataRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-model-lab-checkpoint-'));
  try {
    const modelLab = createModelLab(
      {
        NODE_ENV: 'test',
        MODEL_LAB_MODEL_VERSION: 'muscriptor-tester/v001',
      },
      {
        dataRoot,
        inferenceVersion: 'phase1-v002',
      },
    );

    assert.equal(modelLab.capability().checkpoint, 'phase1-v002');
    assert.deepEqual(
      modelLab.capability().checkpoints.map((checkpoint) => checkpoint.id),
      ['original', 'phase1-v002'],
    );
    assert.equal(modelLab.capability().checkpoints[0].version, 'v001');
    assert.equal(modelLab.capability().checkpoints[1].version, 'v002');
  } finally {
    fs.rmSync(dataRoot, { recursive: true, force: true });
  }
});

test('Model Lab checkpoint list accepts configured safe candidates and ignores paths', () => {
  const checkpoints = buildModelLabCheckpoints(
    'phase1-v002',
    'phase1-v001, ../../secret, phase2-v003, phase1-v001',
  );
  assert.deepEqual(
    checkpoints.map((checkpoint) => checkpoint.id),
    ['original', 'phase1-v001', 'phase2-v003', 'phase1-v002'],
  );
  assert.equal(checkpoints.some((checkpoint) => checkpoint.id.includes('secret')), false);
});

test('Model Lab normalizes a versioned checkpoint path before sending it to RunPod', () => {
  const dataRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-model-lab-checkpoint-'));
  try {
    const modelLab = createModelLab(
      {
        NODE_ENV: 'test',
        MODEL_LAB_MODEL_VERSION: 'models/muscriptor-tester/phase2-v003',
      },
      { dataRoot },
    );

    assert.equal(modelLab.capability().checkpoint, 'phase2-v003');
  } finally {
    fs.rmSync(dataRoot, { recursive: true, force: true });
  }
});
