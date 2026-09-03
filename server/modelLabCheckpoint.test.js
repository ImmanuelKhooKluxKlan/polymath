const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { createModelLab } = require('./modelLab');

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
  } finally {
    fs.rmSync(dataRoot, { recursive: true, force: true });
  }
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
