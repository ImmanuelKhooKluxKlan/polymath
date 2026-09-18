import assert from 'node:assert/strict';
import test from 'node:test';

import { deriveTrustedAlignmentAnchors } from './deriveTrustedAlignmentAnchors.mjs';

test('derives a cutoff boundary without reading anchors at or after the cutoff', () => {
  const result = deriveTrustedAlignmentAnchors({
    anchors: [
      { referenceTime: 2, observedTime: 4, support: 10, kind: 'automatic', structuralSimilarity: 0.8 },
      { referenceTime: 6, observedTime: 8, support: 10, kind: 'automatic', structuralSimilarity: 0.9 },
      { referenceTime: 10, observedTime: 12, support: 10, kind: 'automatic', structuralSimilarity: 0.7 },
      { referenceTime: 14, observedTime: 99, support: 10, kind: 'bad-tail' },
    ],
  }, { cutoffSeconds: 12, minimumSupport: 2 });

  assert.deepEqual(
    result.anchors.map(({ referenceTime, observedTime }) => ({ referenceTime, observedTime })),
    [
      { referenceTime: 2, observedTime: 4 },
      { referenceTime: 6, observedTime: 8 },
      { referenceTime: 10, observedTime: 12 },
      { referenceTime: 12, observedTime: 14 },
    ],
  );
  assert.equal(result.anchors.at(-1).structuralSimilarity, 0.7);
  assert.equal(result.boundaryObservedTime, 14);
  assert.equal(result.excludedSourceAnchorCount, 1);
});

test('rejects a boundary with insufficient supported evidence', () => {
  assert.throws(
    () => deriveTrustedAlignmentAnchors({
      anchors: [{ referenceTime: 2, observedTime: 4, support: 1 }],
    }, { cutoffSeconds: 12 }),
    /At least two/,
  );
});
