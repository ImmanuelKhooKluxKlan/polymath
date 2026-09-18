import unittest

import numpy as np

from ml.training.fit_pianist_texture_decoder import (
    FEATURE_NAMES,
    classification_metrics,
    feature_vector,
    fit_logistic,
    sigmoid,
    standardize,
)


class PianistTextureDecoderTests(unittest.TestCase):
    def test_feature_contract_uses_candidate_and_source_only(self):
        row = {
            "songId": "must-not-be-a-feature",
            "sourceTime": 999.0,
            "targetGestureCount": 3,
            "referencePitchClasses": [[0], [4]],
            "candidateChordSize": 4,
            "candidatePitchClassCount": 3,
            "candidateDuration": 0.4,
            "candidateVelocity": 0.7,
            "candidateSourceVelocity": 0.6,
            "candidateMeanMidiCentered": 0.2,
            "candidatePitchSpan": 19,
            "candidateLeftShare": 0.5,
            "candidateMelodyShare": 0.25,
            "candidateBassShare": 0.25,
            "previousGapSeconds": 0.3,
            "nextGapSeconds": 0.45,
            "sourceNoteCount": 8,
            "sourceAttackNoteCount": 4,
            "sourcePitchClassCount": 4,
            "sourceAttackPitchClassCount": 3,
            "sourceMedianDuration": 0.5,
            "sourceAttackMedianDuration": 0.3,
            "sourceVoiceShare": 0.2,
            "sourceGuitarShare": 0.7,
            "sourceBassShare": 0.1,
            "candidateRoleSignature": "bass+harmony+melody",
        }

        vector = feature_vector(row)

        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(all(np.isfinite(vector)))
        self.assertNotIn("song_id", FEATURE_NAMES)
        self.assertNotIn("source_time", FEATURE_NAMES)
        self.assertFalse(any("reference" in name for name in FEATURE_NAMES))

    def test_logistic_fit_learns_a_separable_gate(self):
        matrix = np.asarray(
            [[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]],
            dtype=float,
        )
        labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
        standardized, _means, _scales = standardize(matrix)
        weights = fit_logistic(
            standardized, labels, ridge=0.1, positive_weight=1.0
        )
        probabilities = sigmoid(standardized @ weights)

        self.assertLess(probabilities[1], 0.5)
        self.assertGreater(probabilities[2], 0.5)

    def test_metrics_weight_precision_separately_from_recall(self):
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        probabilities = np.asarray([0.9, 0.4, 0.8, 0.1])

        metrics = classification_metrics(labels, probabilities, 0.5)

        self.assertEqual(metrics["truePositive"], 1)
        self.assertEqual(metrics["falsePositive"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
