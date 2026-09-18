import unittest

from ml.training.evaluate_pianist_candidate_manifest import error_ceiling, metric_gates


class PianistCandidateManifestAuditTests(unittest.TestCase):
    def test_metric_gates_accept_rounding_noise(self):
        gates = metric_gates(
            {
                "quality": -0.0005,
                "exactF1_100ms": 0.0,
                "exactF1_250ms": -0.0001,
                "pitchClassF1_250ms": 0.01,
                "pitchClassRecall_250ms": -0.0005,
            }
        )
        self.assertTrue(all(gates.values()))

    def test_metric_gates_reject_material_regression(self):
        gates = metric_gates(
            {
                "quality": 0.1,
                "exactF1_100ms": -0.0006,
                "exactF1_250ms": 0.1,
                "pitchClassF1_250ms": 0.1,
                "pitchClassRecall_250ms": 0.1,
            }
        )
        self.assertFalse(gates["exactF1At100msNotWorse"])

    def test_error_ceiling_separates_register_and_timing_gaps(self):
        diagnostic = error_ceiling(
            {
                "notes": {
                    "exactPitchOnset100ms": {"f1": 0.30},
                    "pitchClassOnset100ms": {"f1": 0.55},
                    "pitchClassOnset250ms": {"f1": 0.65},
                }
            }
        )
        self.assertEqual(diagnostic["registerPlacementGapAt100ms"], 0.25)
        self.assertEqual(diagnostic["coarseTimingGap100To250ms"], 0.10)
        self.assertEqual(
            diagnostic["remainingPitchClassOrEventGapAt250ms"], 0.35
        )


if __name__ == "__main__":
    unittest.main()
