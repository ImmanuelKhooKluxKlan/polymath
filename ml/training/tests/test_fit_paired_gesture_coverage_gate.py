import unittest

from ml.training.fit_paired_gesture_coverage_gate import choose_policy


class PairedGestureCoverageGateTests(unittest.TestCase):
    def test_prefers_safe_policy_over_higher_unsafe_score(self):
        unsafe = {
            "aggregate": {"precision": 0.95, "f0_5": 0.9, "recall": 0.9, "predictedPositive": 100},
            "worstFoldPrecision": 0.4,
            "minimumFoldPredictions": 10,
        }
        safe = {
            "aggregate": {"precision": 0.82, "f0_5": 0.7, "recall": 0.5, "predictedPositive": 50},
            "worstFoldPrecision": 0.72,
            "minimumFoldPredictions": 3,
        }
        self.assertIs(choose_policy([unsafe, safe]), safe)


if __name__ == "__main__":
    unittest.main()
