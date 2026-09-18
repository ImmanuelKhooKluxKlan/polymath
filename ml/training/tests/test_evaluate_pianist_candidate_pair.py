import unittest

from ml.training.evaluate_pianist_candidate_pair import evaluate_pair


class EvaluatePianistCandidatePairTests(unittest.TestCase):
    def test_identical_candidate_has_zero_metric_deltas(self):
        reference = {"notes": [{"time": 0.0, "midi": 60, "duration": 0.5, "velocity": 0.7}]}
        alignment = {
            "anchors": [{"sourceSeconds": 0.0, "targetSeconds": 0.0}],
            "trustedSourceRanges": [[0.0, 1.0]],
        }
        candidate = {"notes": [{"time": 0.0, "midi": 60, "duration": 0.5, "velocity": 0.7}]}
        result = evaluate_pair(
            reference,
            alignment,
            candidate,
            candidate,
            reference_already_aligned=True,
        )
        self.assertTrue(all(value == 0.0 for value in result["deltas"].values()))


if __name__ == "__main__":
    unittest.main()
