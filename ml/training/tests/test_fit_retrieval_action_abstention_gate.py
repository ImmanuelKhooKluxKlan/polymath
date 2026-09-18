import unittest

from ml.training.fit_retrieval_action_abstention_gate import (
    BASE_FEATURE_NAMES,
    proposal_feature_names,
    proposal_features,
)


class RetrievalActionAbstentionGateTests(unittest.TestCase):
    def test_feature_contract_matches_vector(self):
        sample = {
            "context": [0.5] * 12,
            "incumbentIntervals": [0, 7],
        }
        ranked = [
            (0.2, {"targetIntervals": [0], "weight": 1.0}),
            (0.4, {"targetIntervals": [0, 7], "weight": 1.0}),
        ]
        vector = proposal_features(
            sample,
            ranked,
            {0, 7},
            {0},
            0.15,
            {
                "actions": 3,
                "incumbentExpectedF1": 0.6,
                "bestExpectedF1": 0.8,
            },
        )
        self.assertEqual(len(vector), len(BASE_FEATURE_NAMES) + 12)
        self.assertEqual(len(vector), len(proposal_feature_names(12)))
        self.assertEqual(vector[0], 1.0)
        self.assertEqual(vector[9], 1)


if __name__ == "__main__":
    unittest.main()
