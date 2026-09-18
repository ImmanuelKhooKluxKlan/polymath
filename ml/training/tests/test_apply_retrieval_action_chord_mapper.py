import math
import unittest

from ml.training.apply_retrieval_action_chord_mapper import gate_probability
from ml.training.fit_retrieval_action_abstention_gate import proposal_feature_names


class ApplyRetrievalActionChordMapperTests(unittest.TestCase):
    def test_zero_logit_has_half_probability(self):
        context_size = 4
        names = proposal_feature_names(context_size)
        features = [0.0] * len(names)
        gate = {
            "featureNames": names,
            "weights": [0.0] * len(names),
            "means": [0.0] * len(names),
            "scales": [1.0] * len(names),
        }
        self.assertTrue(math.isclose(gate_probability(features, gate), 0.5))

    def test_rejects_feature_contract_mismatch(self):
        with self.assertRaises(ValueError):
            gate_probability(
                [0.0] * 18,
                {
                    "featureNames": ["wrong"] * 18,
                    "weights": [0.0] * 18,
                    "means": [0.0] * 18,
                    "scales": [1.0] * 18,
                },
            )


if __name__ == "__main__":
    unittest.main()
