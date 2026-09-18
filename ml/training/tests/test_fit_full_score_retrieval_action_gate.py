import unittest

from ml.training.fit_full_score_retrieval_action_gate import replacement_f1_delta


class FullScoreRetrievalActionGateTests(unittest.TestCase):
    def test_replacement_delta_rewards_correct_pitch(self):
        predictions = [{0, 7}, {2}]
        targets = [{0, 4}, {2}]
        self.assertGreater(
            replacement_f1_delta(predictions, targets, 0, {0, 4}),
            0.0,
        )

    def test_replacement_delta_rewards_reducing_unmatched_extra(self):
        predictions = [{0, 7}, {2}]
        targets = [set(), {2}]
        self.assertGreater(
            replacement_f1_delta(predictions, targets, 0, {0}),
            0.0,
        )

    def test_replacement_delta_rejects_lost_true_positive(self):
        predictions = [{0, 7}, {2}]
        targets = [{0, 7}, {2}]
        self.assertLess(
            replacement_f1_delta(predictions, targets, 0, {0}),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
