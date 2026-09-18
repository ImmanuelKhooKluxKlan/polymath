import unittest

from ml.training.fit_retrieval_action_chord_mapper import decode_retrieval_action


def sample(target):
    return {"targetIntervals": sorted(target), "weight": 1.0}


class RetrievalActionChordMapperTests(unittest.TestCase):
    def test_can_reduce_one_supported_tone_when_cross_song_actions_agree(self):
        ranked = [(0.01, sample({0})), (0.02, sample({0})), (0.03, sample({0, 7}))]
        predicted, _gain, diagnostics = decode_retrieval_action(
            ranked,
            {0, 7},
            {0, 7},
            neighbors=3,
            temperature=0.2,
            incumbent_prior=0.0,
            minimum_gain=0.0,
            maximum_size_change=1,
        )
        self.assertEqual(predicted, {0})
        self.assertEqual(diagnostics["reason"], "accepted")

    def test_never_invents_an_unsupported_pitch_class(self):
        ranked = [(0.01, sample({0, 4})), (0.02, sample({0, 4}))]
        predicted, _gain, _diagnostics = decode_retrieval_action(
            ranked,
            {0},
            {0, 7},
            neighbors=2,
            temperature=0.2,
            incumbent_prior=0.0,
            minimum_gain=0.0,
            maximum_size_change=1,
        )
        self.assertNotIn(4, predicted)
        self.assertTrue(predicted.issubset({0, 7}))

    def test_incumbent_prior_can_keep_the_existing_gesture(self):
        ranked = [(0.01, sample({0})), (0.02, sample({0}))]
        predicted, _gain, diagnostics = decode_retrieval_action(
            ranked,
            {0, 7},
            {0, 7},
            neighbors=2,
            temperature=0.2,
            incumbent_prior=0.5,
            minimum_gain=0.0,
            maximum_size_change=1,
        )
        self.assertEqual(predicted, {0, 7})
        self.assertEqual(diagnostics["reason"], "incumbent-or-insufficient-gain")

    def test_size_change_can_be_frozen(self):
        ranked = [(0.01, sample({0})), (0.02, sample({0}))]
        predicted, _gain, _diagnostics = decode_retrieval_action(
            ranked,
            {0, 7},
            {0, 7},
            neighbors=2,
            temperature=0.2,
            incumbent_prior=0.0,
            minimum_gain=0.0,
            maximum_size_change=0,
        )
        self.assertEqual(len(predicted), 2)


if __name__ == "__main__":
    unittest.main()
