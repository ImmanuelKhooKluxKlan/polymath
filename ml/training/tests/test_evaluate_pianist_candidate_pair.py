import unittest

from ml.training.evaluate_pianist_candidate_pair import evaluate_pair


class EvaluatePianistCandidatePairTests(unittest.TestCase):
    def test_kiss_me_bad_tail_cannot_change_prefix_accuracy(self):
        # Pianella time 150 maps to original-video time 160 here. Cutting the
        # original video at 150 instead would wrongly discard good notes.
        alignment = {
            "anchors": [
                {"referenceTime": 0.0, "observedTime": 10.0},
                {"referenceTime": 200.0, "observedTime": 210.0},
            ]
        }
        reference = {"notes": [
            {"time": 149.0, "midi": 60, "duration": 0.5, "velocity": 0.7},
            {"time": 150.0, "midi": 62, "duration": 0.5, "velocity": 0.7},
            {"time": 170.0, "midi": 64, "duration": 0.5, "velocity": 0.7},
        ]}
        prefix = {"time": 159.0, "midi": 60, "duration": 0.5, "velocity": 0.7}
        baseline = {"notes": [prefix, {"time": 180.0, "midi": 64, "duration": 0.5}]}
        candidate = {"notes": [prefix, {"time": 180.0, "midi": 37, "duration": 4.0}]}
        result = evaluate_pair(
            reference, alignment, baseline, candidate,
            reference_end_seconds=150.0, candidate_end_seconds=160.0,
        )
        self.assertEqual(result["referenceNotes"], 1)
        self.assertEqual(result["candidate"]["exactF1_100ms"], 1.0)
        self.assertTrue(all(value == 0.0 for value in result["deltas"].values()))

    def test_prefix_mistakes_still_reduce_accuracy_with_tail_excluded(self):
        reference = {"notes": [
            {"time": 149.0, "midi": 60, "duration": 0.5, "velocity": 0.7},
            {"time": 170.0, "midi": 64, "duration": 0.5, "velocity": 0.7},
        ]}
        baseline = {"notes": [reference["notes"][0]]}
        candidate = {"notes": [{**reference["notes"][0], "midi": 61}]}
        result = evaluate_pair(
            reference, {}, baseline, candidate,
            reference_already_aligned=True,
            reference_end_seconds=150.0, candidate_end_seconds=150.0,
        )
        self.assertEqual(result["referenceNotes"], 1)
        self.assertEqual(result["deltas"]["exactF1_100ms"], -1.0)
        self.assertIsNone(result["deltas"]["velocityMeanAbsoluteError"])

    def test_unrestricted_other_song_keeps_its_later_notes(self):
        reference = {"notes": [{"time": 185.0, "midi": 60, "duration": 0.5, "velocity": 0.7}]}
        result = evaluate_pair(
            reference, {}, reference, reference, reference_already_aligned=True,
        )
        self.assertEqual(result["referenceNotes"], 1)
        self.assertEqual(result["candidate"]["exactF1_100ms"], 1.0)

    def test_only_rejected_tail_is_not_a_valid_accuracy_test(self):
        reference = {"notes": [{"time": 150.0, "midi": 60, "duration": 0.5}]}
        with self.assertRaisesRegex(ValueError, "No trusted reference notes"):
            evaluate_pair(
                reference, {}, reference, reference,
                reference_already_aligned=True,
                reference_end_seconds=150.0, candidate_end_seconds=150.0,
            )

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
