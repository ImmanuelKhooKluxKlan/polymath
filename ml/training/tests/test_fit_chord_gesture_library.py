import unittest

from ml.training.fit_chord_gesture_library import (
    aligned_target_left_notes,
    chord_size_metrics,
    harmonic_sequence_context,
    predict_size_from_ranked_neighbors,
)


class ChordGestureLibraryTests(unittest.TestCase):
    def test_explicit_pianist_hand_wins_over_fixed_midi_split(self):
        target = {
            "notes": [
                {"midi": 39, "time": 1.0, "duration": 0.2, "hand": "left"},
                {"midi": 55, "time": 1.0, "duration": 0.2, "hand": "right"},
                {"midi": 63, "time": 1.0, "duration": 0.2, "hand": "left"},
            ]
        }
        report = {
            "anchors": [{"targetTime": 0.0, "sourceTime": 0.0}, {"targetTime": 2.0, "sourceTime": 2.0}],
            "qualityWindows": [{"sourceStart": 0.0, "sourceEnd": 2.0, "referenceStart": 0.0, "referenceEnd": 2.0, "status": "trusted"}],
        }
        notes = aligned_target_left_notes(target, report, 60)
        self.assertEqual([note["midi"] for note in notes], [39, 63])

    def test_sequence_context_is_source_only_and_transposition_invariant(self):
        notes = [
            {"time": 0.0, "midi": 48, "duration": 0.4, "velocity": 0.8, "instrument": "bass"},
            {"time": 0.5, "midi": 52, "duration": 0.4, "velocity": 0.8, "instrument": "guitar"},
            {"time": 1.0, "midi": 55, "duration": 0.4, "velocity": 0.8, "instrument": "guitar"},
        ]
        times = [note["time"] for note in notes]
        context, anchor, _ = harmonic_sequence_context(
            notes, times, 0.5, 0.35, [0.0, 0.5, 1.0]
        )
        shifted = [{**note, "midi": note["midi"] + 2} for note in notes]
        shifted_context, shifted_anchor, _ = harmonic_sequence_context(
            shifted, times, 0.5, 0.35, [0.0, 0.5, 1.0]
        )

        self.assertEqual(len(context), 90)
        self.assertEqual(context, shifted_context)
        self.assertEqual((shifted_anchor - anchor) % 12, 2)

    def test_size_vote_does_not_require_target_size(self):
        ranked = [
            (0.01, {"size": 2, "weight": 1.0}),
            (0.02, {"size": 2, "weight": 1.0}),
            (0.03, {"size": 1, "weight": 1.0}),
        ]

        size, confidence, margin = predict_size_from_ranked_neighbors(
            ranked, neighbors=3, temperature=0.2
        )

        self.assertEqual(size, 2)
        self.assertGreater(confidence, 0.5)
        self.assertGreater(margin, 0.0)

    def test_chord_size_metrics_report_real_error(self):
        result = chord_size_metrics([1, 2, 2], [1, 3, 2])
        self.assertAlmostEqual(result["chordSizeAccuracy"], 2 / 3, places=6)
        self.assertAlmostEqual(result["chordSizeMae"], 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
