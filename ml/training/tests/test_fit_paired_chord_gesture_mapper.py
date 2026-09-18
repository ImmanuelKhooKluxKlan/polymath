import unittest

from ml.training.fit_paired_chord_gesture_mapper import (
    canonical_manifest_pairs,
    candidate_left_notes,
    decode_conservative_correction,
    match_group_sequences,
    paired_context,
)


class PairedChordGestureMapperTests(unittest.TestCase):
    def test_structural_manifest_is_normalized_without_copying_paths(self):
        rows = canonical_manifest_pairs(
            {
                "songs": [
                    {
                        "id": "example",
                        "source": "source.json",
                        "reference": "reference.json",
                        "candidate": "candidate.json",
                        "alignment": "alignment.json",
                    }
                ]
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], "reference.json")
        self.assertEqual(rows[0]["alignmentReport"], "alignment.json")

    def test_candidate_explicit_hand_wins_over_midi_threshold(self):
        payload = {
            "notes": [
                {"time": 1.0, "midi": 70, "duration": 0.2, "hand": "left"},
                {"time": 1.0, "midi": 55, "duration": 0.2, "hand": "right"},
            ]
        }
        notes = candidate_left_notes(
            payload, [{"sourceStart": 0.0, "sourceEnd": 2.0}], None
        )
        self.assertEqual([note["midi"] for note in notes], [70])

    def test_time_matching_is_one_to_one_and_does_not_use_pitch(self):
        candidates = [
            [{"time": 1.00, "midi": 48}],
            [{"time": 1.10, "midi": 50}],
        ]
        targets = [
            [{"time": 1.02, "midi": 71}],
            [{"time": 1.12, "midi": 36}],
        ]
        self.assertEqual(match_group_sequences(candidates, targets, 0.05), [(0, 0), (1, 1)])

    def test_context_contains_incumbent_and_sequence_without_target(self):
        source = [
            {"time": 0.0, "midi": 48, "duration": 0.4, "velocity": 0.8, "instrument": "bass"},
            {"time": 0.5, "midi": 52, "duration": 0.4, "velocity": 0.8, "instrument": "guitar"},
            {"time": 1.0, "midi": 55, "duration": 0.4, "velocity": 0.8, "instrument": "guitar"},
        ]
        groups = [
            [{"time": 0.0, "midi": 48, "velocity": 0.5, "duration": 0.3}],
            [{"time": 0.5, "midi": 52, "velocity": 0.6, "duration": 0.4}],
            [{"time": 1.0, "midi": 55, "velocity": 0.7, "duration": 0.5}],
        ]
        context, _anchor, incumbent = paired_context(
            source, [note["time"] for note in source], groups, 1, 0.35
        )
        self.assertEqual(len(context), 83)
        self.assertEqual(len(incumbent), 1)

    def test_correction_falls_back_when_vote_gain_is_too_small(self):
        votes = [0.0] * 12
        votes[0] = 0.60
        votes[4] = 0.62
        predicted, gain = decode_conservative_correction(
            votes, {0}, incumbent_prior=0.0, minimum_gain=0.05
        )
        self.assertEqual(predicted, {0})
        self.assertAlmostEqual(gain, 0.02)


if __name__ == "__main__":
    unittest.main()
