import unittest

from ml.training.apply_paired_chord_gesture_mapper import (
    apply_mapper,
    tight_left_groups,
)


class ApplyPairedChordGestureMapperTests(unittest.TestCase):
    def test_left_groups_exclude_right_hand_and_melody(self):
        payload = {
            "notes": [
                {"midi": 48, "time": 1.0, "hand": "left", "arrangementRole": "harmony"},
                {"midi": 60, "time": 1.0, "hand": "right", "arrangementRole": "harmony"},
                {"midi": 55, "time": 1.0, "hand": "left", "arrangementRole": "melody"},
            ]
        }
        self.assertEqual([[note["midi"] for note in group] for group in tight_left_groups(payload)], [[48]])

    def test_rejects_wrong_feature_contract(self):
        candidate = {
            "notes": [
                {"midi": 48, "time": 1.0, "hand": "left", "arrangementRole": "harmony"}
            ]
        }
        source = {
            "notes": [
                {"midi": 48, "time": 1.0, "duration": 0.2, "velocity": 0.7, "instrument": "guitar"}
            ]
        }
        mapper = {
            "type": "paired-incumbent-sequence-chord-mapper-v1",
            "samples": [{"context": [0.0], "targetIntervals": [0]}],
        }
        with self.assertRaisesRegex(ValueError, "context does not match"):
            apply_mapper(candidate, source, mapper)

    def test_excluded_song_must_leave_eligible_training_samples(self):
        with self.assertRaisesRegex(ValueError, "no eligible samples"):
            apply_mapper(
                {"notes": []},
                {"notes": []},
                {
                    "type": "paired-incumbent-sequence-chord-mapper-v1",
                    "samples": [{"songId": "kiss-me", "context": [0.0]}],
                },
                exclude_song_id="kiss-me",
            )


if __name__ == "__main__":
    unittest.main()
