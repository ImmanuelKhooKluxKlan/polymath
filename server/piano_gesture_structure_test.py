import unittest

from piano_gesture_structure import (
    GESTURE_NOTE_FEATURE_NAMES,
    gesture_groups,
    gesture_note_feature_rows,
    prune_gesture_notes,
)


class PianoGestureStructureTests(unittest.TestCase):
    def test_features_use_candidate_only_continuity_evidence(self):
        notes = [
            {"time": 0.0, "midi": 60, "duration": 0.2, "arrangementRole": "harmony"},
            {"time": 1.0, "midi": 60, "duration": 0.2, "arrangementRole": "harmony"},
            {"time": 1.0, "midi": 64, "duration": 0.2, "arrangementRole": "melody"},
            {"time": 2.0, "midi": 67, "duration": 0.2, "arrangementRole": "harmony"},
        ]
        groups = gesture_groups(notes)
        rows = gesture_note_feature_rows(groups)

        self.assertEqual(set(rows[1][0]), set(GESTURE_NOTE_FEATURE_NAMES))
        self.assertEqual(rows[1][0]["same_midi_previous"], 1.0)
        self.assertEqual(rows[1][1]["is_melody"], 1.0)

    def test_features_expose_inference_safe_cyclic_texture_phase(self):
        notes = [
            {
                "time": 0.0,
                "midi": 55,
                "duration": 0.2,
                "arrangementRole": "harmony",
                "pianistTexturePhase": 3,
                "pianistTextureAction": "compact-preserve",
                "generatedBy": "source-supported-cyclic-harmony-v1",
            }
        ]

        features = gesture_note_feature_rows(gesture_groups(notes))[0][0]

        self.assertEqual(features["is_cyclic_texture_note"], 1.0)
        self.assertEqual(features["texture_phase_3"], 1.0)
        self.assertEqual(features["texture_phase_0"], 0.0)
        self.assertEqual(features["texture_action_preserve"], 1.0)

    def test_pruner_preserves_onset_and_protected_melody(self):
        notes = [
            {"time": 0.0, "midi": 60, "duration": 0.2, "arrangementRole": "harmony"},
            {"time": 0.0, "midi": 64, "duration": 0.2, "arrangementRole": "melody"},
        ]
        model = {
            "featureNames": ["bias", "is_melody"],
            "weights": [-10.0, 20.0],
            "means": [0.0, 0.0],
            "scales": [1.0, 1.0],
        }
        output, report = prune_gesture_notes(
            notes,
            {
                "enabled": True,
                "selectionModel": model,
                "threshold": 0.5,
                "minimumProbabilityGap": 0.1,
                "maximumRemovalsPerGesture": 1,
                "minimumRemainingNotes": 1,
                "preserveMelody": True,
            },
        )

        self.assertEqual([note["midi"] for note in output], [64])
        self.assertEqual(report["removedNotes"], 1)
        self.assertTrue(report["onsetsPreserved"])


if __name__ == "__main__":
    unittest.main()
