import unittest

from ml.training.fit_authored_piano_style import fit


class AuthoredPianoStyleTests(unittest.TestCase):
    def test_explicit_hand_labels_define_pianist_bands_without_blanket_shift(self):
        base = {"id": "base", "decoder": {}}
        reference = {
            "notes": [
                {"midi": 34, "time": 0.0, "duration": 0.2, "hand": "left"},
                {"midi": 58, "time": 0.5, "duration": 0.2, "hand": "left"},
                {"midi": 60, "time": 0.0, "duration": 0.2, "hand": "right"},
                {"midi": 84, "time": 0.5, "duration": 0.2, "hand": "right"},
            ]
        }

        profile = fit(base, reference, profile_id="test", density=1.0)
        style = profile["decoder"]["authoredTwoHandStyle"]

        self.assertEqual(style["lower"]["minimumMidi"], 34)
        self.assertEqual(style["lower"]["maximumMidi"], 58)
        self.assertEqual(style["upper"]["minimumMidi"], 60)
        self.assertEqual(style["upper"]["maximumMidi"], 84)
        self.assertEqual(
            style["preferredOctaveShiftsSemitones"],
            {
                "melodyUpper": 0,
                "bassLower": 0,
                "harmonyUpper": 0,
                "harmonyLower": 0,
            },
        )
        self.assertEqual(
            style["harmonyHandAssignment"], "preserve-native-register"
        )
        self.assertEqual(
            style["referenceEvidence"]["staffSplit"]["method"],
            "explicit-reference-hand-labels",
        )

    def test_reference_register_transpose_moves_bands_without_changing_notes(self):
        base = {"id": "base", "decoder": {}}
        reference = {
            "notes": [
                {"midi": 40, "time": 0.0, "duration": 0.2, "sourceTrack": 0},
                {"midi": 44, "time": 0.5, "duration": 0.2, "sourceTrack": 0},
                {"midi": 64, "time": 0.0, "duration": 0.2, "sourceTrack": 1},
                {"midi": 67, "time": 0.5, "duration": 0.2, "sourceTrack": 1},
            ]
        }
        profile = fit(
            base,
            reference,
            profile_id="test",
            density=1.0,
            reference_transpose_semitones=12,
        )
        style = profile["decoder"]["authoredTwoHandStyle"]
        self.assertEqual(style["lower"]["minimumMidi"], 52)
        self.assertEqual(style["upper"]["maximumMidi"], 79)
        self.assertEqual(style["referenceEvidence"]["registerTransposeSemitones"], 12)

    def test_voicing_experiment_can_preserve_base_density(self):
        base = {
            "id": "base",
            "decoder": {
                "adaptiveSourceDensity": {
                    "lowDensityMultiplier": 1.8,
                    "highDensityMultiplier": 1.75,
                    "longSourceDurationWeight": 0.92,
                }
            },
        }
        reference = {
            "notes": [
                {"midi": 40, "time": 0.0, "duration": 0.2, "sourceTrack": 0},
                {"midi": 44, "time": 0.5, "duration": 0.2, "sourceTrack": 0},
                {"midi": 64, "time": 0.0, "duration": 0.2, "sourceTrack": 1},
                {"midi": 67, "time": 0.5, "duration": 0.2, "sourceTrack": 1},
            ]
        }
        profile = fit(
            base,
            reference,
            profile_id="test",
            density=1.0,
            preserve_base_density=True,
        )
        self.assertEqual(
            profile["decoder"]["adaptiveSourceDensity"],
            base["decoder"]["adaptiveSourceDensity"],
        )


if __name__ == "__main__":
    unittest.main()
