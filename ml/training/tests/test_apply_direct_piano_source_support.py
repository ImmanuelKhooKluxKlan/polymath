import unittest

from ml.training.apply_direct_piano_source_support import apply_source_support


class ApplyDirectPianoSourceSupportTests(unittest.TestCase):
    def test_removes_unsupported_pitch_classes_and_preserves_the_attack(self) -> None:
        candidate = {
            "notes": [
                {"midi": 48, "time": 1.0},
                {"midi": 60, "time": 1.0},
                {"midi": 64, "time": 1.0},
                {"midi": 67, "time": 1.0},
                {"midi": 71, "time": 1.0},
                {"midi": 72, "time": 2.0},
            ]
        }
        source = {
            "notes": [
                {"midi": 48, "time": 1.02},
                {"midi": 64, "time": 1.02},
            ]
        }
        output, diagnostics = apply_source_support(candidate, source)
        self.assertEqual([note["midi"] for note in output["notes"]], [48, 64, 72])
        self.assertEqual(diagnostics["fallbackGroups"], 1)
        self.assertEqual(diagnostics["unsupportedNotesRemoved"], 3)
        self.assertEqual(diagnostics["redundantOctaveLayersRemoved"], 1)

    def test_collapses_redundant_octave_layers_using_source_register(self) -> None:
        candidate = {"notes": [{"midi": 48, "time": 0.0}, {"midi": 60, "time": 0.0}]}
        source = {"notes": [{"midi": 61, "time": 0.01}, {"midi": 60, "time": 0.01}]}
        output, diagnostics = apply_source_support(candidate, source)
        self.assertEqual([note["midi"] for note in output["notes"]], [60])
        self.assertEqual(diagnostics["redundantOctaveLayersRemoved"], 1)


if __name__ == "__main__":
    unittest.main()
