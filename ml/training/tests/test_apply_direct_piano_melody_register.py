import unittest

from ml.training.apply_direct_piano_melody_register import apply_melody_register


class ApplyDirectPianoMelodyRegisterTests(unittest.TestCase):
    def test_shifts_only_the_highest_eligible_note_in_a_low_register_score(self) -> None:
        payload = {
            "notes": [
                {"midi": 48, "time": 1.0, "note": "C3"},
                {"midi": 64, "time": 1.01, "note": "E4"},
                {"midi": 62, "time": 2.0, "note": "D4"},
            ]
        }
        output, diagnostics = apply_melody_register(payload, minimum_midi=64)
        self.assertEqual([note["midi"] for note in output["notes"]], [48, 76, 62])
        self.assertEqual(diagnostics["shiftedNotes"], 1)
        self.assertEqual(payload["notes"][1]["midi"], 64)

    def test_does_not_activate_when_the_score_already_reaches_upper_register(self) -> None:
        payload = {"notes": [{"midi": 84, "time": 0.0, "note": "C6"}]}
        output, diagnostics = apply_melody_register(payload)
        self.assertEqual(output["notes"][0]["midi"], 84)
        self.assertFalse(diagnostics["applied"])


if __name__ == "__main__":
    unittest.main()
