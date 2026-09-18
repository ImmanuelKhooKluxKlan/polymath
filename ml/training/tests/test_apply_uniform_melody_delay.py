import copy
import unittest

from ml.training.apply_uniform_melody_delay import apply_uniform_melody_delay


class ApplyUniformMelodyDelayTests(unittest.TestCase):
    def payload(self):
        return {
            "notes": [
                {
                    "time": 1.0,
                    "midi": 72,
                    "velocity": 0.8,
                    "duration": 0.4,
                    "arrangementRole": "melody",
                    "adaptiveMelodyRegisterShiftSemitones": 12,
                },
                {
                    "time": 1.01,
                    "midi": 48,
                    "velocity": 0.5,
                    "duration": 0.8,
                    "arrangementRole": "harmony",
                },
            ],
            "pianoArrangement": {
                "adaptiveMelodyRegisterSeparation": {"applied": True}
            },
        }

    def test_moves_only_gated_shifted_melody_and_freezes_other_fields(self):
        before = self.payload()
        result, diagnostics = apply_uniform_melody_delay(copy.deepcopy(before), 0.03)
        melody = next(note for note in result["notes"] if note["arrangementRole"] == "melody")
        harmony = next(note for note in result["notes"] if note["arrangementRole"] == "harmony")
        self.assertEqual(melody["time"], 1.03)
        self.assertEqual(melody["midi"], 72)
        self.assertEqual(melody["velocity"], 0.8)
        self.assertEqual(melody["duration"], 0.4)
        self.assertEqual(harmony, before["notes"][1])
        self.assertEqual(diagnostics["changedNotes"], 1)

    def test_reapplication_rebases_from_original_instead_of_stacking(self):
        payload, _ = apply_uniform_melody_delay(self.payload(), 0.03)
        payload, _ = apply_uniform_melody_delay(payload, 0.02)
        melody = next(note for note in payload["notes"] if note["arrangementRole"] == "melody")
        self.assertEqual(melody["time"], 1.02)
        self.assertEqual(melody["originalTimeBeforeAdaptiveMelodyDelay"], 1.0)

    def test_does_nothing_when_adaptive_gate_did_not_activate(self):
        payload = self.payload()
        payload["pianoArrangement"]["adaptiveMelodyRegisterSeparation"]["applied"] = False
        result, diagnostics = apply_uniform_melody_delay(payload, 0.03)
        self.assertEqual(result["notes"][0]["time"], 1.0)
        self.assertEqual(diagnostics["changedNotes"], 0)

    def test_rejects_large_delay(self):
        with self.assertRaises(ValueError):
            apply_uniform_melody_delay(self.payload(), 0.081)


if __name__ == "__main__":
    unittest.main()
