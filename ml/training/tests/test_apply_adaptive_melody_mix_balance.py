import copy
import unittest

from ml.training.apply_adaptive_melody_mix_balance import apply_balance


def note(midi, time, role, *, duration=0.4, velocity=0.72):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "audioDuration": duration,
        "velocity": velocity,
        "arrangementRole": role,
        "performanceGain": 1.0,
    }


class ApplyAdaptiveMelodyMixBalanceTests(unittest.TestCase):
    def test_freezes_score_and_records_diagnostics(self):
        payload = {
            "notes": [
                note(72, 0.0, "melody", velocity=0.82),
                note(48, 0.0, "bass", duration=0.8, velocity=0.58),
                note(55, 0.0, "harmony", duration=0.8, velocity=0.58),
            ],
            "pianoArrangement": {},
        }
        before = copy.deepcopy(payload)
        result, diagnostics = apply_balance(payload)
        frozen = ("midi", "time", "duration", "audioDuration", "velocity")
        self.assertEqual(
            [[item[key] for key in frozen] for item in result["notes"]],
            [[item[key] for key in frozen] for item in before["notes"]],
        )
        self.assertTrue(diagnostics["applied"])
        self.assertFalse(diagnostics["referenceAnswersUsed"])
        self.assertEqual(
            result["pianoArrangement"]["adaptiveMelodyMixBalance"],
            diagnostics,
        )

    def test_rejects_payload_without_note_list(self):
        with self.assertRaises(ValueError):
            apply_balance({})


if __name__ == "__main__":
    unittest.main()
