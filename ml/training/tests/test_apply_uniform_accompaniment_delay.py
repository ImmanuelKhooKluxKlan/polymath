import copy
import unittest

from ml.training.apply_uniform_accompaniment_delay import (
    apply_uniform_accompaniment_delay,
)


class ApplyUniformAccompanimentDelayTests(unittest.TestCase):
    def payload(self):
        return {
            "notes": [
                {
                    "time": 1.0,
                    "midi": 48,
                    "velocity": 0.5,
                    "duration": 0.8,
                    "hand": "left",
                    "arrangementRole": "harmony",
                },
                {
                    "time": 1.01,
                    "midi": 72,
                    "velocity": 0.8,
                    "duration": 0.4,
                    "hand": "right",
                    "arrangementRole": "melody",
                },
            ],
            "pedals": [{"time": 0.9, "value": 1.0}],
        }

    def test_moves_only_accompaniment(self):
        before = self.payload()
        result, diagnostics = apply_uniform_accompaniment_delay(
            copy.deepcopy(before), 0.01
        )
        left = next(note for note in result["notes"] if note["hand"] == "left")
        right = next(note for note in result["notes"] if note["hand"] == "right")
        self.assertEqual(left["time"], 1.01)
        self.assertEqual(left["midi"], 48)
        self.assertEqual(right, before["notes"][1])
        self.assertEqual(result["pedals"], before["pedals"])
        self.assertEqual(diagnostics["changedNotes"], 1)

    def test_reapplication_rebases_from_original(self):
        payload, _ = apply_uniform_accompaniment_delay(self.payload(), 0.02)
        payload, _ = apply_uniform_accompaniment_delay(payload, 0.01)
        left = next(note for note in payload["notes"] if note["hand"] == "left")
        self.assertEqual(left["time"], 1.01)
        self.assertEqual(left["originalTimeBeforeUniformAccompanimentDelay"], 1.0)

    def test_rejects_large_delay(self):
        with self.assertRaises(ValueError):
            apply_uniform_accompaniment_delay(self.payload(), 0.081)


if __name__ == "__main__":
    unittest.main()
