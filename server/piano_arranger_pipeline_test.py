import unittest

from piano_arranger_pipeline import (
    DEFAULT_ARRANGER_PROFILE,
    DEFAULT_RECOVERY_PROFILE,
    DEFAULT_REGISTER_PROFILE,
    apply_acoustic_hand_balance,
    load_json,
    run_pipeline,
)


class ProductionPianoPipelineTests(unittest.TestCase):
    def test_acoustic_hand_balance_is_small_and_does_not_change_notes(self):
        payload = {
            "notes": [
                {
                    "midi": 48,
                    "time": 0.0,
                    "duration": 0.5,
                    "velocity": 0.6,
                    "performanceGain": 1.0,
                },
                {
                    "midi": 60,
                    "time": 0.0,
                    "duration": 0.5,
                    "velocity": 0.7,
                    "performanceGain": 1.0,
                },
                {
                    "midi": 72,
                    "time": 0.5,
                    "duration": 0.25,
                    "velocity": 0.8,
                    "performanceGain": 1.0,
                },
            ]
        }
        output, diagnostics = apply_acoustic_hand_balance(payload)

        self.assertEqual(
            [(item["midi"], item["time"], item["duration"], item["velocity"]) for item in output["notes"]],
            [(48, 0.0, 0.5, 0.6), (60, 0.0, 0.5, 0.7), (72, 0.5, 0.25, 0.8)],
        )
        self.assertEqual(output["notes"][0]["performanceGain"], 1.0)
        self.assertEqual(output["notes"][1]["performanceGain"], 0.98)
        self.assertEqual(output["notes"][2]["performanceGain"], 0.95)
        self.assertEqual(diagnostics["rightNotesAdjusted"], 2)
        self.assertTrue(diagnostics["hammerVelocitiesFrozen"])

    def test_preserves_authored_middle_register_without_blanket_shift(self):
        source = {
            "title": "Authored register fixture",
            "instrument": "piano",
            "notes": [
                {
                    "midi": 60,
                    "note": "C4",
                    "time": 0.0,
                    "duration": 0.5,
                    "velocity": 0.7,
                    "instrument": "acoustic_piano",
                    "sourceInstrument": "acoustic_piano",
                }
            ],
        }
        output, diagnostics = run_pipeline(
            source,
            "instrumental",
            load_json(DEFAULT_ARRANGER_PROFILE),
            load_json(DEFAULT_RECOVERY_PROFILE),
            load_json(DEFAULT_REGISTER_PROFILE),
        )

        self.assertEqual(output["notes"][0]["midi"], 60)
        self.assertEqual(output["notes"][0]["note"], "C4")
        self.assertFalse(diagnostics["blanketGlobalRegisterShiftEnabled"])
        self.assertEqual(output["notes"][0]["performanceGain"], 0.98)
        self.assertEqual(diagnostics["acousticHandBalance"]["rightNotesAdjusted"], 1)
        self.assertEqual(output["arrangementProfile"], "polymath-pianella-v003")


if __name__ == "__main__":
    unittest.main()
