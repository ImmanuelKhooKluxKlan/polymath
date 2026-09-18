import copy
import unittest

from piano_mix_balance import adaptive_melody_mix_balance


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


class AdaptiveMelodyMixBalanceTests(unittest.TestCase):
    def test_is_inert_until_enabled(self):
        notes = [note(60, 0.0, "melody"), note(60, 0.0, "harmony")]
        result, diagnostics = adaptive_melody_mix_balance(notes, {})
        self.assertIs(result, notes)
        self.assertFalse(diagnostics["applied"])

    def test_solves_equal_energy_roles_to_requested_share(self):
        notes = [note(60, 0.0, "melody"), note(60, 0.0, "harmony")]
        result, diagnostics = adaptive_melody_mix_balance(
            notes,
            {"enabled": True, "targetEstimatedMelodyShare": 0.60},
        )
        melody = next(item for item in result if item["arrangementRole"] == "melody")
        harmony = next(item for item in result if item["arrangementRole"] == "harmony")
        self.assertTrue(diagnostics["applied"])
        self.assertAlmostEqual(
            diagnostics["estimatedBalancedMelodyShare"], 0.60, places=5
        )
        self.assertEqual(melody["performanceGain"], 1.3)
        self.assertAlmostEqual(harmony["performanceGain"], 1.3 / 1.5, places=5)

    def test_changes_only_performance_gain_and_provenance(self):
        notes = [
            note(72, 0.1, "melody", duration=0.3, velocity=0.8),
            note(48, 0.0, "bass", duration=0.8, velocity=0.6),
            note(55, 2.0, "harmony", duration=0.2, velocity=0.5),
        ]
        before = copy.deepcopy(notes)
        result, diagnostics = adaptive_melody_mix_balance(
            notes, {"enabled": True}
        )
        frozen = ("midi", "time", "duration", "audioDuration", "velocity")
        self.assertEqual(
            [[item[key] for key in frozen] for item in result],
            [[item[key] for key in frozen] for item in before],
        )
        self.assertEqual(result[2]["performanceGain"], 1.0)
        self.assertTrue(diagnostics["pitchesFrozen"])
        self.assertTrue(diagnostics["hammerVelocitiesFrozen"])

    def test_reports_missing_role_without_modifying_notes(self):
        notes = [note(60, 0.0, "harmony")]
        result, diagnostics = adaptive_melody_mix_balance(
            notes, {"enabled": True}
        )
        self.assertIs(result, notes)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "missing-melody-or-accompaniment")


if __name__ == "__main__":
    unittest.main()
