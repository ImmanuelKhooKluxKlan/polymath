import unittest

from ml.training.train_pianist_phase_suppression import (
    apply_profile,
    learn_phase_slots,
    long_voice_gaps,
    phase_slot,
)


def group(time, state):
    notes = []
    if state in {"left-only", "both"}:
        notes.append({"time": time, "midi": 48, "hand": "left", "velocity": 0.8, "duration": 0.2})
    if state in {"right-only", "both"}:
        notes.append({"time": time, "midi": 72, "hand": "right", "velocity": 0.8, "duration": 0.2})
    return notes


class PianistPhaseSuppressionTest(unittest.TestCase):
    def test_phase_slot_wraps(self):
        self.assertEqual(phase_slot(1.0 + 13 * 0.1, 1.0, 0.1, 16), 13)
        self.assertEqual(phase_slot(1.0 + 29 * 0.1, 1.0, 0.1, 16), 13)

    def test_learns_only_reliable_slot(self):
        candidate = []
        reference = []
        for cycle in range(4):
            candidate.extend([group(cycle * 1.6 + 0.1, "both"), group(cycle * 1.6 + 1.3, "both")])
            reference.extend([group(cycle * 1.6 + 0.1, "both"), group(cycle * 1.6 + 1.3, "left-only")])
        slots, _ = learn_phase_slots(
            reference,
            candidate,
            origin=0.0,
            pulse=0.1,
            subdivisions=16,
            training_start=0.0,
            training_end=7.0,
        )
        self.assertEqual(slots, [13])

    def test_finds_long_voice_gap(self):
        payload = {
            "notes": [
                {"time": 1.0, "midi": 60, "instrument": "voice"},
                {"time": 2.0, "midi": 61, "instrument": "voice"},
                {"time": 30.0, "midi": 62, "instrument": "voice"},
            ]
        }
        self.assertEqual(long_voice_gaps(payload, 20.0), [(2.0, 30.0)])

    def test_application_removes_only_right_hand_in_gap(self):
        candidate = {"notes": group(13.0, "both") + group(29.0, "both")}
        source = {
            "notes": [
                {"time": 1.0, "midi": 60, "instrument": "voice"},
                {"time": 40.0, "midi": 62, "instrument": "voice"},
            ]
        }
        profile = {
            "id": "phase-test",
            "profileSha256": "phase-sha",
            "clock": {"originSeconds": 0.0, "pulseSeconds": 1.0, "subdivisions": 16},
            "selectedSlots": [13],
            "application": {
                "minimumGestureVelocity": 0.75,
                "minimumInstrumentalGapSeconds": 20.0,
            },
        }
        output, diagnostics = apply_profile(candidate, source, profile)
        self.assertEqual(diagnostics["rightNotesRemoved"], 2)
        self.assertEqual(len(output["notes"]), 2)
        self.assertTrue(all(note["hand"] == "left" for note in output["notes"]))


if __name__ == "__main__":
    unittest.main()
