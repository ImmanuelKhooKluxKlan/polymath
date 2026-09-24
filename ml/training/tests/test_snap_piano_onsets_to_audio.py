import unittest

import numpy as np

from ml.training.snap_piano_onsets_to_audio import (
    candidate_groups,
    local_peak_indices,
    snap_payload,
)
from ml.training.search_audio_onset_snap_loso import promotion_decision


class PianoAudioOnsetSnapTests(unittest.TestCase):
    def test_detects_local_peaks(self):
        values = np.asarray([0.0, 0.5, 0.2, 0.8, 0.1])
        self.assertEqual(local_peak_indices(values).tolist(), [1, 3])

    def test_groups_a_chord_but_not_next_attack(self):
        notes = [{"time": 1.0}, {"time": 1.02}, {"time": 1.10}]
        self.assertEqual(candidate_groups(notes, 0.035), [[0, 1], [2]])

    def test_snaps_timing_without_changing_music_content(self):
        payload = {
            "notes": [
                {"midi": 60, "time": 1.08, "duration": 0.4, "velocity": 0.7},
                {"midi": 64, "time": 1.10, "duration": 0.3, "velocity": 0.6},
            ]
        }
        times = np.asarray([0.9, 1.0, 1.1, 1.2])
        strengths = np.asarray([0.0, 0.9, 0.1, 0.0])
        output, diagnostics = snap_payload(
            payload,
            times,
            strengths,
            radius_seconds=0.12,
            minimum_peak_strength=0.5,
            distance_weight=0.1,
            peak_time_offset_seconds=0.0,
            minimum_strength_gain=0.0,
        )
        self.assertAlmostEqual(output["notes"][0]["time"], 1.0)
        self.assertAlmostEqual(output["notes"][1]["time"], 1.02)
        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64])
        self.assertEqual([note["duration"] for note in output["notes"]], [0.4, 0.3])
        self.assertEqual([note["velocity"] for note in output["notes"]], [0.7, 0.6])
        self.assertEqual(diagnostics["snappedGroups"], 1)

    def test_rejects_a_loso_regression(self):
        decision, gates = promotion_decision(
            {
                "baselineExactPitchOnset100msF1": 0.86,
                "candidateExactPitchOnset100msF1": 0.85,
                "baselineExactPitchOnset50msF1": 0.73,
                "candidateExactPitchOnset50msF1": 0.72,
                "baselineExactPitchOnset250msF1": 0.91,
                "candidateExactPitchOnset250msF1": 0.91,
                "allHeldoutExact100NonRegressing": False,
            }
        )
        self.assertEqual(decision, "REJECT_NO_CROSS_SONG_TIMING_GAIN")
        self.assertFalse(gates["exactPitchOnset100msImproves"])


if __name__ == "__main__":
    unittest.main()
