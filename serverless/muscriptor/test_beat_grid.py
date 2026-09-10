import unittest

from serverless.muscriptor.beat_grid import apply_onset_delay, normalize_beat_grid


class BeatGridTests(unittest.TestCase):
    def test_normalizes_detected_tempo_and_aligns_onsets(self):
        grid = normalize_beat_grid({
            "bpm": 100.25,
            "beats_per_bar": 4,
            "first_downbeat": 0.12,
            "onset_delay": 0.08,
        })
        notes = [{"time": 0.18}, {"time": 1.08}]

        applied = apply_onset_delay(notes, grid)

        self.assertEqual(grid["bpm"], 100.25)
        self.assertEqual(grid["beatsPerBar"], 4)
        self.assertEqual(applied, 0.08)
        self.assertEqual(notes, [{"time": 0.1}, {"time": 1.0}])

    def test_rejects_unusable_grid(self):
        self.assertIsNone(normalize_beat_grid({"bpm": 0, "beats_per_bar": 4}))
        self.assertIsNone(normalize_beat_grid({"bpm": 120, "beats_per_bar": 0}))

    def test_keeps_tempo_when_meter_is_unknown_and_supports_early_onsets(self):
        grid = normalize_beat_grid({"bpm": 91, "beats_per_bar": None, "onset_delay": -0.02})
        notes = [{"time": 1.0}]

        apply_onset_delay(notes, grid)

        self.assertIsNone(grid["beatsPerBar"])
        self.assertEqual(notes[0]["time"], 1.02)


if __name__ == "__main__":
    unittest.main()
