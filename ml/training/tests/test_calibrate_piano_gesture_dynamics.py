import unittest

from ml.training.calibrate_piano_gesture_dynamics import (
    blend_chord_size_map,
    fit_chord_size_map,
    fit_quantile_map,
    gesture_rows,
    gesture_velocities,
    quantile,
)


class GestureDynamicsCalibrationTests(unittest.TestCase):
    def test_quantile_handles_exact_last_position(self):
        self.assertEqual(quantile([0.2, 0.5, 0.9], 1.0), 0.9)

    def test_groups_chord_notes_before_fitting_velocity_distribution(self):
        payload = {
            "notes": [
                {"time": 0.0, "velocity": 0.4},
                {"time": 0.01, "velocity": 0.4},
                {"time": 1.0, "velocity": 0.8},
            ]
        }
        self.assertEqual(gesture_velocities(payload, 0.035), [0.4, 0.8])

    def test_rejected_alignment_notes_never_enter_gesture_statistics(self):
        payload = {
            "notes": [
                {"time": 0.0, "midi": 60, "velocity": 0.4},
                {
                    "time": 1.0,
                    "midi": 62,
                    "velocity": 0.99,
                    "trainingEligible": False,
                },
            ]
        }
        self.assertEqual(gesture_rows(payload, 0.035), [{"velocity": 0.4, "size": 1}])

    def test_fits_equal_per_song_monotonic_knots(self):
        result = fit_quantile_map(
            [([0.2, 0.8], 1.0), ([0.4, 1.0], 1.0)], (0.0, 0.5, 1.0)
        )
        self.assertEqual(result, {"0.0": 0.3, "0.5": 0.6, "1.0": 0.9})

    def test_chord_size_fit_weights_songs_equally_and_uses_clean_fallbacks(self):
        result = fit_chord_size_map(
            [
                ([{"size": 1, "velocity": 0.4}], 1.0),
                ([{"size": 1, "velocity": 0.8}], 1.0),
            ]
        )
        self.assertEqual(result["1"], 0.6)
        self.assertEqual(result["6"], 0.86)

    def test_chord_size_blend_can_preserve_listening_winner_touch(self):
        existing = {"1": 0.68, "2": 0.67, "3": 0.71}
        fitted = {str(size): 0.9 for size in range(1, 7)}

        preserved = blend_chord_size_map(existing, fitted, 0.0)
        halfway = blend_chord_size_map(existing, fitted, 0.5)

        self.assertEqual(preserved["2"], 0.67)
        self.assertEqual(preserved["6"], 0.86)
        self.assertEqual(halfway["2"], 0.785)

    def test_chord_size_blend_can_change_only_large_accent_gestures(self):
        existing = {str(size): 0.6 + size * 0.01 for size in range(1, 7)}
        fitted = {str(size): 0.9 for size in range(1, 7)}

        result = blend_chord_size_map(
            existing,
            fitted,
            1.0,
            minimum_fitted_size=4,
        )

        self.assertEqual(result["1"], existing["1"])
        self.assertEqual(result["3"], existing["3"])
        self.assertEqual(result["4"], 0.9)
        self.assertEqual(result["6"], 0.9)


if __name__ == "__main__":
    unittest.main()
