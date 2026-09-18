import unittest

from piano_gesture_calibration import (
    GESTURE_CALIBRATION_FEATURE_NAMES,
    GESTURE_DURATION_FEATURE_NAMES,
    calibrated_gesture_duration,
    calibrated_gesture_velocity,
    gesture_calibration_features,
    gesture_duration_features,
    gesture_sequence_contexts,
)


class PianoGestureCalibrationTests(unittest.TestCase):
    def test_sequence_context_distinguishes_intro_and_outro(self):
        groups = [
            [{"midi": 60, "time": 0.0, "duration": 0.2, "arrangementRole": "melody"}],
            [{"midi": 64, "time": 1.0, "duration": 0.4, "arrangementRole": "harmony"}],
            [{"midi": 67, "time": 2.0, "duration": 0.6, "arrangementRole": "bass"}],
        ]
        contexts = gesture_sequence_contexts(groups, [0.4, 0.6, 0.8])

        self.assertEqual(len(contexts), 3)
        self.assertGreater(contexts[0]["intro_strength"], contexts[-1]["intro_strength"])
        self.assertLess(contexts[0]["outro_strength"], contexts[-1]["outro_strength"])
        self.assertEqual(contexts[0]["song_progress"], 0.0)
        self.assertEqual(contexts[-1]["song_progress"], 1.0)
        self.assertGreater(contexts[1]["local_4s_velocity_std"], 0.0)
        self.assertEqual(contexts[1]["previous_gap_seconds"], 1.0)
        self.assertEqual(contexts[1]["next_gap_seconds"], 1.0)

    def test_features_describe_one_shared_hammer_gesture(self):
        group = [
            {
                "midi": 48,
                "duration": 0.4,
                "hand": "left",
                "arrangementRole": "bass",
                "sourceInstrument": "electric_bass",
                "sourceVelocityBeforeArrangement": 0.4,
            },
            {
                "midi": 60,
                "duration": 0.2,
                "hand": "right",
                "arrangementRole": "melody",
                "sourceInstrument": "voice",
                "sourceVelocityBeforeArrangement": 0.8,
            },
        ]
        features = gesture_calibration_features(group, 0.7)
        self.assertEqual(set(features), set(GESTURE_CALIBRATION_FEATURE_NAMES))
        self.assertEqual(features["chord_size_2"], 1.0)
        self.assertEqual(features["melody_share"], 0.5)
        self.assertEqual(features["left_hand_share"], 0.5)
        self.assertAlmostEqual(features["source_velocity"], 0.6)

        duration_features = gesture_duration_features(
            group,
            0.3,
            {"next_gap_seconds": 0.25},
        )
        self.assertEqual(
            set(duration_features), set(GESTURE_DURATION_FEATURE_NAMES)
        )
        self.assertAlmostEqual(duration_features["melody_share"], 0.5)
        self.assertAlmostEqual(duration_features["duration_over_next_gap"], 1.2)

    def test_calibration_is_bounded_and_fails_closed_on_bad_contract(self):
        group = [{"midi": 60, "duration": 0.2, "velocity": 0.7}]
        config = {
            "enabled": True,
            "featureNames": ["bias"],
            "weights": [-1.0],
            "means": [0.0],
            "scales": [1.0],
            "blend": 1.0,
            "maximumCorrection": 0.1,
        }
        value, change = calibrated_gesture_velocity(
            group, 0.7, config, minimum_velocity=0.38, maximum_velocity=0.94
        )
        self.assertAlmostEqual(value, 0.6)
        self.assertAlmostEqual(change, -0.1)

        unchanged, change = calibrated_gesture_velocity(
            group,
            0.7,
            {**config, "featureNames": ["unknown"]},
            minimum_velocity=0.38,
            maximum_velocity=0.94,
        )
        self.assertEqual((unchanged, change), (0.7, 0.0))

    def test_duration_calibration_supports_articulation_ratio_and_gate(self):
        group = [
            {
                "midi": 60,
                "duration": 0.4,
                "velocity": 0.7,
                "arrangementRole": "melody",
            }
        ]
        config = {
            "enabled": True,
            "featureNames": ["bias"],
            "weights": [0.0],  # exp(0) = one next-onset interval.
            "means": [0.0],
            "scales": [1.0],
            "predictionMode": "articulation-ratio",
            "blend": 1.0,
            "maximumLogCorrection": 2.0,
        }
        value, change = calibrated_gesture_duration(
            group,
            0.4,
            config,
            context={"next_gap_seconds": 0.2},
        )
        self.assertAlmostEqual(value, 0.2)
        self.assertAlmostEqual(change, -0.2)

        gated = {
            **config,
            "minimumMelodyShare": 0.01,
            "maximumNonMelodyBaseDurationSeconds": 0.18,
        }
        harmony = [{**group[0], "arrangementRole": "harmony"}]
        self.assertEqual(
            calibrated_gesture_duration(
                harmony,
                0.4,
                gated,
                context={"next_gap_seconds": 0.2},
            ),
            (0.4, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
