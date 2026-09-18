import unittest

from ml.training.fit_paired_gesture_velocity import (
    cross_validate,
    fit_ridge_correction,
    match_gesture_groups,
    metric_summary,
    model_config,
    predict_examples,
    song_equal_weights,
)
from server.piano_gesture_calibration import GESTURE_CALIBRATION_FEATURE_NAMES


def note(midi, time, velocity=0.7):
    return {"midi": midi, "time": time, "duration": 0.2, "velocity": velocity}


class PairedGestureVelocityTests(unittest.TestCase):
    def test_match_requires_timing_and_pitch_class_support(self):
        reference = [[note(60, 0.0)], [note(62, 1.0)]]
        candidate = [[note(72, 0.02)], [note(63, 1.01)]]

        matched = match_gesture_groups(reference, candidate, 0.1, 1)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][2], 1)

    def test_ridge_correction_learns_a_repeated_soft_touch_bias(self):
        features = [0.0] * len(GESTURE_CALIBRATION_FEATURE_NAMES)
        features[0] = 1.0
        examples = []
        for song in ("one", "two"):
            for index in range(20):
                examples.append(
                    {
                        "songId": song,
                        "baseVelocity": 0.7,
                        "targetVelocity": 0.62,
                        "targetBand": "soft-0.55-0.64",
                        "features": list(features),
                    }
                )
        fitted = fit_ridge_correction(examples, 1.0)
        config = model_config(
            fitted,
            profile_id="test",
            blend=1.0,
            maximum_correction=0.12,
        )
        predicted = predict_examples(examples, config)

        self.assertLess(
            metric_summary(examples, predicted)["meanAbsoluteError"],
            metric_summary(examples)["meanAbsoluteError"],
        )
        self.assertAlmostEqual(predicted[0], 0.62, places=3)

    def test_cross_validation_reports_velocity_band_regression(self):
        examples = []
        for song_index, song in enumerate(("one", "two")):
            for index in range(20):
                base = 0.55 if index < 10 else 0.85
                target = base - 0.05 if index < 10 else base + 0.05
                features = [0.0] * len(GESTURE_CALIBRATION_FEATURE_NAMES)
                features[0] = 1.0
                features[1] = base
                features[2] = base * base
                examples.append(
                    {
                        "songId": song,
                        "baseVelocity": base,
                        "targetVelocity": target + song_index * 0.001,
                        "targetBand": "soft" if index < 10 else "accent",
                        "features": features,
                    }
                )

        result = cross_validate(examples, [1.0], [0.5], [0.08], "test")

        self.assertIn("worstVelocityBandRegression", result["best"])
        self.assertIn("velocityBandDeltas", result["best"]["folds"][0])

    def test_band_balance_gives_each_band_equal_weight_inside_each_song(self):
        examples = []
        for song in ("one", "two"):
            for index in range(9):
                examples.append(
                    {"songId": song, "targetBand": "common", "features": [1.0]}
                )
            examples.append(
                {"songId": song, "targetBand": "rare", "features": [1.0]}
            )

        weights = song_equal_weights(examples, band_balance=1.0)
        common = sum(
            weight
            for weight, example in zip(weights, examples)
            if example["songId"] == "one" and example["targetBand"] == "common"
        )
        rare = sum(
            weight
            for weight, example in zip(weights, examples)
            if example["songId"] == "one" and example["targetBand"] == "rare"
        )

        self.assertAlmostEqual(common, rare, places=9)


if __name__ == "__main__":
    unittest.main()
