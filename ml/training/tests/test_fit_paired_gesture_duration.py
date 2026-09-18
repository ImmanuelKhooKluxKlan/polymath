import unittest

from ml.training.fit_paired_gesture_duration import (
    duration_band,
    metrics,
)


class PairedGestureDurationTests(unittest.TestCase):
    def test_duration_bands_cover_short_medium_and_long_holds(self):
        self.assertEqual(duration_band(0.10), "short-<0.18")
        self.assertEqual(duration_band(0.30), "medium-0.18-0.54")
        self.assertEqual(duration_band(0.80), "long->=0.55")

    def test_metrics_reports_lower_error_for_better_prediction(self):
        examples = [
            {"baseDuration": 0.30, "targetDuration": 0.10, "targetBand": "short-<0.18"},
            {"baseDuration": 0.30, "targetDuration": 0.70, "targetBand": "long->=0.55"},
        ]
        baseline = metrics(examples)
        candidate = metrics(examples, [0.12, 0.68])
        self.assertLess(
            candidate["meanAbsoluteErrorSeconds"],
            baseline["meanAbsoluteErrorSeconds"],
        )


if __name__ == "__main__":
    unittest.main()
