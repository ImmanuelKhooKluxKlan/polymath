import unittest

from ml.training.apply_gesture_note_pruner import profile_with_overrides


class ApplyGestureNotePrunerCliTests(unittest.TestCase):
    def test_overrides_do_not_mutate_the_frozen_profile(self) -> None:
        profile = {
            "threshold": 0.7,
            "minimumProbabilityGap": 0.15,
            "maximumRemovalsPerGesture": 1,
            "minimumGroupSize": 5,
        }
        candidate = profile_with_overrides(
            profile,
            threshold=0.75,
            maximum_removals_per_gesture=2,
            minimum_group_size=4,
        )
        self.assertEqual(profile["threshold"], 0.7)
        self.assertEqual(candidate["threshold"], 0.75)
        self.assertEqual(candidate["maximumRemovalsPerGesture"], 2)
        self.assertEqual(candidate["minimumGroupSize"], 4)

    def test_rejects_invalid_group_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            profile_with_overrides({}, minimum_group_size=1)


if __name__ == "__main__":
    unittest.main()
