import unittest

from ml.training.search_pianist_pitch_class_application import (
    application_paths,
    is_song_safe,
    trial_selection_score,
)


def safe_gates():
    return {
        "duration_error_not_over_10_percent_worse": True,
        "severe_cutoff_rate_not_worse": True,
        "visual_duration_error_not_over_10_percent_worse": True,
        "visual_cutoff_rate_not_worse": True,
        "physical_duration_error_not_over_10_percent_worse": True,
        "physical_cutoff_rate_not_worse": True,
        "rapid_retriggers_not_worse": True,
    }


class PitchClassApplicationSearchTest(unittest.TestCase):
    def test_baseline_can_be_compared_to_a_shared_predecoder_start(self):
        control, starting = application_paths(
            {"baseline": "production.json", "candidate": "shared-base.json"}
        )
        self.assertEqual(control.name, "production.json")
        self.assertEqual(starting.name, "shared-base.json")

    def test_legacy_manifest_uses_candidate_for_both_paths(self):
        control, starting = application_paths({"candidate": "current.json"})
        self.assertEqual(control, starting)

    def test_song_safety_rejects_a_pitch_class_regression(self):
        deltas = {
            "quality": 0.01,
            "exactF1_100ms": 0.01,
            "exactF1_250ms": 0.01,
            "pitchClassF1_250ms": -0.001,
            "pitchClassRecall_250ms": 0.01,
        }
        self.assertFalse(is_song_safe(deltas, safe_gates()))

    def test_song_safety_allows_only_half_point_rounding_tolerance(self):
        deltas = {
            "quality": -0.0005,
            "exactF1_100ms": 0.0,
            "exactF1_250ms": 0.0,
            "pitchClassF1_250ms": 0.0,
            "pitchClassRecall_250ms": 0.0,
        }
        self.assertTrue(is_song_safe(deltas, safe_gates()))

    def test_selection_score_rewards_exact_and_pitch_class_gains(self):
        neutral = {
            "weight": 1.0,
            "deltas": {
                "quality": 0.0,
                "exactF1_100ms": 0.0,
                "exactF1_250ms": 0.0,
                "pitchClassF1_250ms": 0.0,
                "pitchClassRecall_250ms": 0.0,
                "chordSizeDistance": 0.0,
                "handOccupancyDistance": 0.0,
            },
        }
        improved = {
            "weight": 1.0,
            "deltas": {
                **neutral["deltas"],
                "quality": 0.01,
                "exactF1_100ms": 0.02,
                "pitchClassF1_250ms": 0.01,
            },
        }
        self.assertGreater(
            trial_selection_score([improved]), trial_selection_score([neutral])
        )

    def test_zero_delta_is_not_a_material_gain(self):
        neutral = {
            "weight": 1.0,
            "deltas": {
                "quality": 0.0,
                "exactF1_100ms": 0.0,
                "exactF1_250ms": 0.0,
                "pitchClassF1_250ms": 0.0,
                "pitchClassRecall_250ms": 0.0,
                "chordSizeDistance": 0.0,
                "handOccupancyDistance": 0.0,
            },
        }
        self.assertEqual(trial_selection_score([neutral]), 0.0)
