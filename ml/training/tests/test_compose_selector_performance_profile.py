import copy
import unittest

from ml.training.compose_selector_performance_profile import (
    canonical_profile_hash,
    compose_selector_performance_profile,
)


class ComposeSelectorPerformanceProfileTest(unittest.TestCase):
    def setUp(self):
        self.selector = {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "selector-v1",
            "profileSha256": "selector-hash",
            "selectionModel": {"weights": [1.0]},
            "durationModel": {"weights": [2.0]},
            "routing": {"mode": "frozen"},
            "decoder": {"sourceDurationWeight": 0.5},
            "training": {},
        }
        self.performance = {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "touch-v2",
            "profileSha256": "touch-hash",
            "decoder": {
                "defaultPipeline": True,
                "defaultFullMix": {"mustNotCopy": True},
                "gestureDynamics": {
                    "enabled": True,
                    "pairedCalibration": {"weights": [3.0]},
                },
            },
        }

    def test_copies_only_gesture_layer_and_preserves_selector(self):
        original_selector = copy.deepcopy(self.selector)
        result = compose_selector_performance_profile(
            self.selector,
            self.performance,
            profile_id="hybrid-v3",
            created_at="2026-09-16T00:00:00+00:00",
        )
        self.assertEqual(self.selector, original_selector)
        self.assertEqual(result["selectionModel"], self.selector["selectionModel"])
        self.assertEqual(result["durationModel"], self.selector["durationModel"])
        self.assertEqual(result["routing"], self.selector["routing"])
        self.assertEqual(
            result["decoder"]["gestureDynamics"],
            self.performance["decoder"]["gestureDynamics"],
        )
        self.assertNotIn("defaultPipeline", result["decoder"])
        self.assertNotIn("defaultFullMix", result["decoder"])
        self.assertEqual(result["profileSha256"], canonical_profile_hash(result))

    def test_rejects_missing_enabled_gesture_layer(self):
        self.performance["decoder"]["gestureDynamics"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "gestureDynamics"):
            compose_selector_performance_profile(
                self.selector,
                self.performance,
                profile_id="invalid",
            )

    def test_embeds_default_pipeline_fallback_for_conditional_route(self):
        result = compose_selector_performance_profile(
            self.selector,
            self.performance,
            profile_id="conditional-v4",
            created_at="2026-09-16T00:00:00+00:00",
            conditional_route={
                "maximumVoiceRatio": 0.02,
                "minimumBassRatio": 0.5,
                "maximumPianoRatio": 0.08,
                "minimumSourceNotes": 64,
            },
        )
        route = result["decoder"]["conditionalLearnedRoute"]
        self.assertTrue(route["enabled"])
        self.assertTrue(route["fallbackDecoder"]["defaultPipeline"])
        self.assertIn("gestureDynamics", route["fallbackDecoder"])
        self.assertTrue(
            result["training"]["selectorPerformanceComposition"]["routingChanged"]
        )

    def test_embeds_separate_monophonic_fallback_without_full_mix_gesture_layer(self):
        result = compose_selector_performance_profile(
            self.selector,
            self.performance,
            profile_id="conditional-v5",
            created_at="2026-09-16T00:00:00+00:00",
            conditional_route={
                "maximumVoiceRatio": 0.02,
                "minimumBassRatio": 0.5,
                "maximumPianoRatio": 0.08,
                "minimumSourceNotes": 64,
            },
            monophonic_vocal_route={
                "minimumVoiceRatio": 0.95,
                "maximumBassRatio": 0.02,
                "maximumPianoRatio": 0.02,
                "minimumSourceNotes": 32,
                "fallbackProfileId": "monophonic-vocal-v1",
                "maximumFloorVelocity": 0.46,
                "minimumRunNotes": 6,
                "maximumRunGapSeconds": 0.35,
                "onsetDelaySeconds": 0.02,
            },
        )
        route = result["decoder"]["conditionalLearnedRoute"][
            "monophonicVocalRoute"
        ]
        self.assertEqual(route["fallbackProfileId"], "monophonic-vocal-v1")
        self.assertTrue(route["fallbackDecoder"]["defaultPipeline"])
        self.assertNotIn("gestureDynamics", route["fallbackDecoder"])
        self.assertEqual(
            route["fallbackDecoder"]["monophonicVocalCleanup"][
                "onsetDelaySeconds"
            ],
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
