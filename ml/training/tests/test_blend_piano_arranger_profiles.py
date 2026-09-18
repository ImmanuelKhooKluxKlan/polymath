import math
import unittest

from ml.training.blend_piano_arranger_profiles import blend_selection_models, compose_profile


def logit(model, row):
    return sum(
        weight * ((value - mean) / scale)
        for weight, value, mean, scale in zip(
            model["weights"], row, model["means"], model["scales"]
        )
    )


class BlendPianoArrangerProfilesTest(unittest.TestCase):
    def test_append_only_context_contract_preserves_old_model_logits(self):
        old = {
            "featureNames": ["bias", "pitch"],
            "weights": [0.2, 0.8],
            "means": [0.0, 0.5],
            "scales": [1.0, 0.25],
            "threshold": 0.4,
        }
        context = {
            "featureNames": ["bias", "pitch", "voice_near"],
            "weights": [-0.1, 0.3, 0.9],
            "means": [0.0, -0.2, 0.7],
            "scales": [1.0, 1.5, 0.5],
            "threshold": 0.7,
        }
        row = [1.0, 0.35, 0.91]

        lifted_old = blend_selection_models(old, context, 1.0)

        self.assertEqual(lifted_old["featureNames"], context["featureNames"])
        self.assertAlmostEqual(logit(lifted_old, row), logit(old, row[:2]), places=9)
        self.assertEqual(lifted_old["weights"][-1], 0.0)

    def test_blended_raw_coefficients_equal_the_weighted_source_logits(self):
        names = ["bias", "pitch", "density"]
        base = {
            "featureNames": names,
            "weights": [0.2, 0.8, -0.4],
            "means": [0.0, 0.5, 0.1],
            "scales": [1.0, 0.25, 2.0],
            "threshold": 0.4,
        }
        other = {
            "featureNames": names,
            "weights": [-0.1, 0.3, 0.9],
            "means": [0.0, -0.2, 0.7],
            "scales": [1.0, 1.5, 0.5],
            "threshold": 0.7,
        }
        row = [1.0, 0.35, 1.2]
        share = 0.73

        blended = blend_selection_models(base, other, share)

        expected = share * logit(base, row) + (1.0 - share) * logit(other, row)
        self.assertTrue(math.isclose(logit(blended, row), expected, abs_tol=1e-9))
        self.assertAlmostEqual(blended["threshold"], 0.481, places=6)

    def test_candidate_controls_are_frozen_into_the_composed_profile(self):
        names = ["bias", "pitch"]
        model = {
            "featureNames": names,
            "weights": [0.2, 0.8],
            "means": [0.0, 0.5],
            "scales": [1.0, 0.25],
            "threshold": 0.4,
        }
        base = {"id": "base", "selectionModel": model, "decoder": {}}
        other = {
            "id": "other",
            "selectionModel": model,
            "durationModel": {"predictionWeight": 0.1},
            "registerModel": {
                "type": "hierarchical-categorical-octave-shift-v1",
                "groups": {},
            },
        }

        result = compose_profile(
            base,
            other,
            base_share=0.5,
            profile_id="candidate",
            take_duration_from_other=True,
            take_register_from_other=True,
            density_multiplier=1.8,
            duration_prediction_weight=0.2,
            preferred_global_shift_semitones=7,
            source_duration_weight=0.8,
            minimum_rendered_duration_seconds=0.14,
            adaptive_source_density={
                'lowSourceNotesPerSecond': 20,
                'highSourceNotesPerSecond': 30,
                'lowDensityMultiplier': 1.8,
                'highDensityMultiplier': 2.0,
                'lowDurationPredictionWeight': 0.0,
                'highDurationPredictionWeight': 0.4,
                'minimumVoiceRatioForDensityAdaptation': 0.05,
                'nonVocalDensityMultiplier': 1.45,
                'minimumVoiceRatioForDisablingSparseExpansion': 0.05,
                'maximumPianoRatioForVocalAdaptation': 0.05,
                'lowQuotaBackfillRatio': 1.0,
                'highQuotaBackfillRatio': 0.65,
                'nonVocalQuotaBackfillRatio': 1.0,
            },
            adaptive_selection_blend={
                'lowSourceNotesPerSecond': 20,
                'highSourceNotesPerSecond': 30,
                'lowSourceBaseShare': 0.95,
                'highSourceBaseShare': 0.80,
                'minimumVoiceRatioForAggressiveBlend': 0.05,
            },
        )

        self.assertEqual(result["decoder"]["preCleanupDensityMultiplier"], 1.8)
        self.assertEqual(result["durationModel"]["predictionWeight"], 0.2)
        self.assertEqual(result["residualBlend"]["durationPredictionWeight"], 0.2)
        self.assertEqual(
            result["registerModel"]["type"],
            "hierarchical-categorical-octave-shift-v1",
        )
        self.assertEqual(result["residualBlend"]["registerFrom"], "other")
        self.assertEqual(result["decoder"]["preferredGlobalRegisterShiftSemitones"], 12)
        self.assertEqual(
            result["residualBlend"]["preferredGlobalRegisterShiftSemitones"], 12
        )
        self.assertEqual(result["decoder"]["sourceDurationWeight"], 0.8)
        self.assertEqual(result["residualBlend"]["sourceDurationWeight"], 0.8)
        self.assertEqual(result["decoder"]["minimumRenderedDurationSeconds"], 0.14)
        self.assertEqual(
            result["residualBlend"]["minimumRenderedDurationSeconds"], 0.14
        )
        self.assertEqual(result["training"]["baseProfile"]["id"], "base")
        self.assertEqual(result["training"]["otherProfile"]["id"], "other")
        self.assertEqual(result["training"]["method"], "raw-feature-logit-residual-blend")
        self.assertTrue(result['decoder']['adaptiveSourceDensity']['enabled'])
        self.assertEqual(
            result['decoder']['adaptiveSourceDensity'][
                'minimumVoiceRatioForDensityAdaptation'
            ],
            0.05,
        )
        self.assertEqual(
            result['decoder']['adaptiveSourceDensity'][
                'nonVocalDensityMultiplier'
            ],
            1.45,
        )
        self.assertEqual(
            result['decoder']['adaptiveSourceDensity'][
                'minimumVoiceRatioForDisablingSparseExpansion'
            ],
            0.05,
        )
        self.assertEqual(
            result['decoder']['adaptiveSourceDensity'][
                'maximumPianoRatioForVocalAdaptation'
            ],
            0.05,
        )
        self.assertEqual(
            result['decoder']['adaptiveSourceDensity'][
                'highQuotaBackfillRatio'
            ],
            0.65,
        )
        selection_policy = result['decoder']['adaptiveSelectionBlend']
        self.assertTrue(selection_policy['enabled'])
        self.assertEqual(selection_policy['lowSourceBaseShare'], 0.95)
        self.assertEqual(selection_policy['highSourceBaseShare'], 0.8)
        self.assertEqual(
            selection_policy['minimumVoiceRatioForAggressiveBlend'], 0.05
        )
        self.assertEqual(
            selection_policy['lowSourceSelectionModel']['means'], [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
