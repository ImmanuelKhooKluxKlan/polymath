import unittest

from ml.training.apply_onset_gesture_profile import (
    EXPECTED_MODEL_TYPE,
    apply_onset_profile,
    blend_standardized_logistic_models,
    profile_sha256,
)


class ApplyOnsetGestureProfileTests(unittest.TestCase):
    def test_replaces_only_onset_model_and_records_provenance(self):
        base = {
            "id": "base",
            "decoder": {
                "leftHandAccompaniment": {
                    "enabled": True,
                    "targetOnsetKeepRatio": 0.86,
                    "selectionModel": {"type": "note-model"},
                    "onsetSelectionModel": {"type": EXPECTED_MODEL_TYPE, "weights": [0]},
                }
            },
            "training": {"existing": True},
            "profileSha256": "old",
        }
        onset = {
            "id": "new-onset",
            "selectionModel": {"type": EXPECTED_MODEL_TYPE, "weights": [1]},
            "recommendedTargetOnsetKeepRatio": 0.6,
            "profileSha256": "onset-hash",
        }

        result = apply_onset_profile(
            base,
            onset,
            profile_id="candidate",
            onset_profile_path="onset.json",
            validation_report_path="report.json",
            created_at="fixed-time",
        )

        self.assertEqual(result["id"], "candidate")
        self.assertEqual(
            result["decoder"]["leftHandAccompaniment"]["onsetSelectionModel"]["weights"],
            [1],
        )
        self.assertEqual(
            result["decoder"]["leftHandAccompaniment"]["selectionModel"],
            {"type": "note-model"},
        )
        self.assertEqual(
            result["decoder"]["leftHandAccompaniment"]["targetOnsetKeepRatio"],
            0.86,
        )
        self.assertEqual(
            result["training"]["onsetGestureUpgrade"]["decision"],
            "EXPERIMENTAL_ONLY",
        )
        self.assertEqual(result["profileSha256"], profile_sha256(result))
        self.assertEqual(base["profileSha256"], "old")

    def test_can_override_arranger_stage_keep_ratio(self):
        result = apply_onset_profile(
            {"decoder": {"leftHandAccompaniment": {"enabled": True}}},
            {"selectionModel": {"type": EXPECTED_MODEL_TYPE}},
            profile_id="candidate",
            onset_profile_path="onset.json",
            target_onset_keep_ratio=0.75,
            created_at="fixed-time",
        )
        self.assertEqual(
            result["decoder"]["leftHandAccompaniment"]["targetOnsetKeepRatio"],
            0.75,
        )

    def test_blends_logits_after_removing_different_standardizations(self):
        base = {
            "type": EXPECTED_MODEL_TYPE,
            "featureNames": ["bias", "velocity"],
            "weights": [0.2, 0.5],
            "means": [0.0, 0.7],
            "scales": [1.0, 0.1],
            "threshold": 0.5,
        }
        new = {
            "type": EXPECTED_MODEL_TYPE,
            "featureNames": ["bias", "velocity"],
            "weights": [-0.1, 0.8],
            "means": [0.0, 0.5],
            "scales": [1.0, 0.2],
            "threshold": 0.5,
        }

        result = blend_standardized_logistic_models(base, new, 0.75)
        features = [1.0, 0.82]

        def logit(model):
            return sum(
                weight * ((value - mean) / scale)
                for weight, value, mean, scale in zip(
                    model["weights"], features, model["means"], model["scales"]
                )
            )

        self.assertAlmostEqual(logit(result), 0.75 * logit(base) + 0.25 * logit(new))
        self.assertEqual(result["blend"]["newShare"], 0.25)


if __name__ == "__main__":
    unittest.main()
