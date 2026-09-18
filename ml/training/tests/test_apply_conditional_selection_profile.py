import unittest

from ml.training.apply_conditional_selection_profile import (
    apply_conditional_selector,
    profile_sha256,
)


class ApplyConditionalSelectionProfileTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "base",
            "selectionModel": {
                "featureNames": ["bias"],
                "weights": [0.0],
                "means": [0.0],
                "scales": [1.0],
                "threshold": 0.5,
            },
            "decoder": {},
            "profileSha256": "old",
        }
        self.alternative = {
            "id": "contextual",
            "profileSha256": "alternate-hash",
            "selectionModel": {
                "featureNames": ["bias", "local_density"],
                "weights": [0.2, -0.4],
                "means": [0.0, 0.3],
                "scales": [1.0, 0.2],
                "threshold": 0.61,
            },
        }

    def test_installs_bounded_model_without_mutating_inputs(self):
        result = apply_conditional_selector(
            self.base,
            self.alternative,
            profile_id="candidate",
            alternative_profile_path="alternate.json",
            alternative_share=0.35,
            source_families=["Guitar", "guitar"],
            minimum_source_midi=60,
            maximum_source_midi=84,
            preserve_base_window_counts=True,
            created_at="2026-09-14T00:00:00+00:00",
        )
        policy = result["decoder"]["conditionalSelectionBlend"]
        self.assertEqual(result["id"], "candidate")
        self.assertEqual(policy["sourceFamilies"], ["guitar"])
        self.assertEqual(policy["alternativeShare"], 0.35)
        self.assertEqual(policy["selectionModel"]["threshold"], 0.61)
        self.assertTrue(policy["preserveBaseWindowCounts"])
        self.assertFalse(policy["preserveBaseOnsetCounts"])
        self.assertNotIn("conditionalSelectionBlend", self.base["decoder"])
        self.assertEqual(result["profileSha256"], profile_sha256(result))

    def test_rejects_an_invalid_range(self):
        with self.assertRaisesRegex(ValueError, "MIDI range"):
            apply_conditional_selector(
                self.base,
                self.alternative,
                profile_id="candidate",
                alternative_profile_path="alternate.json",
                alternative_share=0.5,
                source_families=["guitar"],
                minimum_source_midi=90,
                maximum_source_midi=60,
            )


if __name__ == "__main__":
    unittest.main()
