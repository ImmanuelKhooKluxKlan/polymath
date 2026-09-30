from __future__ import annotations

import unittest

from ml.training.analyze_multiview_evidence import (
    evidence_counts,
    primary_preserving_union,
)


class AnalyzeMultiviewEvidenceTests(unittest.TestCase):
    def test_union_can_recover_complementary_pitch_classes(self) -> None:
        reference = [
            {"midi": 60, "time": 0.0, "duration": 0.2},
            {"midi": 64, "time": 1.0, "duration": 0.2},
        ]
        first = [{"midi": 72, "time": 0.01, "duration": 0.2}]
        second = [{"midi": 76, "time": 1.01, "duration": 0.2}]

        first_metrics = evidence_counts(reference, first, 0.05)
        union_metrics = evidence_counts(reference, first + second, 0.05)

        self.assertEqual(first_metrics["pitchClassRecallCeiling"], 0.5)
        self.assertEqual(union_metrics["pitchClassRecallCeiling"], 1.0)
        self.assertEqual(union_metrics["exactPitchRecallCeiling"], 0.0)

    def test_added_dense_view_cannot_reduce_primary_recall(self) -> None:
        reference = [
            {"midi": 60, "time": 0.0, "duration": 0.2},
            {"midi": 60, "time": 0.2, "duration": 0.2},
        ]
        primary = [
            {"midi": 60, "time": 0.0, "duration": 0.2},
            {"midi": 60, "time": 0.2, "duration": 0.2},
        ]
        noisy = [
            {"midi": 60, "time": 0.01, "duration": 0.2},
            {"midi": 60, "time": 0.02, "duration": 0.2},
            {"midi": 61, "time": 0.1, "duration": 0.2},
        ]
        union = primary_preserving_union({"primary": primary, "noisy": noisy}, "primary")

        primary_score = evidence_counts(reference, primary, 0.05)
        union_score = evidence_counts(reference, union, 0.05)

        self.assertGreaterEqual(
            union_score["pitchClassRecallCeiling"],
            primary_score["pitchClassRecallCeiling"],
        )
        self.assertEqual(union_score["pitchClassRecallCeiling"], 1.0)


if __name__ == "__main__":
    unittest.main()
