from __future__ import annotations

import unittest

from ml.training.analyze_arranger_error_budget import (
    greedy_onset_matches,
    perfect_pruned_metrics,
    stage_metrics,
)


def note(midi: int, time: float) -> dict[str, float | int]:
    return {
        "midi": midi,
        "time": time,
        "duration": 0.4,
        "visualDuration": 0.4,
        "audioDuration": 0.4,
        "velocity": 0.7,
    }


class ArrangerErrorBudgetTests(unittest.TestCase):
    def test_onset_matching_is_one_to_one_and_ignores_pitch(self) -> None:
        reference = [note(60, 1.0), note(64, 1.1)]
        observed = [note(72, 1.04)]

        matches = greedy_onset_matches(reference, observed, 0.10)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1]["midi"], 72)

    def test_perfect_pruned_ceiling_does_not_count_raw_false_positives(self) -> None:
        metrics = perfect_pruned_metrics(10, 8)

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.8)
        self.assertEqual(metrics["f1"], 0.888889)

    def test_stage_budget_separates_pitch_class_from_register(self) -> None:
        reference = [note(60, 1.0), note(64, 2.0)]
        raw = [note(72, 1.0), note(64, 2.0), note(67, 3.0)]
        candidate = [note(72, 1.0), note(65, 2.0)]

        report = stage_metrics(reference, raw, candidate, 0.10)

        self.assertEqual(report["rawEvidence"]["pitchClassRecallCeiling"], 1.0)
        self.assertEqual(report["rawEvidence"]["exactPitchRecallCeiling"], 0.5)
        self.assertEqual(report["candidate"]["onset"]["f1"], 1.0)
        self.assertEqual(report["candidate"]["pitchClass"]["f1"], 0.5)
        self.assertEqual(report["candidate"]["exactPitch"]["f1"], 0.0)
        self.assertEqual(
            report["approximateReferenceErrorCounts"][
                "candidatePitchClassButWrongRegister"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
